from dataclasses import replace

import jax
import jax.numpy as jnp

from adze_t.backends.deterministic import DeterministicOps
from adze_t.config import REFERENCE_SMALL_V0
from adze_t.decoder import apply_decoder
from adze_t.dit import DiTConfig, apply_dit, apply_dit_cycle, apply_rope, init_dit_params
from adze_t.encoder import encode_target
from adze_t.mamba import MambaConfig, apply_mamba_stack, init_mamba_stack
from adze_t.model import apply_model, init_model_params
from adze_t.objectives import cross_entropy, loss_components, total_loss
from adze_t.packing import build_pack_metadata_core, pack_values
from adze_t.proposal import apply_proposal
from adze_t.teacher import canonical_teacher_structure
from adze_t.training import initialise_training, make_fixed_structure_batch, train_step


def _tiny_config(*, cycles: int = 2, physical_blocks: int = 2):
    cfg = REFERENCE_SMALL_V0
    return replace(
        cfg,
        carrier=replace(cfg.carrier, C=8, h_dim=16, L_max=2),
        packing=replace(cfg.packing, M_max=4, K=2),
        model=replace(
            cfg.model,
            d_front=8,
            d_ctx=16,
            frontend_layers=1,
            context_layers=1,
            target_layers=1,
            proposal_layers=1,
            proposal_hidden_dim=8,
            d_model=16,
            heads=2,
            head_dim=8,
            ffn_hidden=32,
            physical_blocks_L=physical_blocks,
            cycles_Q=cycles,
            d_dec=16,
            decoder_layers=1,
            mamba_expand=1,
            mamba_state_dim=4,
        ),
    )


class CountingOps(DeterministicOps):
    def __init__(self):
        self.calls: list[str] = []

    def linear(self, x, params, *, name="linear"):
        self.calls.append(name)
        return super().linear(x, params, name=name)

    def embedding(self, indices, params, *, name):
        self.calls.append(name)
        return super().embedding(indices, params, name=name)

    def depthwise_conv1d(self, x, params, *, name):
        self.calls.append(name)
        return super().depthwise_conv1d(x, params, name=name)

    def parameter(self, value, *, name):
        self.calls.append(name)
        return super().parameter(value, name=name)


def test_full_graph_really_routes_learned_operations_through_backend():
    cfg = _tiny_config()
    params = init_model_params(jax.random.PRNGKey(0), cfg)
    values = jnp.arange(1, 9, dtype=jnp.int32)[None]
    mask = jnp.ones_like(values, dtype=bool)
    ops = CountingOps()
    apply_model(params, values, mask, values, mask, config=cfg, ops=ops)
    required = (
        "frontend.byte_embed",
        "context.input_proj",
        "target.h",
        "proposal.prior",
        "model.carrier_input",
        "dit.input_proj",
        "dit.conditioning_trunk",
        "dit.block_0.q",
        "dit.block_1.ffn_down",
        "model.carrier_output",
        "model.h_head",
        "model.b_head",
        "model.l_head",
        "decoder.h",
        "decoder.stack.layer_0.dbc_proj",
        "decoder.out",
    )
    assert all(name in ops.calls for name in required)
    assert any(name.endswith(".conv") for name in ops.calls)
    assert any(name.endswith(".a_log") for name in ops.calls)


def test_teacher_partition_is_generic_and_separate_from_boundaries():
    cfg = REFERENCE_SMALL_V0
    values = jnp.arange(1, 12, dtype=jnp.int32)[None]
    teacher = canonical_teacher_structure(values, jnp.ones_like(values, dtype=bool), cfg)
    assert teacher.length[0, :4].tolist() == [4, 4, 3, 0]
    assert teacher.activity[0, :4].tolist() == [True, True, True, False]
    assert jnp.array_equal(teacher.slot_bytes[0, :3].reshape(-1)[:11], values[0])
    assert jnp.where(teacher.boundaries[0])[0].tolist() == [7, 15, 23, 31]

    eight = canonical_teacher_structure(values[:, :8], jnp.ones((1, 8), dtype=bool), cfg)
    assert eight.length[0, :4].tolist() == [4, 4, 0, 0]
    assert jnp.array_equal(eight.boundaries, teacher.boundaries)


def test_selective_ssm_is_finite_causal_and_uses_configured_depth():
    ops = DeterministicOps()
    cfg = MambaConfig(width=8, layers=2, expand=1, state_dim=4, conv_kernel=3)
    params = init_mamba_stack(jax.random.PRNGKey(1), cfg, ops, name="test")
    assert len(params) == 2
    x = jax.random.normal(jax.random.PRNGKey(2), (1, 6, 8))
    output = apply_mamba_stack(x, params, cfg, ops, name="test")
    perturbed = x.at[:, -1].add(100.0)
    changed = apply_mamba_stack(perturbed, params, cfg, ops, name="test")
    assert bool(jnp.all(jnp.isfinite(output)))
    assert jnp.allclose(output[:, :-1], changed[:, :-1], atol=1.0e-6)
    assert not jnp.allclose(output[:, -1], changed[:, -1])
    assert all(jnp.all(layer["a_log"] >= 0.0) for layer in params)


def _dit_inputs(*, inactive=False):
    boundaries = jnp.array([[0, 1, 0, 1, 0, 1]], dtype=jnp.int32)
    activity = jnp.ones_like(boundaries)
    if inactive:
        activity = activity.at[:, 2].set(0)
    metadata = build_pack_metadata_core(boundaries, activity, M_max=3, K=2)
    values = jax.random.normal(jax.random.PRNGKey(3), (1, 6, 16))
    packed = pack_values(values, metadata)
    cfg = DiTConfig(
        d_model=16,
        heads=2,
        head_dim=8,
        ffn_hidden=32,
        physical_blocks=2,
        cycles=3,
        carrier_capacity=6,
        d_context=16,
        max_blocks=3,
        max_slots=2,
    )
    params = init_dit_params(jax.random.PRNGKey(4), cfg)
    context = jax.random.normal(jax.random.PRNGKey(5), (1, 16))
    return packed, metadata, cfg, params, context


def _leaf_count(tree):
    return sum(x.size for x in jax.tree_util.tree_leaves(tree))


def test_dit_tying_depth_modulation_and_q_independent_parameter_count():
    packed, metadata, cfg, params, context = _dit_inputs()
    _, aux = apply_dit(packed, metadata, context, params, cfg)
    assert aux["effective_depths"].tolist() == [0, 1, 2, 3, 4, 5]
    assert params["blocks"][0] is not params["blocks"][1]
    assert params["blocks"][0]["modulation"] is not params["blocks"][1]["modulation"]
    assert not jnp.array_equal(
        params["blocks"][0]["q"]["weight"], params["blocks"][1]["q"]["weight"]
    )
    assert _leaf_count(params) == _leaf_count(
        init_dit_params(jax.random.PRNGKey(4), replace(cfg, cycles=7))
    )
    out_one, aux_one = apply_dit(packed, metadata, context, params, cfg, cycles=1)
    out_three, aux_three = apply_dit(packed, metadata, context, params, cfg, cycles=3)
    assert aux_one["trajectory"].shape[0] == 1
    assert aux_three["trajectory"].shape[0] == 3
    assert not jnp.allclose(out_one, out_three)


def test_dit_manual_unroll_matches_recurrent_driver():
    packed, metadata, cfg, params, context = _dit_inputs()
    output, aux = apply_dit(packed, metadata, context, params, cfg)
    batch, blocks, slots, _ = packed.shape
    query = metadata.query_mask.reshape(batch, blocks * slots)
    kv = metadata.kv_mask.reshape(batch, blocks * slots)
    block_id = metadata.block_id.reshape(batch, blocks * slots)
    carrier_id = jnp.maximum(metadata.carrier_id.reshape(batch, blocks * slots), 0)
    within = metadata.within_block_pos.reshape(batch, blocks * slots)
    ops = DeterministicOps()
    x = ops.linear(
        packed.reshape(batch, blocks * slots, -1), params["input_proj"], name="dit.input_proj"
    )
    x = (
        x
        + ops.embedding(carrier_id, params["carrier_embed"], name="dit.carrier_embed")
        + ops.embedding(block_id, params["block_embed"], name="dit.block_embed")
        + ops.embedding(within, params["within_embed"], name="dit.within_embed")
        + ops.embedding(jnp.zeros_like(carrier_id), params["length_embed"], name="dit.length_embed")
        + ops.embedding(
            jnp.where(carrier_id == 0, 2, 0),
            params["boundary_left_embed"],
            name="dit.boundary_left_embed",
        )
        + ops.embedding(
            jnp.zeros_like(carrier_id),
            params["boundary_right_embed"],
            name="dit.boundary_right_embed",
        )
    )
    x = jnp.where(query[..., None], x, 0.0)
    mask = aux["mask"]
    for cycle in range(cfg.cycles):
        x, _, _, _ = apply_dit_cycle(
            x,
            params,
            context,
            mask,
            carrier_id,
            query,
            cfg,
            cycle_index=cycle,
        )
    x = ops.linear(x, params["output_proj"], name="dit.output_proj")
    expected = jnp.where(query[..., None], x, 0.0).reshape(batch, blocks, slots, -1)
    assert jnp.allclose(output, expected, atol=1.0e-6)
    assert kv.shape == query.shape


def test_persistent_coordinate_rope_and_attention_connectivity():
    q = jnp.arange(2 * 2 * 1 * 8, dtype=jnp.float32).reshape(2, 2, 1, 8)
    k = q + 1
    coords = jnp.array([[3, 7], [7, 3]])
    qr, kr = apply_rope(q, k, coords)
    assert jnp.allclose(qr[0, 0], apply_rope(q[:1, :1], k[:1, :1], jnp.array([[3]]))[0][0, 0])
    assert not jnp.allclose(qr[0, 0], apply_rope(q[:1, :1], k[:1, :1], jnp.array([[4]]))[0][0, 0])
    assert bool(jnp.all(jnp.isfinite(kr)))

    packed, metadata, cfg, params, context = _dit_inputs()
    draft_a, _ = apply_dit(packed, metadata, context, params, cfg, mode="draft")
    draft_b, _ = apply_dit(packed.at[:, 2].add(20), metadata, context, params, cfg, mode="draft")
    assert jnp.allclose(draft_a[:, :2], draft_b[:, :2], atol=1.0e-5)
    refine_a, _ = apply_dit(packed, metadata, context, params, cfg, mode="refine")
    refine_b, _ = apply_dit(packed.at[:, 2].add(20), metadata, context, params, cfg, mode="refine")
    assert not jnp.allclose(refine_a[:, :2], refine_b[:, :2], atol=1.0e-5)

    packed, metadata, cfg, params, context = _dit_inputs(inactive=True)
    inactive_changed = packed.at[:, 1, 0].add(50)
    out_a, _ = apply_dit(packed, metadata, context, params, cfg)
    out_b, _ = apply_dit(inactive_changed, metadata, context, params, cfg)
    flat_a, flat_b = out_a.reshape(1, 6, -1), out_b.reshape(1, 6, -1)
    assert not jnp.allclose(flat_a[:, 2], flat_b[:, 2])
    assert jnp.allclose(flat_a[:, :2], flat_b[:, :2], atol=1.0e-5)
    assert jnp.allclose(flat_a[:, 3:], flat_b[:, 3:], atol=1.0e-5)


def test_dit_jit_and_every_physical_block_has_first_step_gradients():
    packed, metadata, cfg, params, context = _dit_inputs()

    def loss(p):
        output, _ = apply_dit(packed, metadata, context, p, cfg)
        return jnp.mean(output**2)

    grads = jax.grad(loss)(params)
    assert float(jnp.linalg.norm(grads["input_proj"]["weight"])) > 0
    assert float(jnp.linalg.norm(grads["output_proj"]["weight"])) > 0
    assert float(jnp.linalg.norm(grads["conditioning_trunk"]["weight"])) > 0
    for block in grads["blocks"]:
        for name in ("modulation", "q", "k", "v", "o", "up", "gate", "down"):
            norm = jnp.linalg.norm(block[name]["weight"])
            assert bool(jnp.isfinite(norm)) and float(norm) > 0
    assert bool(jnp.isfinite(jax.jit(loss)(params)))


def test_target_codec_site_distinction_proposal_prior_and_decoder_order():
    cfg = _tiny_config()
    params = init_model_params(jax.random.PRNGKey(6), cfg)
    target_a = jnp.arange(1, 9, dtype=jnp.int32)[None]
    target_b = target_a.at[:, 3].set(12)
    mask = jnp.ones_like(target_a, dtype=bool)
    encoded_a = encode_target(target_a, mask, params["encoder"], cfg)
    encoded_b = encode_target(target_b, mask, params["encoder"], cfg)
    assert not jnp.allclose(encoded_a["h0"][:, 0], encoded_a["h0"][:, 1])
    assert not jnp.allclose(encoded_a["h0"], encoded_b["h0"])

    context = jnp.zeros((1, cfg.model.d_ctx))
    prior_a = jnp.zeros((1, cfg.carrier.C, cfg.carrier.h_dim))
    prior_b = prior_a.at[:, 0].set(1)
    proposal_a = apply_proposal(context, prior_a, params["proposal"], cfg)[0]
    proposal_b = apply_proposal(context, prior_b, params["proposal"], cfg)[0]
    assert not jnp.allclose(proposal_a, proposal_b)

    lengths = jnp.array([[2, 2, 2, 2, 0, 0, 0, 0]], dtype=jnp.int32)
    logits, emit = apply_decoder(encoded_a["h0"], lengths, params["decoder"], cfg)
    assert logits.shape[:3] == emit.shape == (1, 8, 2)
    assert jnp.where(emit.reshape(-1))[0].tolist() == list(range(8))
    teacher = encoded_a["teacher"]
    assert jnp.array_equal(teacher.slot_bytes[teacher.slot_mask], target_a.reshape(-1))
    assert jnp.array_equal(teacher.slot_mask, emit)


def test_clean_target_loss_weights_and_byte_path_gradient():
    cfg = _tiny_config()
    params = init_model_params(jax.random.PRNGKey(7), cfg)
    target = jnp.arange(1, 9, dtype=jnp.int32)[None]
    mask = jnp.ones_like(target, dtype=bool)
    outputs = apply_model(params, target, mask, target, mask, config=cfg)
    components = loss_components(outputs)
    expected_h = jnp.mean((outputs["prediction"][0] - outputs["target"]["h0"]) ** 2)
    assert jnp.allclose(components["h"], expected_h)
    changed = replace(cfg, training=replace(cfg.training, h_weight=3.0))
    assert jnp.allclose(
        total_loss(components, changed) - total_loss(components, cfg), 2.0 * components["h"]
    )
    assert not jnp.allclose(outputs["target"]["h0"], jnp.zeros_like(outputs["target"]["h0"]))

    teacher = outputs["target"]["teacher"]

    def byte_loss(h_head):
        updated = {**params, "h_head": h_head}
        result = apply_model(updated, target, mask, target, mask, config=cfg)
        return cross_entropy(result["byte_logits"], teacher.slot_bytes, teacher.slot_mask)

    h_grad = jax.grad(byte_loss)(params["h_head"])
    assert float(jnp.linalg.norm(h_grad["weight"])) > 0


def test_full_first_step_gradient_coverage_including_clean_heads():
    cfg = _tiny_config()
    params = init_model_params(jax.random.PRNGKey(9), cfg)
    target = jnp.arange(1, 9, dtype=jnp.int32)[None]
    mask = jnp.ones_like(target, dtype=bool)

    def objective(p):
        return total_loss(
            loss_components(apply_model(p, target, mask, target, mask, config=cfg)), cfg
        )

    grads = jax.grad(objective)(params)

    def positive(tree):
        norm = jnp.sqrt(sum(jnp.sum(x**2) for x in jax.tree_util.tree_leaves(tree)))
        return bool(jnp.isfinite(norm)) and float(norm) > 0

    for name in ("carrier_in", "carrier_out", "h_head", "b_head", "l_head"):
        assert positive(grads[name]), name
    assert positive(grads["dit"]["conditioning_trunk"])
    for index, block in enumerate(grads["dit"]["blocks"]):
        for name in ("modulation", "q", "k", "v", "o", "up", "gate", "down"):
            assert positive(block[name]), (index, name)


def test_full_eager_jit_and_optimizer_update():
    cfg = _tiny_config()
    cfg = replace(cfg, packing=replace(cfg.packing, M_max=6))
    params, moments = initialise_training(jax.random.PRNGKey(8), cfg)
    target = jnp.arange(1, 9, dtype=jnp.int32)[None]
    batch = make_fixed_structure_batch(target, target, config=cfg)
    jitted = jax.jit(apply_model, static_argnames=("config", "mode"))
    output = jitted(
        params,
        batch["prompt"],
        batch["prompt_mask"],
        batch["target"],
        batch["target_mask"],
        config=cfg,
    )
    assert output["byte_logits"].shape == (1, 8, 2, 256)
    assert output["metadata"].packed_to_carrier.shape[1] == 4
    teacher = output["target"]["teacher"]
    padded = apply_model(
        params,
        batch["prompt"],
        batch["prompt_mask"],
        batch["target"],
        batch["target_mask"],
        config=cfg,
        committed_c_b=teacher.boundaries,
        committed_length=teacher.length,
    )
    assert padded["metadata"].packed_to_carrier.shape[1] == cfg.packing.M_max
    assert jnp.allclose(output["byte_logits"], padded["byte_logits"], atol=1.0e-5)
    old = params["h_head"]["weight"]
    params, _, metrics = jax.jit(train_step, static_argnames=("config",))(
        params, moments, 1, batch, config=cfg
    )
    assert bool(jnp.isfinite(metrics["loss"]))
    assert not jnp.array_equal(old, params["h_head"]["weight"])
