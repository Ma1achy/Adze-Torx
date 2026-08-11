from dataclasses import replace

import jax
import jax.numpy as jnp
import pytest

from adze_t.backends.deterministic import DeterministicOps
from adze_t.config import REFERENCE_SMALL_V0
from adze_t.dit import DiTConfig, apply_attention, masked_softmax
from adze_t.mamba import MambaConfig, apply_mamba_stack, init_mamba_stack
from adze_t.objectives import adamw_step
from adze_t.teacher import canonical_teacher_structure, canonical_teacher_structure_core
from adze_t.training import (
    codec_pretrain_step,
    codec_update_mask,
    initialise_training,
    make_fixed_structure_batch,
    model_update_mask,
    train_step,
)


def _tiny_config():
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
            physical_blocks_L=2,
            cycles_Q=2,
            d_dec=16,
            decoder_layers=1,
            mamba_expand=1,
            mamba_state_dim=4,
        ),
    )


def _tree_equal(left, right):
    return all(
        bool(jnp.array_equal(a, b))
        for a, b in zip(
            jax.tree_util.tree_leaves(left), jax.tree_util.tree_leaves(right), strict=True
        )
    )


def test_masked_adamw_preserves_frozen_parameter_and_moments_despite_decay():
    params = {"active": jnp.array([2.0]), "frozen": jnp.array([3.0])}
    grads = {"active": jnp.array([1.0]), "frozen": jnp.array([7.0])}
    moments = (
        {"active": jnp.array([0.4]), "frozen": jnp.array([0.5])},
        {"active": jnp.array([0.6]), "frozen": jnp.array([0.7])},
    )
    mask = {"active": jnp.asarray(True), "frozen": jnp.asarray(False)}
    updated, new_moments, norm = adamw_step(
        params,
        grads,
        moments,
        3,
        learning_rate=0.1,
        weight_decay=0.5,
        clip_norm=100.0,
        update_mask=mask,
    )
    assert jnp.array_equal(updated["frozen"], params["frozen"])
    assert jnp.array_equal(new_moments[0]["frozen"], moments[0]["frozen"])
    assert jnp.array_equal(new_moments[1]["frozen"], moments[1]["frozen"])
    assert not jnp.array_equal(updated["active"], params["active"])
    assert jnp.allclose(norm, 1.0)


def test_codec_and_model_update_masks_are_complementary_at_codec_boundary():
    cfg = _tiny_config()
    params, _ = initialise_training(jax.random.PRNGKey(0), cfg)
    codec_mask = codec_update_mask(params)
    model_mask = model_update_mask(params)
    assert bool(codec_mask["encoder"]["target_h"]["weight"])
    assert not bool(model_mask["encoder"]["target_h"]["weight"])
    assert bool(codec_mask["encoder"]["frontend"][0]["a_log"])
    assert not bool(model_mask["encoder"]["frontend"][0]["a_log"])
    assert bool(codec_mask["decoder"]["out"]["weight"])
    assert bool(model_mask["decoder"]["out"]["weight"])
    assert not bool(codec_mask["dit"]["input_proj"]["weight"])
    assert bool(model_mask["dit"]["input_proj"]["weight"])


def test_real_training_steps_do_not_move_frozen_parameters_or_moments():
    cfg = _tiny_config()
    params, moments = initialise_training(jax.random.PRNGKey(1), cfg)
    values = jnp.arange(8, dtype=jnp.int32)[None]
    batch = make_fixed_structure_batch(values, values, config=cfg)

    proposal_before = params["proposal"]
    proposal_moments_before = (moments[0]["proposal"], moments[1]["proposal"])
    params, moments, _ = codec_pretrain_step(params, moments, 1, batch, config=cfg)
    assert _tree_equal(params["proposal"], proposal_before)
    assert _tree_equal(moments[0]["proposal"], proposal_moments_before[0])
    assert _tree_equal(moments[1]["proposal"], proposal_moments_before[1])

    target_before = params["encoder"]["target"]
    target_moments_before = (
        moments[0]["encoder"]["target"],
        moments[1]["encoder"]["target"],
    )
    context_before = params["encoder"]["context"]
    params, moments, _ = train_step(params, moments, 2, batch, config=cfg)
    assert _tree_equal(params["encoder"]["target"], target_before)
    assert _tree_equal(moments[0]["encoder"]["target"], target_moments_before[0])
    assert _tree_equal(moments[1]["encoder"]["target"], target_moments_before[1])
    assert not _tree_equal(params["encoder"]["context"], context_before)


def test_masked_softmax_and_post_projection_attention_are_zero_for_empty_rows():
    scores = jnp.array([[[[2.0, -1.0], [0.5, 0.25]]]])
    mask = jnp.zeros_like(scores, dtype=bool)
    assert jnp.array_equal(masked_softmax(scores, mask), jnp.zeros_like(scores))

    cfg = DiTConfig(d_model=4, heads=1, head_dim=4, ffn_hidden=8, physical_blocks=1)
    zeros = {"weight": jnp.zeros((4, 4)), "bias": jnp.zeros((4,))}
    block = {
        "q": zeros,
        "k": zeros,
        "v": zeros,
        "o": {"weight": jnp.zeros((4, 4)), "bias": jnp.array([1.0, -2.0, 3.0, -4.0])},
    }
    contribution = apply_attention(
        jnp.ones((1, 2, 4)),
        block,
        jnp.zeros((1, 2, 2), dtype=bool),
        jnp.array([[0, 1]], dtype=jnp.int32),
        cfg,
        DeterministicOps(),
        name="test",
    )
    assert jnp.array_equal(contribution, jnp.zeros_like(contribution))


def test_masked_mamba_scan_is_a_state_noop_for_prefix_tail_and_internal_holes():
    ops = DeterministicOps()
    cfg = MambaConfig(width=4, layers=2, expand=1, state_dim=3, conv_kernel=3)
    params = init_mamba_stack(jax.random.PRNGKey(2), cfg, ops, name="masked")
    x = jax.random.normal(jax.random.PRNGKey(3), (1, 7, 4))

    all_valid = jnp.ones((1, 7), dtype=bool)
    unmasked = apply_mamba_stack(x, params, cfg, ops, name="masked")
    explicitly_valid = apply_mamba_stack(x, params, cfg, ops, name="masked", mask=all_valid)
    assert jnp.array_equal(unmasked, explicitly_valid)

    inactive_tail = jnp.array([[1, 1, 1, 1, 0, 0, 0]], dtype=bool)
    prefix_output = apply_mamba_stack(x, params, cfg, ops, name="masked", mask=inactive_tail)
    truncated = apply_mamba_stack(x[:, :4], params, cfg, ops, name="masked")
    assert jnp.allclose(prefix_output[:, :4], truncated, atol=1.0e-6)
    assert jnp.array_equal(prefix_output[:, 4:], jnp.zeros_like(prefix_output[:, 4:]))

    prefix_padding = jnp.array([[0, 0, 1, 1, 1, 1, 1]], dtype=bool)
    padded_base = apply_mamba_stack(x, params, cfg, ops, name="masked", mask=prefix_padding)
    padded_changed = apply_mamba_stack(
        x.at[:, :2].add(1000.0),
        params,
        cfg,
        ops,
        name="masked",
        mask=prefix_padding,
    )
    assert jnp.array_equal(padded_base[:, :2], jnp.zeros_like(padded_base[:, :2]))
    assert jnp.allclose(padded_base, padded_changed, atol=1.0e-6)

    for hole_mask in (
        jnp.array([[1, 1, 0, 1, 1, 1, 1]], dtype=bool),
        jnp.array([[1, 1, 0, 0, 1, 1, 1]], dtype=bool),
    ):
        base = apply_mamba_stack(x, params, cfg, ops, name="masked", mask=hole_mask)
        changed = apply_mamba_stack(
            jnp.where(hole_mask[..., None], x, x + 1000.0),
            params,
            cfg,
            ops,
            name="masked",
            mask=hole_mask,
        )
        assert jnp.array_equal(base[~hole_mask], jnp.zeros_like(base[~hole_mask]))
        assert jnp.allclose(base, changed, atol=1.0e-6)
        earlier_changed = apply_mamba_stack(
            x.at[:, 1].add(1.0), params, cfg, ops, name="masked", mask=hole_mask
        )
        assert not jnp.allclose(base[:, 4:], earlier_changed[:, 4:])

    compiled = jax.jit(apply_mamba_stack, static_argnames=("config", "ops", "name"))
    jit_output = compiled(x, params, cfg, ops, name="masked", mask=inactive_tail)
    assert jnp.allclose(jit_output, prefix_output, atol=1.0e-6)


def test_zero_byte_is_data_and_explicit_masks_control_padding():
    cfg = _tiny_config()
    values = jnp.array([[0, 2, 0, 4]], dtype=jnp.int32)
    dense = make_fixed_structure_batch(values, values, config=cfg)
    assert dense["target_mask"].tolist() == [[True, True, True, True]]
    assert dense["committed_length"][0, :3].tolist() == [2, 2, 0]
    assert dense["target"][0].tolist() == [0, 2, 0, 4]

    mask = jnp.array([[1, 1, 0, 0]], dtype=bool)
    padded = make_fixed_structure_batch(
        values, values, prompt_mask=mask, target_mask=mask, config=cfg
    )
    assert padded["committed_length"][0, :3].tolist() == [2, 0, 0]


def test_teacher_overflow_and_nonprefix_masks_are_explicit_and_jittable():
    cfg = _tiny_config()
    capacity = cfg.carrier.C * cfg.carrier.L_max
    too_long = jnp.arange(capacity + 1, dtype=jnp.int32)[None]
    too_long_mask = jnp.ones_like(too_long, dtype=bool)
    core = jax.jit(canonical_teacher_structure_core, static_argnames=("config",))(
        too_long, too_long_mask, cfg
    )
    assert core.capacity_overflow.tolist() == [True]
    with pytest.raises(ValueError, match="exceeds carrier emission capacity"):
        canonical_teacher_structure(too_long, too_long_mask, cfg)

    values = jnp.arange(4, dtype=jnp.int32)[None]
    hole = jnp.array([[1, 0, 1, 0]], dtype=bool)
    core = canonical_teacher_structure_core(values, hole, cfg)
    assert core.prefix_mask_valid.tolist() == [False]
    with pytest.raises(ValueError, match="prefix-valid"):
        canonical_teacher_structure(values, hole, cfg)

    exact = jnp.arange(capacity, dtype=jnp.int32)[None]
    teacher = canonical_teacher_structure(exact, jnp.ones_like(exact, dtype=bool), cfg)
    assert teacher.capacity_overflow.tolist() == [False]
    assert int(jnp.sum(teacher.slot_mask)) == capacity
