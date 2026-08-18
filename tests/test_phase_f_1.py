"""Phase F.1 explicit-carrier, leakage, and one-step training contracts."""

from __future__ import annotations

from dataclasses import replace

import jax
import jax.numpy as jnp
import pytest

from adze_t.backends.deterministic import DeterministicOps
from adze_t.backends.mapping import deterministic_to_torx
from adze_t.config import REFERENCE_SMALL_V0
from adze_t.corruption import corrupt_h
from adze_t.decoder import apply_decoder
from adze_t.dit import apply_dit
from adze_t.model import _dit_config, apply_clean_target_teacher, apply_model, init_model_params
from adze_t.objectives import adamw_init
from adze_t.packing import pack_values
from adze_t.phase_f_1 import (
    DENOISE_V1,
    DENOISE_V1_TARGET_DOMAIN,
    dataset_audit,
    generate_denoise_v0,
    initial_diffusion_epsilon,
)
from adze_t.teacher import canonical_teacher_structure_core
from adze_t.training import stochastic_denoise_train_step
from adze_t.unpool import unpool_values


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
        training=replace(cfg.training, proposal_weight=0.0),
    )


def _inputs():
    prompt = jnp.full((2, 1), 0xF1, dtype=jnp.int32)
    target = jnp.arange(16, dtype=jnp.int32).reshape(2, 8)
    return prompt, jnp.ones_like(prompt, bool), target, jnp.ones_like(target, bool)


def _tree_equal(left, right):
    return all(
        bool(jnp.array_equal(a, b))
        for a, b in zip(
            jax.tree_util.tree_leaves(left), jax.tree_util.tree_leaves(right), strict=True
        )
    )


def test_denoise_v0_generation_is_deterministic_constant_context_and_random_target():
    first = generate_denoise_v0(64, 940)
    second = generate_denoise_v0(64, 940)
    assert all(jnp.array_equal(a, b) for a, b in zip(first, second, strict=True))
    prompt, target, ids = first
    assert dataset_audit(prompt, target)["constant_prompt"]
    assert jnp.array_equal(ids, jnp.arange(64, dtype=jnp.uint32))
    assert bool(jnp.any(target[1:] != target[:-1]))


def test_denoise_v1_changes_only_the_target_domain():
    prompt, target, ids = generate_denoise_v0(4_096, 940, spec=DENOISE_V1)
    audit = dataset_audit(prompt, target, spec=DENOISE_V1)
    assert audit["task_version"] == "DENOISE_V1"
    assert audit["target_domain"] == list(DENOISE_V1_TARGET_DOMAIN)
    assert audit["target_domain_valid"]
    assert int(jnp.min(target)) == 1
    assert int(jnp.max(target)) == 32
    assert jnp.all(prompt == 0xF1)
    assert jnp.array_equal(ids, jnp.arange(4_096, dtype=jnp.uint32))


def test_validation_epsilon_is_paired_across_nu_and_training_occurrences_are_fresh():
    h0 = jnp.arange(24, dtype=jnp.float32).reshape(2, 3, 4) / 10.0
    ids = jnp.asarray([7, 11], dtype=jnp.uint32)
    root = jax.random.key(950)
    epsilon_low = initial_diffusion_epsilon(h0, root, ids)
    epsilon_high = initial_diffusion_epsilon(h0, root, ids)
    low = corrupt_h(h0, jnp.asarray([0.1, 0.1]), epsilon_low)
    high = corrupt_h(h0, jnp.asarray([0.9, 0.9]), epsilon_high)
    assert jnp.array_equal(epsilon_low, epsilon_high)
    assert not jnp.array_equal(low, high)
    train_a = initial_diffusion_epsilon(h0, root, ids, optimizer_step=1)
    train_b = initial_diffusion_epsilon(h0, root, ids, optimizer_step=2)
    assert not jnp.array_equal(train_a, train_b)
    assert jnp.array_equal(
        train_a,
        initial_diffusion_epsilon(h0, root, ids, optimizer_step=1),
    )


def test_fixed_eight_byte_structure_and_pack_metadata_are_content_invariant():
    cfg = _tiny_config()
    target = jax.random.randint(jax.random.key(951), (1024, 8), 0, 256)
    mask = jnp.ones_like(target, bool)
    teacher = canonical_teacher_structure_core(target, mask, cfg)
    assert jnp.all(teacher.boundaries == teacher.boundaries[0])
    assert jnp.all(teacher.length == teacher.length[0])
    assert jnp.all(teacher.activity == teacher.activity[0])
    from adze_t.packing import build_pack_metadata_core

    metadata = build_pack_metadata_core(
        teacher.boundaries,
        teacher.activity,
        M_max=cfg.packing.M_max,
        K=cfg.packing.K,
    )
    for field in (
        "carrier_to_m",
        "carrier_to_k",
        "packed_to_carrier",
        "slot_valid",
        "kv_mask",
        "query_mask",
        "block_id",
        "carrier_id",
        "within_block_pos",
    ):
        values = getattr(metadata, field)
        assert jnp.all(values == values[0])


def test_legacy_path_equals_explicit_proposal_state_and_zero_corruption_is_exact():
    cfg = _tiny_config()
    params = init_model_params(jax.random.key(952), cfg)
    prompt, prompt_mask, target, target_mask = _inputs()
    legacy = apply_model(params, prompt, prompt_mask, target, target_mask, config=cfg)
    explicit = apply_model(
        params,
        prompt,
        prompt_mask,
        target,
        target_mask,
        config=cfg,
        carrier_h_input=legacy["proposal"][0],
    )
    for key in (
        "packed_carrier",
        "packed_output",
        "unpooled_carrier",
        "pre_head_carrier",
        "carrier",
        "byte_logits",
    ):
        assert jnp.array_equal(legacy[key], explicit[key])
    epsilon = jax.random.normal(jax.random.key(953), legacy["target"]["h0"].shape)
    assert jnp.array_equal(
        corrupt_h(legacy["target"]["h0"], 0.0, epsilon),
        legacy["target"]["h0"],
    )


def test_explicit_state_matches_independent_faithful_heavy_core_reference():
    cfg = _tiny_config()
    params = init_model_params(jax.random.key(954), cfg)
    prompt, prompt_mask, target, target_mask = _inputs()
    state = jax.random.normal(jax.random.key(955), (2, cfg.carrier.C, cfg.carrier.h_dim))
    actual = apply_model(
        params,
        prompt,
        prompt_mask,
        target,
        target_mask,
        config=cfg,
        carrier_h_input=state,
        noise_level=jnp.asarray([0.25, 0.75]),
    )
    ops = DeterministicOps()
    packed = pack_values(
        ops.linear(state, params["carrier_in"], name="model.carrier_input"),
        actual["metadata"],
    )
    packed_out, _ = apply_dit(
        packed,
        actual["metadata"],
        actual["context_global"],
        params["dit"],
        _dit_config(cfg),
        ops=ops,
        observed_b=actual["target"]["teacher"].boundaries,
        observed_l=actual["target"]["teacher"].length,
        noise=jnp.asarray([0.25, 0.75]),
        denoise_step=0,
    )
    unpooled = unpool_values(packed_out, actual["metadata"], C=cfg.carrier.C)
    residual = state + ops.linear(unpooled, params["carrier_out"], name="model.carrier_output")
    h_hat = ops.linear(residual, params["h_head"], name="model.h_head")
    b_logits = ops.categorical_logits(residual, params["b_head"], name="model.b_head")
    l_logits = ops.categorical_logits(residual, params["l_head"], name="model.l_head")
    byte_logits, _ = apply_decoder(
        h_hat,
        actual["target"]["teacher"].length,
        params["decoder"],
        cfg,
        ops.with_scope("output"),
        name="decoder",
    )
    for expected, observed in (
        (packed, actual["packed_carrier"]),
        (packed_out, actual["packed_output"]),
        (unpooled, actual["unpooled_carrier"]),
        (residual, actual["pre_head_carrier"]),
        (h_hat, actual["prediction"][0]),
        (b_logits, actual["prediction"][1]),
        (l_logits, actual["prediction"][2]),
        (byte_logits, actual["byte_logits"]),
    ):
        assert jnp.allclose(expected, observed, rtol=0.0, atol=1.0e-7)


def test_teacher_content_cannot_leak_when_carrier_and_structure_are_fixed():
    cfg = _tiny_config()
    params = init_model_params(jax.random.key(956), cfg)
    prompt, prompt_mask, target_a, target_mask = _inputs()
    target_b = target_a + 100
    ops = DeterministicOps()
    analysis_a = apply_clean_target_teacher(params, target_a, target_mask, config=cfg, ops=ops)
    analysis_b = apply_clean_target_teacher(params, target_b, target_mask, config=cfg, ops=ops)
    structure = analysis_a["target"]["teacher"]
    state = jax.random.normal(jax.random.key(957), (2, cfg.carrier.C, cfg.carrier.h_dim))

    def run(target, analysis):
        return apply_model(
            params,
            prompt,
            prompt_mask,
            target,
            target_mask,
            config=cfg,
            target_analysis=analysis,
            carrier_h_input=state,
            committed_c_b=structure.boundaries,
            committed_length=structure.length,
            noise_level=0.5,
        )

    first, second = run(target_a, analysis_a), run(target_b, analysis_b)
    for key in ("packed_carrier", "packed_output", "carrier", "byte_logits"):
        assert jnp.array_equal(first[key], second[key])
    assert not jnp.array_equal(first["target"]["h0"], second["target"]["h0"])


def test_explicit_carrier_and_noise_conditioning_are_inference_visible():
    cfg = _tiny_config()
    params = init_model_params(jax.random.key(958), cfg)
    prompt, prompt_mask, target, target_mask = _inputs()
    state = jnp.zeros((2, cfg.carrier.C, cfg.carrier.h_dim))

    def run(carrier, nu):
        return apply_model(
            params,
            prompt,
            prompt_mask,
            target,
            target_mask,
            config=cfg,
            carrier_h_input=carrier,
            noise_level=nu,
        )

    baseline = run(state, 0.25)
    changed_state = run(state.at[:, 0, 0].set(1.0), 0.25)
    changed_nu = run(state, 0.75)
    assert not jnp.array_equal(baseline["packed_carrier"], changed_state["packed_carrier"])
    assert not jnp.array_equal(baseline["byte_logits"], changed_state["byte_logits"])
    assert not jnp.array_equal(baseline["packed_output"], changed_nu["packed_output"])


@pytest.mark.integration
def test_f1_stochastic_gradient_is_finite_and_freezes_teacher_and_rho():
    cfg = _tiny_config()
    deterministic = init_model_params(jax.random.key(959), cfg)
    params, _ = deterministic_to_torx(deterministic)
    zeros = adamw_init(params)
    prompt, prompt_mask, target, target_mask = _inputs()
    batch = {
        "prompt": prompt,
        "prompt_mask": prompt_mask,
        "target": target,
        "target_mask": target_mask,
        "nu": jnp.asarray([0.4, 0.6], dtype=jnp.float32),
        "global_example_id": jnp.asarray([0, 1], dtype=jnp.uint32),
        "diffusion_occurrence": jnp.asarray(1, dtype=jnp.uint32),
    }
    update = jax.jit(stochastic_denoise_train_step, static_argnames=("config",))
    updated, _, metrics = update(
        params,
        (zeros, zeros),
        1,
        batch,
        jax.random.key(960),
        jax.random.key(961),
        config=cfg,
    )
    assert all(bool(jnp.all(jnp.isfinite(x))) for x in jax.tree_util.tree_leaves(metrics))
    assert float(metrics["grad_permitted_norm"]) > 0.0
    assert float(metrics["grad_dit_qkvo"]) > 0.0
    assert float(metrics["grad_dit_ffn"]) > 0.0
    assert float(metrics["grad_output_heads"]) > 0.0
    assert float(metrics["grad_decoder"]) > 0.0
    assert float(metrics["grad_proposal"]) == 0.0
    assert float(metrics["grad_rho_applied_norm"]) == 0.0
    assert _tree_equal(updated["encoder"]["target"], params["encoder"]["target"])
    for path, before in jax.tree_util.tree_leaves_with_path(params):
        if "['rho']" in jax.tree_util.keystr(path):
            after = dict(
                (jax.tree_util.keystr(p), value)
                for p, value in jax.tree_util.tree_leaves_with_path(updated)
            )[jax.tree_util.keystr(path)]
            assert jnp.array_equal(before, after)
