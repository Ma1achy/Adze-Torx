from __future__ import annotations

from dataclasses import replace
import hashlib
from pathlib import Path
import pickle

import jax
import jax.numpy as jnp
import pytest

from adze_t.backends.mapping import deterministic_to_torx
from adze_t.backends.deterministic import DeterministicOps
from adze_t.backends.torx import TorxOperatorConfig, TorxOps
from adze_t.config import REFERENCE_SMALL_V0
from adze_t.denoise import as_sanitized_model_target_analysis, make_sanitized_target_analysis
from adze_t.generated_structure import generated_routing_core
from adze_t.model import apply_target_codec, init_model_params
from adze_t.objectives import adamw_init
from adze_t.phase_g import (
    evaluate_committed_replay,
    fail_closed_adamw_step,
    generated_structure_truth,
    generated_train_step,
    generated_two_pass_objective,
    guarded_generated_replay,
    neutral_structure,
)
from adze_t.state import CommittedStructure
from adze_t.training import stochastic_model_update_mask


ROOT = Path(__file__).resolve().parents[1]
F1_PARENT = ROOT / "results/runs/phase_f/f1/denoise_v1/one_step/init0_stoch0.pkl"
F1_SHA256 = "520d3723d3a3187aa08ff3f467f434d43b579cdc4b0ce667727dfad96e83cf57"
CODEC = ROOT / "results/phase_b/checkpoints/target_codec_b1.pkl"
CODEC_SHA256 = "55f057f78f795f1585d7aac28fa7ff37846d4c1098507ede375e86b21c4f8ff6"


def _config():
    cfg = REFERENCE_SMALL_V0
    return replace(
        cfg,
        carrier=replace(cfg.carrier, C=4, h_dim=8, L_max=2),
        packing=replace(cfg.packing, M_max=4, K=4),
        model=replace(
            cfg.model,
            d_front=8,
            d_ctx=8,
            frontend_layers=1,
            context_layers=1,
            target_layers=1,
            proposal_layers=1,
            proposal_hidden_dim=8,
            d_model=8,
            heads=1,
            head_dim=8,
            ffn_hidden=16,
            physical_blocks_L=1,
            cycles_Q=1,
            d_dec=8,
            decoder_layers=1,
            mamba_expand=1,
            mamba_state_dim=2,
        ),
        training=replace(cfg.training, proposal_weight=0.0),
    )


def _fixture():
    cfg = _config()
    deterministic = init_model_params(jax.random.key(600), cfg)
    params, _ = deterministic_to_torx(deterministic)
    target = jnp.arange(1, 9, dtype=jnp.int32)[None, :]
    batch = {
        "prompt": jnp.asarray([[0xF1]], jnp.int32),
        "prompt_mask": jnp.ones((1, 1), bool),
        "target": target,
        "target_mask": jnp.ones_like(target, bool),
        "nu": jnp.asarray([0.1]),
        "global_example_id": jnp.asarray([17], jnp.uint32),
        "structure_boundary": jnp.asarray([[0, 1, 0, 1]], jnp.int32),
        "structure_length": jnp.full((1, 4), 2, jnp.int32),
    }
    return cfg, params, batch


def _length(structure: CommittedStructure) -> jax.Array:
    assert structure.length is not None
    return structure.length


@pytest.mark.integration
def test_accepted_torx_checkpoint_codec_and_guarded_replay_are_repeatable():
    assert hashlib.sha256(F1_PARENT.read_bytes()).hexdigest() == F1_SHA256
    assert hashlib.sha256(CODEC.read_bytes()).hexdigest() == CODEC_SHA256
    with F1_PARENT.open("rb") as stream:
        state = pickle.load(stream)  # noqa: S301
    with CODEC.open("rb") as stream:
        codec_params = pickle.load(stream)  # noqa: S301
    target = jnp.arange(1, 9, dtype=jnp.int32)[None, :]
    mask = jnp.ones_like(target, dtype=bool)
    ops = TorxOps.create(
        jax.random.key(952),
        config=TorxOperatorConfig(operator_stochasticity=True, lambda_op=0.0),
        optimizer_step=0,
    )
    first = apply_target_codec(state["params"], target, mask, config=REFERENCE_SMALL_V0, ops=ops)
    second = apply_target_codec(state["params"], target, mask, config=REFERENCE_SMALL_V0, ops=ops)
    assert jax.tree_util.tree_all(jax.tree.map(jnp.array_equal, first, second))
    assert all(bool(jnp.all(jnp.isfinite(leaf))) for leaf in jax.tree.leaves(first))
    batch = {
        "prompt": jnp.asarray([[0xF1]], jnp.int32),
        "prompt_mask": jnp.ones((1, 1), bool),
        "target": target,
        "target_mask": mask,
        "nu": jnp.asarray([0.1]),
        "global_example_id": jnp.asarray([1024], jnp.uint32),
    }
    committed = neutral_structure(1, REFERENCE_SMALL_V0)
    first_metrics, first_replay = evaluate_committed_replay(
        state["params"],
        batch,
        committed,
        jax.random.key(953),
        jax.random.key(954),
        0,
        clean_params=codec_params,
        clean_ops=DeterministicOps(),
        config=REFERENCE_SMALL_V0,
    )
    second_metrics, second_replay = evaluate_committed_replay(
        state["params"],
        batch,
        committed,
        jax.random.key(953),
        jax.random.key(954),
        0,
        clean_params=codec_params,
        clean_ops=DeterministicOps(),
        config=REFERENCE_SMALL_V0,
    )
    assert bool(first_replay.executed)
    assert jax.tree_util.tree_all(
        jax.tree.map(
            jnp.array_equal,
            (first_metrics, first_replay),
            (second_metrics, second_replay),
        )
    )
    assert all(
        bool(jnp.all(jnp.isfinite(leaf))) for leaf in jax.tree.leaves((first_metrics, first_replay))
    )


def test_fail_closed_optimizer_preserves_params_moments_and_step_exactly():
    params = {"x": jnp.asarray([1.0, 2.0])}
    moments = (adamw_init(params), adamw_init(params))
    grads = {"x": jnp.asarray([3.0, 4.0])}
    mask = {"x": jnp.asarray(True)}
    unchanged, unchanged_moments, step, norm = fail_closed_adamw_step(
        params, grads, moments, 0, jnp.asarray(False), mask
    )
    assert jax.tree_util.tree_all(jax.tree.map(jnp.array_equal, unchanged, params))
    assert jax.tree_util.tree_all(jax.tree.map(jnp.array_equal, unchanged_moments, moments))
    assert step == 0
    assert norm == 0
    updated, _, next_step, _ = fail_closed_adamw_step(
        params, grads, moments, 0, jnp.asarray(True), mask
    )
    assert next_step == 1
    assert not jnp.array_equal(updated["x"], params["x"])
    assert jnp.all(jnp.isfinite(updated["x"]))


def test_invalid_generated_route_returns_nonexecuted_zero_replay():
    cfg, params, batch = _fixture()
    invalid = CommittedStructure(
        jnp.ones((1, 4), jnp.int32), jnp.ones((1, 4), bool), jnp.ones((1, 4), jnp.int32)
    )
    route = generated_routing_core(invalid, M_max=1, K=4, max_extent=2)
    neutral = neutral_structure(1, cfg)
    sanitized = make_sanitized_target_analysis(
        neutral.c_b, _length(neutral), config=cfg, target_width=8
    )
    replay = guarded_generated_replay(
        params,
        batch,
        jnp.ones((1, 4, 8)),
        as_sanitized_model_target_analysis(sanitized),
        neutral,
        route,
        TorxOps.create(
            jax.random.key(601),
            config=TorxOperatorConfig(operator_stochasticity=True, lambda_op=0.0),
        ),
        config=cfg,
    )
    assert not bool(replay.executed)
    assert jnp.all(replay.h_hat == 0)
    assert jnp.all(replay.byte_logits == 0)


@pytest.mark.integration
def test_guarded_replay_is_independent_of_real_and_placeholder_target_content():
    cfg, params, batch = _fixture()
    neutral = neutral_structure(1, cfg)
    route = generated_routing_core(neutral, M_max=4, K=4, max_extent=2)
    sanitized = make_sanitized_target_analysis(
        neutral.c_b, _length(neutral), config=cfg, target_width=8, placeholder_value=13
    )
    target_analysis = as_sanitized_model_target_analysis(sanitized)
    carrier = jnp.arange(32, dtype=jnp.float32).reshape(1, 4, 8) / 10
    ops = TorxOps.create(
        jax.random.key(602),
        config=TorxOperatorConfig(operator_stochasticity=True, lambda_op=0.0),
    )
    first = guarded_generated_replay(
        params, batch, carrier, target_analysis, neutral, route, ops, config=cfg
    )
    changed = {
        **batch,
        "target": jnp.full_like(batch["target"], 31),
        "target_mask": jnp.zeros_like(batch["target_mask"]),
    }
    second = guarded_generated_replay(
        params, changed, carrier, target_analysis, neutral, route, ops, config=cfg
    )
    other_sanitized = make_sanitized_target_analysis(
        neutral.c_b, _length(neutral), config=cfg, target_width=8, placeholder_value=29
    )
    third = guarded_generated_replay(
        params,
        changed,
        carrier,
        as_sanitized_model_target_analysis(other_sanitized),
        neutral,
        route,
        ops,
        config=cfg,
    )
    assert jnp.array_equal(first.h_hat, second.h_hat)
    assert jnp.array_equal(first.byte_logits, second.byte_logits)
    assert jnp.array_equal(first.h_hat, third.h_hat)
    assert jnp.array_equal(first.byte_logits, third.byte_logits)


def test_task_structure_is_deterministic_from_observable_codec_slots():
    slots = jnp.asarray([[[4, 9], [28, 3], [0, 0], [0, 0]]], jnp.int32)
    lengths = jnp.asarray([[2, 2, 0, 0]], jnp.int32)
    first = generated_structure_truth(slots, lengths, K=4)
    second = generated_structure_truth(slots.copy(), lengths.copy(), K=4)
    assert jnp.array_equal(first.c_b, second.c_b)
    assert first.c_b.tolist() == [[1, 0, 0, 1]]
    changed = generated_structure_truth(slots.at[0, 0, 0].set(20), lengths, K=4)
    assert not jnp.array_equal(first.c_b, changed.c_b)


@pytest.mark.integration
def test_teacher_two_pass_is_finite_and_uses_explicit_variable_structure_truth():
    cfg, params, batch = _fixture()
    loss, metrics = generated_two_pass_objective(
        params,
        batch,
        jax.random.key(603),
        jax.random.key(604),
        jax.random.key(605),
        0,
        arm="teacher",
        config=cfg,
    )
    assert bool(metrics["executed"])
    assert bool(jnp.isfinite(loss))


@pytest.mark.integration
def test_predictor_and_replay_ignore_sanitized_placeholder_values():
    cfg, params, batch = _fixture()
    keys = (jax.random.key(615), jax.random.key(616), jax.random.key(617))
    first = generated_two_pass_objective(
        params,
        batch,
        *keys,
        0,
        arm="teacher",
        inert_placeholder_value=0,
        config=cfg,
    )
    second = generated_two_pass_objective(
        params,
        batch,
        *keys,
        0,
        arm="teacher",
        inert_placeholder_value=31,
        config=cfg,
    )
    assert jax.tree_util.tree_all(jax.tree.map(jnp.array_equal, first, second))


@pytest.mark.integration
def test_full_model_prospective_update_is_finite_and_preserves_frozen_leaves():
    cfg, params, batch = _fixture()
    moments = (adamw_init(params), adamw_init(params))
    compiled_update = jax.jit(generated_train_step, static_argnames=("arm", "config"))
    updated, updated_moments, next_step, metrics = compiled_update(
        params,
        moments,
        0,
        batch,
        jax.random.key(609),
        jax.random.key(610),
        jax.random.key(611),
        arm="teacher",
        config=cfg,
    )
    mask = stochastic_model_update_mask(params)
    assert bool(metrics["executed"])
    assert bool(jnp.isfinite(metrics["loss"]))
    assert bool(jnp.isfinite(metrics["grad_raw_norm"]))
    assert next_step == 1
    changed_enabled = False
    for before, after, enabled in zip(
        jax.tree.leaves(params),
        jax.tree.leaves(updated),
        jax.tree.leaves(mask),
        strict=True,
    ):
        assert bool(jnp.all(jnp.isfinite(after)))
        if bool(enabled):
            changed_enabled |= not bool(jnp.array_equal(before, after))
        else:
            assert jnp.array_equal(before, after)
    assert changed_enabled
    for old_moment_tree, moment_tree in zip(moments, updated_moments, strict=True):
        assert all(bool(jnp.all(jnp.isfinite(leaf))) for leaf in jax.tree.leaves(moment_tree))
        for before, after, enabled in zip(
            jax.tree.leaves(old_moment_tree),
            jax.tree.leaves(moment_tree),
            jax.tree.leaves(mask),
            strict=True,
        ):
            if not bool(enabled):
                assert jnp.array_equal(before, after)


@pytest.mark.integration
def test_full_model_invalid_batch_skips_loss_parameter_moments_and_step():
    cfg, params, batch = _fixture()
    moments = (adamw_init(params), adamw_init(params))
    invalid_batch = {
        **batch,
        "structure_length": batch["structure_length"].at[0, 0].set(1),
    }
    updated, updated_moments, next_step, metrics = generated_train_step(
        params,
        moments,
        0,
        invalid_batch,
        jax.random.key(612),
        jax.random.key(613),
        jax.random.key(614),
        arm="teacher",
        config=cfg,
    )
    assert not bool(metrics["executed"])
    assert metrics["loss"] == 0
    assert next_step == 0
    assert jax.tree_util.tree_all(jax.tree.map(jnp.array_equal, updated, params))
    assert jax.tree_util.tree_all(jax.tree.map(jnp.array_equal, updated_moments, moments))


@pytest.mark.integration
def test_full_model_nonfinite_objective_skips_optimizer_mutation():
    cfg, params, batch = _fixture()
    moments = (adamw_init(params), adamw_init(params))
    nonfinite_batch = {**batch, "nu": jnp.asarray([jnp.nan])}
    updated, updated_moments, next_step, metrics = generated_train_step(
        params,
        moments,
        0,
        nonfinite_batch,
        jax.random.key(618),
        jax.random.key(619),
        jax.random.key(620),
        arm="teacher",
        config=cfg,
    )
    assert not bool(metrics["executed"])
    assert not bool(metrics["gradients_finite"])
    assert next_step == 0
    assert jax.tree_util.tree_all(jax.tree.map(jnp.array_equal, updated, params))
    assert jax.tree_util.tree_all(jax.tree.map(jnp.array_equal, updated_moments, moments))


@pytest.mark.integration
def test_real_model_score_gradient_changes_when_sample_log_probability_is_detached():
    cfg, params, batch = _fixture()

    def objective(value, connected):
        return generated_two_pass_objective(
            value,
            batch,
            jax.random.key(606),
            jax.random.key(607),
            jax.random.key(608),
            0,
            arm="sample",
            temperature=1.0,
            apply_score_correction=connected,
            config=cfg,
        )[0]

    connected = jax.grad(lambda value: objective(value, True))(params)
    detached = jax.grad(lambda value: objective(value, False))(params)
    differences = jax.tree.map(lambda left, right: left - right, connected, detached)
    b_norm = jnp.sqrt(sum(jnp.sum(value**2) for value in jax.tree.leaves(differences["b_head"])))
    l_norm = jnp.sqrt(sum(jnp.sum(value**2) for value in jax.tree.leaves(differences["l_head"])))
    assert bool(jnp.isfinite(b_norm)) and float(b_norm) > 0
    assert bool(jnp.isfinite(l_norm)) and float(l_norm) > 0
