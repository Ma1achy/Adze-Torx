"""Phase F.3 self-generated S=2 rollout contracts."""

from __future__ import annotations

from dataclasses import replace
import inspect

import jax
import jax.numpy as jnp
import pytest

from adze_t.backends.mapping import deterministic_to_torx
from adze_t.backends.torx import TorxOps
from adze_t.config import REFERENCE_SMALL_V0
from adze_t.denoise import (
    F3_NATIVE_S_CONDITIONING_TRAINED,
    apply_denoising_trajectory,
    make_sanitized_target_analysis,
)
from adze_t.model import init_model_params
from adze_t.objectives import adamw_init
from adze_t.phase_f_3 import (
    PHASE_F_3_S2_TRAINING_BENEFIT,
    PHASE_F_3_S2_TRAINING_NEUTRAL,
    PHASE_F_3_S2_TRAINING_NEGATIVE,
    PHASE_F_3_S2_TRAINING_UNRESOLVED,
    bucket_effect,
    classify,
    effect_pattern,
    is_pareto_best_5k,
    require_exact_protocol,
    should_extend_to_10k,
    step0_forgetting,
)
from adze_t.training import (
    _CODEC_ENCODER_NAMES,
    s2_rollout_objective,
    stochastic_model_update_mask,
    stochastic_s2_rollout_train_step,
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
        training=replace(cfg.training, proposal_weight=0.0),
    )


def _fixture():
    cfg = _tiny_config()
    deterministic = init_model_params(jax.random.key(1300), cfg)
    params, _ = deterministic_to_torx(deterministic)
    prompt = jnp.full((2, 1), 0xF1, dtype=jnp.int32)
    target = jnp.arange(1, 17, dtype=jnp.int32).reshape(2, 8)
    batch = {
        "prompt": prompt,
        "prompt_mask": jnp.ones_like(prompt, dtype=bool),
        "target": target,
        "target_mask": jnp.ones_like(target, dtype=bool),
        "nu": jnp.asarray([0.4, 0.8], dtype=jnp.float32),
        "global_example_id": jnp.asarray([17, 29], dtype=jnp.uint32),
        "diffusion_occurrence": jnp.asarray(5001, dtype=jnp.uint32),
    }
    return cfg, params, batch


def _tree_equal(left, right):
    return all(
        bool(jnp.array_equal(a, b))
        for a, b in zip(jax.tree.leaves(left), jax.tree.leaves(right), strict=True)
    )


def test_s2_objective_is_self_generated_native_conditioned_and_has_one_parameter_tree():
    cfg, params, batch = _fixture()
    loss, auxiliary = s2_rollout_objective(
        params,
        batch,
        jax.random.key(1301),
        jax.random.key(1302),
        5001,
        config=cfg,
        lambda_op=0.0,
    )
    trajectory = auxiliary["trajectory"]
    assert jnp.isfinite(loss)
    assert jnp.array_equal(
        loss,
        0.5 * (auxiliary["step0"]["loss"] + auxiliary["step1"]["loss"]),
    )
    assert float(auxiliary["step1_input_error"]) == 0.0
    assert jnp.array_equal(trajectory.actual_s_indices[:, 0], jnp.asarray([0, 1]))
    assert jnp.array_equal(trajectory.denoise_condition_indices[:, 0], jnp.asarray([0, 1]))
    assert F3_NATIVE_S_CONDITIONING_TRAINED == "F3_NATIVE_S_CONDITIONING_TRAINED"
    assert tuple(inspect.signature(s2_rollout_objective).parameters).count("params") == 1
    assert "h0" not in inspect.signature(s2_rollout_objective).parameters
    assert jnp.all(trajectory.recorruption_epsilon == 0)
    teacher = auxiliary["teacher"]
    sanitized = make_sanitized_target_analysis(
        teacher.boundaries,
        teacher.length,
        config=cfg,
        target_width=batch["target"].shape[1],
    )
    s1 = apply_denoising_trajectory(
        params,
        batch["prompt"],
        batch["prompt_mask"],
        trajectory.input_states[0],
        sanitized,
        batch["nu"],
        batch["global_example_id"],
        s_exec=1,
        eta_diff=0,
        diffusion_root=jax.random.key(1302),
        operator_backend="torx",
        operator_root=jax.random.key(1301),
        operator_optimizer_step=5001,
        conditioning_mode=F3_NATIVE_S_CONDITIONING_TRAINED,
        config=cfg,
    )
    assert jnp.array_equal(s1.h_hat[0], trajectory.h_hat[0])


def test_s2_objective_rejects_nonfrozen_loss_weights():
    cfg, params, batch = _fixture()
    changed = replace(cfg, training=replace(cfg.training, byte_weight=0.5))
    try:
        s2_rollout_objective(
            params,
            batch,
            jax.random.key(1306),
            jax.random.key(1307),
            5001,
            config=changed,
            lambda_op=0.0,
        )
    except ValueError as error:
        assert "frozen DENOISE_V1 loss weights" in str(error)
    else:
        raise AssertionError("F3 accepted a mutated rollout loss contract")


def test_s2_update_freezes_rho_teacher_and_proposal_auxiliary():
    cfg, params, batch = _fixture()
    moments = adamw_init(params)
    updated, updated_moments, metrics = stochastic_s2_rollout_train_step(
        params,
        (moments, moments),
        5001,
        batch,
        jax.random.key(1303),
        jax.random.key(1304),
        config=cfg,
        lambda_op=0.0,
    )
    assert float(metrics["step1_input_error"]) == 0.0
    assert float(metrics["grad_rho_applied_norm"]) == 0.0
    assert float(metrics["grad_proposal"]) == 0.0
    assert all(bool(jnp.all(jnp.isfinite(value))) for value in jax.tree.leaves(metrics))
    assert all(bool(jnp.all(jnp.isfinite(value))) for value in jax.tree.leaves(updated))
    assert all(bool(jnp.all(jnp.isfinite(value))) for value in jax.tree.leaves(updated_moments))
    before_paths = dict(
        (jax.tree_util.keystr(path), value)
        for path, value in jax.tree_util.tree_leaves_with_path(params)
    )
    after_paths = dict(
        (jax.tree_util.keystr(path), value)
        for path, value in jax.tree_util.tree_leaves_with_path(updated)
    )
    moment_paths = dict(
        (jax.tree_util.keystr(path), value)
        for path, value in jax.tree_util.tree_leaves_with_path(updated_moments)
    )
    before_moment_paths = dict(
        (jax.tree_util.keystr(path), value)
        for path, value in jax.tree_util.tree_leaves_with_path((moments, moments))
    )
    for path, before in before_paths.items():
        teacher_path = any(f"['encoder']['{name}']" in path for name in _CODEC_ENCODER_NAMES)
        if "['rho']" in path or teacher_path:
            assert jnp.array_equal(before, after_paths[path])
            for moment_index in (0, 1):
                moment_path = f"[{moment_index}]{path}"
                assert jnp.array_equal(before_moment_paths[moment_path], moment_paths[moment_path])
    assert not _tree_equal(updated, params)
    update_mask = stochastic_model_update_mask(params)
    assert any(
        bool(enabled) and not bool(jnp.array_equal(before, after))
        for enabled, before, after in zip(
            jax.tree.leaves(update_mask),
            jax.tree.leaves(params),
            jax.tree.leaves(updated),
            strict=True,
        )
    )


def test_full_pathwise_credit_crosses_a_deterministic_recorruption_boundary():
    x = jnp.asarray(1.25)
    alpha_next = jnp.asarray(0.7)
    step1_weight = jnp.asarray(1.1)

    def step1_loss(step0_weight):
        h_hat_0 = step0_weight * x
        h_1_input = alpha_next * h_hat_0
        h_hat_1 = jax.lax.stop_gradient(step1_weight) * h_1_input
        return (h_hat_1 - 0.2) ** 2

    assert abs(float(jax.grad(step1_loss)(jnp.asarray(0.9)))) > 0.0


@pytest.mark.integration
def test_full_model_step1_gradient_differs_when_s2_boundary_is_detached():
    cfg, params, batch = _fixture()

    def step1_loss(model_params, detach_between_steps):
        _, auxiliary = s2_rollout_objective(
            model_params,
            batch,
            jax.random.key(1308),
            jax.random.key(1309),
            5001,
            config=cfg,
            lambda_op=0.0,
            detach_between_steps=detach_between_steps,
        )
        return auxiliary["step1"]["loss"]

    connected = jax.grad(lambda value: step1_loss(value, False))(params)
    detached = jax.grad(lambda value: step1_loss(value, True))(params)
    difference = jax.tree.map(lambda left, right: left - right, connected, detached)
    difference_norm = jnp.sqrt(
        jnp.sum(jnp.stack([jnp.sum(value * value) for value in jax.tree.leaves(difference)]))
    )
    assert jnp.isfinite(difference_norm)
    assert float(difference_norm) > 0.0


def test_operator_keys_separate_absolute_step_global_id_and_actual_s():
    root = jax.random.key(1305)

    def key(absolute_step, example_id, actual_s):
        return (
            TorxOps.create(
                root,
                optimizer_step=absolute_step,
                global_example_id=example_id,
            )
            .with_occurrence(denoise_step=actual_s)
            .context.key_for("f3.oracle")
        )

    baseline = key(5001, 17, 0)
    assert not jnp.array_equal(baseline, key(5001, 17, 1))
    assert not jnp.array_equal(baseline, key(5002, 17, 0))
    assert not jnp.array_equal(baseline, key(5001, 29, 0))


def _paired(direction: str):
    if direction == "benefit":
        return {
            "byte_accuracy": {
                "bootstrap_percentile_95_ci_low": 0.005,
                "bootstrap_percentile_95_ci_high": 0.02,
            },
            "nll": {
                "bootstrap_percentile_95_ci_low": -0.2,
                "bootstrap_percentile_95_ci_high": -0.01,
            },
        }
    if direction == "degradation":
        return {
            "byte_accuracy": {
                "bootstrap_percentile_95_ci_low": -0.02,
                "bootstrap_percentile_95_ci_high": -0.005,
            },
            "nll": {
                "bootstrap_percentile_95_ci_low": 0.01,
                "bootstrap_percentile_95_ci_high": 0.2,
            },
        }
    return {
        "byte_accuracy": {
            "bootstrap_percentile_95_ci_low": -0.001,
            "bootstrap_percentile_95_ci_high": 0.001,
        },
        "nll": {
            "bootstrap_percentile_95_ci_low": -0.001,
            "bootstrap_percentile_95_ci_high": 0.001,
        },
    }


def _row(nu, effect):
    benefit = effect == "benefit"
    degradation = effect == "degradation"
    return {
        "nu": nu,
        "s1_accuracy": 0.4,
        "s2_accuracy": 0.415 if benefit else 0.385 if degradation else 0.4,
        "s1_nll": 2.0,
        "s2_nll": 1.9 if benefit else 2.1 if degradation else 2.0,
        "paired": _paired(effect),
        "step0_forgetting": False,
    }


def test_predeclared_effect_forgetting_extension_and_label_gates():
    benefit_rows = [_row(nu, "benefit") for nu in (0.25, 0.5, 0.75)]
    assert bucket_effect(benefit_rows[0], require_uncertainty=True) == "benefit"
    assert (
        classify(benefit_rows, healthy=True, converged=True, stochastic_agrees=True)
        == PHASE_F_3_S2_TRAINING_BENEFIT
    )
    negative_rows = [_row(nu, "degradation") for nu in (0.25, 0.5, 0.75)]
    assert (
        classify(negative_rows, healthy=True, converged=True, stochastic_agrees=True)
        == PHASE_F_3_S2_TRAINING_NEGATIVE
    )
    forgetting_row = {
        "trained_s1_accuracy": 0.37,
        "f1_s1_accuracy": 0.40,
        "trained_s1_nll": 2.0,
        "f1_s1_nll": 2.0,
    }
    assert step0_forgetting(forgetting_row)
    rows_2k = [{"nu": nu, "s2_accuracy": 0.3, "s2_nll": 2.0} for nu in (0.25, 0.5, 0.75)]
    rows_5k = [{"nu": nu, "s2_accuracy": 0.311, "s2_nll": 2.0} for nu in (0.25, 0.5, 0.75)]
    assert should_extend_to_10k(rows_2k, rows_5k)
    assert is_pareto_best_5k([rows_2k], rows_5k)


@pytest.mark.parametrize("pattern", ("benefit", "degradation", "neutral"))
def test_unconverged_training_cannot_receive_a_terminal_scientific_label(pattern):
    rows = [_row(nu, pattern) for nu in (0.25, 0.5, 0.75)]
    assert effect_pattern(rows, require_uncertainty=True) == pattern
    assert (
        classify(rows, healthy=True, converged=False, stochastic_agrees=True)
        == PHASE_F_3_S2_TRAINING_UNRESOLVED
    )


@pytest.mark.parametrize(
    ("pattern", "terminal_label"),
    (
        ("benefit", PHASE_F_3_S2_TRAINING_BENEFIT),
        ("degradation", PHASE_F_3_S2_TRAINING_NEGATIVE),
        ("neutral", PHASE_F_3_S2_TRAINING_NEUTRAL),
    ),
)
def test_finite_torx_confirmation_is_symmetric(pattern, terminal_label):
    rows = [_row(nu, pattern) for nu in (0.25, 0.5, 0.75)]
    assert classify(rows, healthy=True, converged=True, stochastic_agrees=True) == terminal_label
    assert (
        classify(rows, healthy=True, converged=True, stochastic_agrees=False)
        == PHASE_F_3_S2_TRAINING_UNRESOLVED
    )


def test_protocol_provenance_rejects_mixed_or_stale_results():
    expected = {
        "contract": "PHASE_F_3_EVALUATION_PROTOCOL_V0",
        "checkpoint": {"file_sha256": "checkpoint-a"},
        "operator": {"root_index": 3, "lambda_op": 1.0},
        "dataset": {"ordered_global_ids_sha256": "ids-a"},
    }
    require_exact_protocol(expected, expected, "matching")
    for changed in (
        {**expected, "checkpoint": {"file_sha256": "checkpoint-b"}},
        {**expected, "operator": {"root_index": 4, "lambda_op": 1.0}},
        {**expected, "dataset": {"ordered_global_ids_sha256": "ids-b"}},
    ):
        with pytest.raises(RuntimeError, match="mixed or stale"):
            require_exact_protocol(changed, expected, "regression")
