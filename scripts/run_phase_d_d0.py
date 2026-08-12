"""Generate the Phase-D finite-noise contract evidence."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp

from adze_t.backends.mapping import deterministic_to_torx
from adze_t.backends.torx import (
    PHASE_D_DIRECT_PARAMETER_NOISE_POLICY_V0_MEAN_ONLY,
    PHASE_D_INITIAL_SIGMA,
    PHASE_D_NOISE_POLICY_V0,
    PHASE_D_SIGMA_MAX,
    PHASE_D_SIGMA_MIN,
    TorxOperatorConfig,
    TorxOps,
    rho_from_sigma,
    sigma_from_rho,
    stable_occurrence_id,
)
from adze_t.model import apply_model, init_model_params


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results" / "phase_d_1" / "d0"
SAMPLES = 4096


def _write(name: str, value: Any) -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)

    def default(item):
        array = jax.device_get(item)
        return float(array) if getattr(array, "ndim", 0) == 0 else array.tolist()

    (OUTPUT / name).write_text(
        json.dumps(value, indent=2, sort_keys=True, default=default) + "\n", encoding="utf-8"
    )


def _config(lambda_op=1.0):
    return TorxOperatorConfig(operator_stochasticity=True, lambda_op=lambda_op)


def _primitives():
    deterministic = {
        "affine": {
            "weight": jnp.array([[0.25, -0.5], [0.75, 0.125]], dtype=jnp.float32),
            "bias": jnp.array([0.2, -0.1], dtype=jnp.float32),
        },
        "embedding": jnp.arange(12, dtype=jnp.float32).reshape(6, 2) / 9,
        "conv": {
            "kernel": jnp.arange(6, dtype=jnp.float32).reshape(3, 2) / 11,
            "bias": jnp.array([0.15, -0.2], dtype=jnp.float32),
        },
    }
    return deterministic_to_torx(deterministic)[0]


def _factor_value(kind, params, key, lambda_op=1.0):
    ops = TorxOps.create(key, config=_config(lambda_op))
    x = jnp.array([[[0.3, -0.7], [0.5, 0.2]]], dtype=jnp.float32)
    indices = jnp.array([[1, 4]], dtype=jnp.int32)
    if kind == "affine":
        return ops.linear(x, params["affine"], name="oracle.affine")[0, 0, 0]
    if kind == "categorical_logits":
        return ops.categorical_logits(x, params["affine"], name="oracle.logits")[0, 0, 0]
    if kind == "embedding":
        return ops.embedding(indices, params["embedding"], name="oracle.embedding")[0, 0, 0]
    return ops.depthwise_conv1d(x, params["conv"], name="oracle.conv")[0, 0, 0]


def primitive_evidence() -> dict[str, Any]:
    params = _primitives()
    keys = jax.random.split(jax.random.key(6000), SAMPLES)
    moment_records = []
    variance = PHASE_D_INITIAL_SIGMA**2
    mean_limit = 5 * PHASE_D_INITIAL_SIGMA / jnp.sqrt(SAMPLES)
    variance_limit = 5 * variance * jnp.sqrt(2 / (SAMPLES - 1))
    for kind in ("affine", "categorical_logits", "embedding", "depthwise_conv1d"):
        values = jax.jit(jax.vmap(lambda key: _factor_value(kind, params, key)))(keys)
        analytic_mean = _factor_value(kind, params, jax.random.key(0), 0.0)
        sample_mean = jnp.mean(values)
        sample_variance = jnp.var(values, ddof=1)
        mean_error = jnp.abs(sample_mean - analytic_mean)
        variance_error = jnp.abs(sample_variance - variance)
        moment_records.append(
            {
                "factor": kind,
                "samples": SAMPLES,
                "analytic_mean": analytic_mean,
                "empirical_mean": sample_mean,
                "mean_absolute_error": mean_error,
                "five_mean_standard_errors": mean_limit,
                "analytic_variance": variance,
                "unbiased_sample_variance": sample_variance,
                "variance_absolute_error": variance_error,
                "five_gaussian_variance_estimator_standard_errors": variance_limit,
                "passed": bool(
                    (mean_error <= mean_limit + 1.0e-7)
                    & (variance_error <= variance_limit + 1.0e-9)
                ),
            }
        )

    affine = params["affine"]
    x = jnp.array([[0.3, -0.7], [0.5, 0.2]], dtype=jnp.float32)
    indices = jnp.array([[1, 4]], dtype=jnp.int32)
    root = jax.random.key(6001)
    occurrence_key = TorxOps.create(root, config=_config()).context.key_for("oracle.affine")

    def affine_factor_objective(input_x, parameter):
        y = TorxOps.create(root, config=_config()).linear(input_x, parameter, name="oracle.affine")
        return 0.5 * jnp.sum(y**2)

    def affine_reference_objective(input_x, parameter):
        mean = input_x @ parameter["mean"]["weight"] + parameter["mean"]["bias"]
        epsilon = jax.random.normal(occurrence_key, mean.shape, dtype=mean.dtype)
        y = mean + sigma_from_rho(parameter["rho"]) * epsilon
        return 0.5 * jnp.sum(y**2)

    actual = jax.grad(affine_factor_objective, argnums=(0, 1))(x, affine)
    expected = jax.grad(affine_reference_objective, argnums=(0, 1))(x, affine)
    fixed_errors = [
        jnp.max(jnp.abs(left - right))
        for left, right in zip(
            jax.tree_util.tree_leaves(actual), jax.tree_util.tree_leaves(expected), strict=True
        )
    ]

    def primitive_inputs(kind):
        return (
            x[None, :, :] if kind == "depthwise_conv1d" else x,
            indices,
            params[
                "affine"
                if kind in ("affine", "categorical_logits")
                else "embedding"
                if kind == "embedding"
                else "conv"
            ],
        )

    def factor_objective(kind, root, input_x, parameter):
        ops = TorxOps.create(root, config=_config())
        if kind == "affine":
            output = ops.linear(input_x, parameter, name="oracle.affine")
        elif kind == "categorical_logits":
            output = ops.categorical_logits(input_x, parameter, name="oracle.logits")
        elif kind == "embedding":
            output = ops.embedding(indices, parameter, name="oracle.embedding")
        else:
            output = ops.depthwise_conv1d(input_x, parameter, name="oracle.conv")
        return 0.5 * jnp.sum(output**2)

    def reference_objective(kind, root, input_x, parameter):
        name = {
            "affine": "oracle.affine",
            "categorical_logits": "oracle.logits",
            "embedding": "oracle.embedding",
            "depthwise_conv1d": "oracle.conv",
        }[kind]
        if kind in ("affine", "categorical_logits"):
            mean_value = input_x @ parameter["mean"]["weight"] + parameter["mean"]["bias"]
        elif kind == "embedding":
            mean_value = parameter["mean"][indices]
        else:
            kernel = parameter["mean"]["kernel"][:, None, :]
            padded = jnp.pad(input_x, ((0, 0), (kernel.shape[0] - 1, 0), (0, 0)))
            mean_value = (
                jax.lax.conv_general_dilated(
                    padded,
                    kernel,
                    (1,),
                    "VALID",
                    dimension_numbers=("NWC", "WIO", "NWC"),
                    feature_group_count=input_x.shape[-1],
                )
                + parameter["mean"]["bias"]
            )
        epsilon = jax.random.normal(
            TorxOps.create(root, config=_config()).context.key_for(name),
            mean_value.shape,
            dtype=mean_value.dtype,
        )
        return 0.5 * jnp.sum((mean_value + sigma_from_rho(parameter["rho"]) * epsilon) ** 2)

    fixed_key_records = []
    mc_gradient_records = []
    for kind in ("affine", "categorical_logits", "embedding", "depthwise_conv1d"):
        input_x, _, parameter = primitive_inputs(kind)
        finite_root = jax.random.key(6010)
        actual_gradient = jax.grad(
            lambda a, p: factor_objective(kind, finite_root, a, p), argnums=(0, 1)
        )(input_x, parameter)
        reference_gradient = jax.grad(
            lambda a, p: reference_objective(kind, finite_root, a, p), argnums=(0, 1)
        )(input_x, parameter)
        errors = [
            jnp.max(jnp.abs(left - right))
            for left, right in zip(
                jax.tree_util.tree_leaves(actual_gradient),
                jax.tree_util.tree_leaves(reference_gradient),
                strict=True,
            )
        ]
        fixed_key_records.append(
            {
                "factor": kind,
                "lambda_op": 1.0,
                "max_absolute_error": max(errors),
                "rho_gradient_max": jnp.max(jnp.abs(actual_gradient[1]["rho"])),
                "passed": bool(
                    max(errors) <= 2e-10 and jnp.max(jnp.abs(actual_gradient[1]["rho"])) > 0
                ),
            }
        )

        def expected_objective(a, p):
            # Independent JAX expectation: E[.5||mean+sigma eps||²].
            if kind in ("affine", "categorical_logits"):
                mean_value = a @ p["mean"]["weight"] + p["mean"]["bias"]
            elif kind == "embedding":
                mean_value = p["mean"][indices]
            else:
                kernel = p["mean"]["kernel"][:, None, :]
                padded = jnp.pad(a, ((0, 0), (kernel.shape[0] - 1, 0), (0, 0)))
                mean_value = (
                    jax.lax.conv_general_dilated(
                        padded,
                        kernel,
                        (1,),
                        "VALID",
                        dimension_numbers=("NWC", "WIO", "NWC"),
                        feature_group_count=a.shape[-1],
                    )
                    + p["mean"]["bias"]
                )
            sigma = sigma_from_rho(p["rho"])
            return 0.5 * jnp.sum(mean_value**2) + 0.5 * (mean_value.size / sigma.size) * jnp.sum(
                sigma**2
            )

        samples = jax.jit(
            jax.vmap(
                lambda sample_key: jax.grad(
                    lambda a, p: factor_objective(kind, sample_key, a, p), argnums=(0, 1)
                )(input_x, parameter)
            )
        )(keys)
        analytic_gradient = jax.grad(expected_objective, argnums=(0, 1))(input_x, parameter)
        leaves = []
        for observed, expected_value in zip(
            jax.tree_util.tree_leaves(samples),
            jax.tree_util.tree_leaves(analytic_gradient),
            strict=True,
        ):
            sem = jnp.std(observed, axis=0, ddof=1) / jnp.sqrt(SAMPLES)
            error = jnp.abs(jnp.mean(observed, axis=0) - expected_value)
            leaves.append(bool(jnp.all(error <= 5 * sem + 2e-6)))
        mc_gradient_records.append(
            {
                "factor": kind,
                "samples": SAMPLES,
                "oracle": "independent_expected_jax",
                "passed": all(leaves),
            }
        )

    gradient_x = jnp.array([0.3, -0.7], dtype=jnp.float32)

    def one_gradient(key):
        def objective(input_x, parameter):
            y = TorxOps.create(key, config=_config()).linear(
                input_x, parameter, name="oracle.affine"
            )
            return 0.5 * jnp.sum(y**2)

        return jax.grad(objective, argnums=(0, 1))(gradient_x, affine)

    gradient_samples = jax.jit(jax.vmap(one_gradient))(keys)
    weight = affine["mean"]["weight"]
    bias = affine["mean"]["bias"]
    mean = gradient_x @ weight + bias
    sigma = sigma_from_rho(affine["rho"])
    analytic = (
        mean @ weight.T,
        {
            "mean": {"weight": gradient_x[:, None] * mean[None, :], "bias": mean},
            "rho": sigma * jax.nn.sigmoid(affine["rho"]),
        },
    )
    coordinate_records = []
    names = ("input", "bias", "weight", "rho")
    for name, observed, expected_value in zip(
        names,
        jax.tree_util.tree_leaves(gradient_samples),
        jax.tree_util.tree_leaves(analytic),
        strict=True,
    ):
        empirical = jnp.mean(observed, axis=0)
        sem = jnp.std(observed, axis=0, ddof=1) / jnp.sqrt(SAMPLES)
        error = jnp.abs(empirical - expected_value)
        limit = 5 * sem + 2.0e-6
        coordinate_records.append(
            {
                "leaf": name,
                "coordinates": int(expected_value.size),
                "max_absolute_error": jnp.max(error),
                "max_five_sem_plus_floor": jnp.max(limit),
                "all_coordinates_passed": bool(jnp.all(error <= limit)),
            }
        )

    zero_ops = TorxOps.create(jax.random.key(6002), config=_config(0.0))

    def zero_objective(rho):
        changed = {**affine, "rho": rho}
        return jnp.sum(zero_ops.linear(x, changed, name="oracle.affine") ** 2)

    lambda_scaling = []
    for kind in ("affine", "categorical_logits", "embedding", "depthwise_conv1d"):
        clean = _factor_value(kind, params, jax.random.key(6003), 0.0)
        unit = _factor_value(kind, params, jax.random.key(6003), 1.0)
        for lambda_op in (0.1, 0.25, 0.5, 1.0):
            value = _factor_value(kind, params, jax.random.key(6003), lambda_op)
            error = jnp.abs((value - clean) - lambda_op * (unit - clean))
            lambda_scaling.append(
                {
                    "factor": kind,
                    "lambda_op": lambda_op,
                    "residual_scaling_error": error,
                    "passed": bool(error < 1e-7),
                }
            )

    policy = {
        "name": PHASE_D_NOISE_POLICY_V0,
        "direct_parameter_policy": PHASE_D_DIRECT_PARAMETER_NOISE_POLICY_V0_MEAN_ONLY,
        "initial_sigma": PHASE_D_INITIAL_SIGMA,
        "sigma_min": PHASE_D_SIGMA_MIN,
        "sigma_max": PHASE_D_SIGMA_MAX,
        "transformed_initial_sigma": sigma_from_rho(rho_from_sigma(PHASE_D_INITIAL_SIGMA)),
        "lower_clamp_observed": sigma_from_rho(jnp.asarray(-100.0)),
        "upper_clamp_observed": sigma_from_rho(jnp.asarray(100.0)),
    }
    passed = (
        all(item["passed"] for item in moment_records)
        and max(float(value) for value in fixed_errors) <= 1.0e-9
        and all(item["passed"] for item in fixed_key_records)
        and all(item["passed"] for item in mc_gradient_records)
        and all(item["all_coordinates_passed"] for item in coordinate_records)
        and all(item["passed"] for item in lambda_scaling)
        and bool(
            jnp.array_equal(jax.grad(zero_objective)(affine["rho"]), jnp.zeros_like(affine["rho"]))
        )
    )
    return {
        "policy": policy,
        "moments": moment_records,
        "fixed_key_reparameterized_gradient_max_errors": fixed_errors,
        "fixed_key_gradients_by_family": fixed_key_records,
        "mc_expected_gradients_by_family": mc_gradient_records,
        "mean_pathwise_gradients": coordinate_records,
        "lambda_residual_scaling": lambda_scaling,
        "lambda_zero_rho_gradient_max": jnp.max(jnp.abs(jax.grad(zero_objective)(affine["rho"]))),
        "passed": passed,
    }


def occurrence_evidence() -> dict[str, Any]:
    params = _primitives()
    x = jnp.array([[0.3, -0.7]], dtype=jnp.float32)
    roots = jax.random.split(jax.random.key(6004), SAMPLES)

    def q_residual(root, cycle):
        ops = (
            TorxOps.create(root, config=_config())
            .with_scope("mode:draft")
            .with_occurrence(recurrence_cycle=cycle, physical_layer=0)
        )
        noisy = ops.linear(x, params["affine"], name="dit.block_0.q")
        clean = TorxOps(ops.context, _config(0.0)).linear(x, params["affine"], name="dit.block_0.q")
        return noisy[0, 0] - clean[0, 0]

    q_samples = jax.jit(
        jax.vmap(lambda root: jnp.stack([q_residual(root, cycle) for cycle in range(3)]))
    )(roots)
    q_variances = jnp.var(q_samples, axis=0, ddof=1)
    expected_variance = PHASE_D_INITIAL_SIGMA**2
    variance_limit = 5 * expected_variance * jnp.sqrt(2 / (SAMPLES - 1)) + 1.0e-9

    deterministic = init_model_params(jax.random.key(6005))
    model_params, mapping = deterministic_to_torx(deterministic)
    selected = []
    occurrences = []

    def sample_observer(kind, name, path, key, mean, output):
        if name in ("frontend.byte_embed", "dit.block_0.q"):
            selected.append(
                {
                    "kind": kind,
                    "name": name,
                    "path": path,
                    "key": jax.random.key_data(key),
                    "mean_rms": jnp.sqrt(jnp.mean(mean**2)),
                    "residual_rms": jnp.sqrt(jnp.mean((output - mean) ** 2)),
                    "residual": output - mean,
                    "mean": mean,
                }
            )

    def occurrence_observer(kind, name, path, key):
        occurrences.append({"kind": kind, "name": name, "path": path, "key": key})

    values = jnp.arange(1, 9, dtype=jnp.int32)[None]
    mask = jnp.ones_like(values, dtype=bool)

    def execute_full_model():
        apply_model(
            model_params,
            values,
            mask,
            values,
            mask,
            ops=TorxOps.create(
                jax.random.key(6006),
                config=_config(),
                sample_observer=sample_observer,
                occurrence_observer=occurrence_observer,
            ),
        )

    execute_full_model()
    first_selected = list(selected)
    first_occurrences = list(occurrences)
    selected.clear()
    occurrences.clear()
    execute_full_model()
    repeated = list(selected)
    reproducible = all(
        left["path"] == right["path"]
        and jnp.array_equal(left["key"], right["key"])
        and jnp.array_equal(left["mean"], right["mean"])
        and jnp.array_equal(left["residual"], right["residual"])
        for left, right in zip(first_selected, repeated, strict=True)
    )
    selected = first_selected
    occurrences = first_occurrences
    identities = {
        *(f"module:{item['name']}" for item in occurrences),
        *(
            f"scope:{scope}"
            for item in occurrences
            for scope in item["path"].rsplit("/", 1)[0].split("/")
            if "/" in item["path"]
        ),
    }
    identity_ids = {identity: stable_occurrence_id(identity) for identity in identities}
    frontend = [item for item in selected if item["name"] == "frontend.byte_embed"]
    q_records = [item for item in selected if item["name"] == "dit.block_0.q"]
    residuals_distinct = (
        not jnp.array_equal(frontend[0]["residual"], frontend[1]["residual"])
        and len({tuple(map(int, item["key"])) for item in q_records}) == 3
        and all(
            not jnp.array_equal(left["residual"], right["residual"])
            for left, right in zip(q_records, q_records[1:], strict=False)
        )
    )
    frontend_means_equal = jnp.array_equal(frontend[0]["mean"], frontend[1]["mean"])
    summary = [
        {key: value for key, value in item.items() if key not in ("residual", "mean")}
        for item in selected
    ]
    q_destination = [
        entry.torx_path for entry in mapping if entry.deterministic_path == "dit/blocks/0/q/weight"
    ]
    passed = bool(
        jnp.all(jnp.abs(q_variances - expected_variance) <= variance_limit)
        and residuals_distinct
        and frontend_means_equal
        and reproducible
        and len(set(identity_ids.values())) == len(identity_ids)
        and q_destination == ["dit/blocks/0/q/mean/weight"]
    )
    return {
        "samples_per_q_occurrence": SAMPLES,
        "q_local_unbiased_variances": q_variances,
        "analytic_local_variance": expected_variance,
        "five_variance_standard_errors_plus_floor": variance_limit,
        "q_or_effective_depth_normalization": False,
        "full_model_selected_occurrences": summary,
        "frontend_output_means_equal_for_equal_inputs": bool(frontend_means_equal),
        "distinct_reproducible_residuals": bool(residuals_distinct),
        "same_root_repeat_bitwise_reproducible": bool(reproducible),
        "block_0_q_mean_destination": q_destination,
        "observed_scope_module_ids": identity_ids,
        "observed_scope_module_ids_collision_free": len(set(identity_ids.values()))
        == len(identity_ids),
        "passed": passed,
    }


def main() -> None:
    primitive = primitive_evidence()
    occurrence = occurrence_evidence()
    _write("primitive_moments_gradients.json", primitive)
    _write("fixed_key_gradients.json", primitive["fixed_key_gradients_by_family"])
    _write("mc_gradient_oracles.json", primitive["mc_expected_gradients_by_family"])
    _write("occurrence_noise.json", occurrence)
    passed = primitive["passed"] and occurrence["passed"]
    decision = "D0_FINITE_NOISE_CONTRACT_PASS" if passed else "D0_FINITE_NOISE_CONTRACT_FAILURE"
    (OUTPUT / "D0_NOISE_CONTRACT.md").write_text(
        "# D0 — finite-noise contract\n\n"
        f"Decision: **{decision}**.\n\n"
        f"Policy: `{PHASE_D_NOISE_POLICY_V0}`; sigma `{PHASE_D_INITIAL_SIGMA}`; rho frozen. "
        f"Direct coefficients: `{PHASE_D_DIRECT_PARAMETER_NOISE_POLICY_V0_MEAN_ONLY}`.\n\n"
        "The committed JSON records contain 4,096-sample unbiased moment oracles, fixed-key "
        "and mean pathwise gradient comparisons, lambda scaling, Q-local variance, full-model "
        "occurrence residuals, and observed scope/module collision checks.\n",
        encoding="utf-8",
    )
    print(decision)
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
