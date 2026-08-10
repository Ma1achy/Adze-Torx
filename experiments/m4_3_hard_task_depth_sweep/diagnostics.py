"""M4.3 initialization, convergence, and trajectory diagnostics."""

from __future__ import annotations

from math import sqrt

import jax
import jax.numpy as jnp
import optax

from experiments.m4_3_hard_task_depth_sweep.model import apply_core_trace, deterministic_trace

T_CRITICAL_95 = {3: 4.303, 5: 2.776}


def paired_t_interval(q1: list[float], q_other: list[float]) -> dict[str, float]:
    differences = jnp.asarray(q1) - jnp.asarray(q_other)
    n = len(differences)
    mean = float(jnp.mean(differences))
    standard_error = float(jnp.std(differences, ddof=1) / sqrt(n)) if n > 1 else float("inf")
    critical = T_CRITICAL_95.get(n, 2.0)
    return {
        "mean_delta_q1_minus_q": mean,
        "ci_low": mean - critical * standard_error,
        "ci_high": mean + critical * standard_error,
        "standard_error": standard_error,
    }


def initial_report(config, params, key):
    state = jnp.linspace(-0.5, 0.5, config.width)
    _, blocks, cycles = deterministic_trace(config, params, state)
    samples = jax.vmap(lambda sample_key: apply_core_trace(config, params, state, sample_key)[0])(
        jax.random.split(key, 64)
    )

    def objective(current):
        output = deterministic_trace(config, current, state)[0]
        return jnp.mean(output[:8] ** 2)

    gradients = jax.grad(objective)(params)
    optimiser = optax.adam(0.03)
    updates, _ = optimiser.update(gradients, optimiser.init(params), params)
    return {
        "state_error": float(jnp.linalg.norm(cycles[-1] - state)),
        "parameter_count": sum(leaf.size for leaf in jax.tree_util.tree_leaves(params)),
        "nominal_variance": config.total_variance,
        "post_stack_variance": float(jnp.mean(jnp.var(samples, axis=0))),
        "initial_gradient_norm": float(
            jnp.sqrt(sum(jnp.sum(x**2) for x in jax.tree_util.tree_leaves(gradients)))
        ),
        "initial_update_norm": float(
            jnp.sqrt(sum(jnp.sum(x**2) for x in jax.tree_util.tree_leaves(updates)))
        ),
        "block_conditioning_error": float(jnp.max(jnp.abs(blocks[:, 8:] - state[8:]))),
    }


def trajectory_report(result, keys):
    config = result["core"]
    batch = result["validation_batch"]
    trajectories = jax.vmap(
        lambda state, key: apply_core_trace(config, result["params"], state, key)[2]
    )(batch.initial, keys)
    target = batch.target[:, :8]
    rows = []
    errors = []
    for q in range(config.q + 1):
        error = jnp.mean((trajectories[:, q, :8] - target) ** 2, axis=-1)
        errors.append(error)
        rows.append(
            {
                "q": q,
                "mse": float(jnp.mean(error)),
                "update_norm": 0.0
                if q == 0
                else float(
                    jnp.mean(jnp.linalg.norm(trajectories[:, q] - trajectories[:, q - 1], axis=-1))
                ),
                "improve_fraction": 0.0 if q == 0 else float(jnp.mean(error < errors[q - 1])),
            }
        )
    return rows


def convergence_limited(
    validation: list[dict[str, float]], relative_threshold: float = 0.01
) -> bool:
    if len(validation) < 3:
        return False
    previous = validation[-3]["loss"]
    latest = validation[-1]["loss"]
    return (previous - latest) / max(abs(previous), 1e-8) > relative_threshold
