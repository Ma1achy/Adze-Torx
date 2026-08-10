"""Forward and recurrence diagnostics for M4."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp

from adze_t.model.core import (
    RecurrentCoreConfig,
    apply_core,
    deterministic_core,
    nominal_accumulated_variance,
    prefix_config,
)


def trajectory_summary(
    config: RecurrentCoreConfig,
    params: Any,
    initial: jax.Array,
    target: jax.Array,
    key: jax.Array,
    sample_count: int = 8,
) -> dict[str, jax.Array | float]:
    trajectory = apply_core(config, params, initial, key, return_trajectory=True)
    errors = jnp.mean((trajectory - target[None, :]) ** 2, axis=-1)
    updates = jnp.concatenate(
        (jnp.zeros((1,)), jnp.linalg.norm(jnp.diff(trajectory, axis=0), axis=-1))
    )
    norms = jnp.linalg.norm(trajectory, axis=-1)
    sample_keys = jax.random.split(jax.random.fold_in(key, 991), sample_count)
    sample_trajectories = jax.vmap(
        lambda sample_key: apply_core(config, params, initial, sample_key, return_trajectory=True)
    )(sample_keys)
    variances = jnp.mean(jnp.var(sample_trajectories, axis=0), axis=-1)
    mean_trajectory = deterministic_core(config, params, initial, return_trajectory=True)
    mean_errors = jnp.mean((mean_trajectory - target[None, :]) ** 2, axis=-1)
    return {
        "h_mse_by_cycle": errors,
        "mean_h_mse_by_cycle": mean_errors,
        "update_norm_by_cycle": updates,
        "state_norm_by_cycle": norms,
        "variance_by_cycle": variances,
        "nominal_accumulated_variance": nominal_accumulated_variance(config),
    }


def current_prefix_diagnostics(
    config: RecurrentCoreConfig,
    params: Any,
    initial: jax.Array,
    target: jax.Array,
    key: jax.Array,
) -> list[dict[str, jax.Array | float]]:
    """Compare prefixes using the same family/noise convention."""
    return [
        trajectory_summary(prefix_config(config, q), params, initial, target, key)
        for q in range(1, config.q + 1)
    ]
