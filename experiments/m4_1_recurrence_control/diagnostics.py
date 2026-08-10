"""Initialization and trajectory diagnostics for M4.1."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp

from adze_t.model.core import (
    RecurrentCoreConfig,
    apply_core,
    deterministic_core,
    effective_linear_map,
    initialise_params,
    nominal_accumulated_variance,
)
from experiments.m4_1_recurrence_control.model import M4Config, local_features, predict_batch


def initial_map_report(config: RecurrentCoreConfig, key: jax.Array) -> dict[str, float | jax.Array]:
    params = initialise_params(config, key)
    state = jnp.linspace(-1.0, 1.0, config.width)
    output = deterministic_core(config, params, state)
    linear = effective_linear_map(config, params)
    singular = jnp.linalg.svd(linear, compute_uv=False)[0]
    spectral = jnp.max(jnp.abs(jnp.linalg.eigvals(linear)))
    return {
        "state_error": float(jnp.linalg.norm(output - state)),
        "linear_error": float(jnp.linalg.norm(linear - jnp.eye(config.width))),
        "spectral_radius": float(jnp.real(spectral)),
        "largest_singular": float(singular),
        "nominal_variance": nominal_accumulated_variance(config),
    }


def _boundary_f1(predicted: jax.Array, target: jax.Array, observed: jax.Array) -> jax.Array:
    nonforced = jnp.arange(target.shape[-1]) > 0
    unknown = (observed == 2) & nonforced
    positive = target == 1
    predicted_positive = predicted == 1
    tp = jnp.sum(unknown & positive & predicted_positive)
    fp = jnp.sum(unknown & ~positive & predicted_positive)
    fn = jnp.sum(unknown & positive & ~predicted_positive)
    precision = tp / jnp.maximum(tp + fp, 1)
    recall = tp / jnp.maximum(tp + fn, 1)
    return 2 * precision * recall / jnp.maximum(precision + recall, 1e-8)


def trajectory_report(
    config: M4Config,
    params: Any,
    data,
    keys: jax.Array,
    alpha: float = 0.6,
    sigma: float = 0.5,
    rho_b: float = 0.5,
    rho_length: float = 0.5,
    sample_count: int = 4,
) -> list[dict[str, float | int]]:
    predictions = predict_batch(
        params, data, keys, config, alpha, sigma, rho_b, rho_length, return_trajectory=True
    )
    if predictions.trajectory is None:
        raise ValueError("trajectory diagnostics require return_trajectory=True")
    trajectories = predictions.trajectory.reshape(
        (data.h.shape[0], config.core.q + 1, config.structure.capacity, config.structure.latent_dim)
    )
    per_example_mse = jnp.mean((trajectories - data.h[:, None, :, :]) ** 2, axis=(2, 3))
    h_mse = jnp.mean(per_example_mse, axis=0)
    mean_trajectories = jax.vmap(
        lambda state: deterministic_core(config.core, params["core"], state, True)
    )(predictions.h_corrupt.reshape((data.h.shape[0], -1)))
    mean_trajectories = mean_trajectories.reshape(trajectories.shape)
    mean_mse = jnp.mean((mean_trajectories - data.h[:, None, :, :]) ** 2, axis=(0, 2, 3))
    updates = jnp.mean(
        jnp.linalg.norm(
            jnp.diff(trajectories.reshape((data.h.shape[0], config.core.q + 1, -1)), axis=1),
            axis=-1,
        ),
        axis=0,
    )
    states = jnp.mean(
        jnp.linalg.norm(trajectories.reshape((data.h.shape[0], config.core.q + 1, -1)), axis=-1),
        axis=0,
    )
    improve_fraction = jnp.mean(per_example_mse[:, 1:] < per_example_mse[:, :-1], axis=0)
    sample_keys = jax.random.split(jax.random.fold_in(keys[0], 991), sample_count * data.h.shape[0])
    sample_keys = sample_keys.reshape((sample_count, data.h.shape[0]))
    sampled = jax.vmap(
        lambda key_batch: jax.vmap(
            lambda state, key: apply_core(config.core, params["core"], state.reshape(-1), key, True)
        )(predictions.h_corrupt, key_batch)
    )(sample_keys)
    sampled = sampled.reshape((sample_count, data.h.shape[0], config.core.q + 1, -1))
    variance = jnp.mean(jnp.var(sampled, axis=0), axis=-1)
    reports = []
    for cycle in range(config.core.q + 1):
        state = trajectories[:, cycle]
        features = jax.vmap(lambda h, b, length: local_features(h, b, length, config))(
            state, predictions.observed_b, predictions.observed_length
        )
        boundary_logits = (
            features @ params["boundary"]["boundary_weight"] + params["boundary"]["boundary_bias"]
        )
        length_logits = (
            features @ params["length"]["length_weight"] + params["length"]["length_bias"]
        )
        boundary_pred = jnp.argmax(boundary_logits, axis=-1)
        length_pred = jnp.argmax(length_logits, axis=-1)
        length_unknown = predictions.observed_length == config.structure.length_unknown
        correct_unknown = jnp.logical_and(length_unknown, length_pred == data.length)
        length_acc = jnp.sum(correct_unknown) / jnp.maximum(jnp.sum(length_unknown), 1)
        reports.append(
            {
                "cycle": cycle,
                "h_mse": float(h_mse[cycle]),
                "mean_h_mse": float(mean_mse[cycle]),
                "delta_mse": 0.0 if cycle == 0 else float(h_mse[cycle] - h_mse[cycle - 1]),
                "improve_fraction": 0.0 if cycle == 0 else float(improve_fraction[cycle - 1]),
                "boundary_f1": float(_boundary_f1(boundary_pred, data.b, predictions.observed_b)),
                "length_accuracy": float(length_acc),
                "update_norm": 0.0 if cycle == 0 else float(updates[cycle - 1]),
                "state_norm": float(states[cycle]),
                "variance": float(jnp.mean(variance[:, cycle])),
            }
        )
    return reports
