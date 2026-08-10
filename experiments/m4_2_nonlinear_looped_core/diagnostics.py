"""Interpretability and evaluation diagnostics for M4.2."""

from __future__ import annotations

from typing import Any

import jax
import jax.nn as jnn
import jax.numpy as jnp

from experiments.m3_carrier_structure.data import StructuredCarrier
from experiments.m4_2_nonlinear_looped_core.model import (
    M42Config,
    NonlinearCoreConfig,
    apply_core,
    deterministic_core,
    predict_reconstruction,
)


def boundary_f1(logits: jax.Array, target: jax.Array) -> jax.Array:
    prediction = jnp.argmax(logits, axis=-1)
    true_positive = jnp.sum((prediction == 1) & (target == 1))
    precision = true_positive / jnp.maximum(jnp.sum(prediction == 1), 1)
    recall = true_positive / jnp.maximum(jnp.sum(target == 1), 1)
    return 2 * precision * recall / jnp.maximum(precision + recall, 1e-8)


def reconstruction_metrics(
    params: Any,
    batch: StructuredCarrier,
    keys: jax.Array,
    config: M42Config,
) -> dict[str, jax.Array]:
    predictions = jax.vmap(lambda target, key: predict_reconstruction(params, target, key, config))(
        batch, keys
    )
    h_mse = jnp.mean((predictions.h - batch.h) ** 2)
    observed_mse = jnp.mean((predictions.h_corrupt - batch.h) ** 2)
    boundary_loss = -jnp.take_along_axis(
        jnn.log_softmax(predictions.boundary_logits), batch.b[..., None], -1
    )[..., 0]
    length_loss = -jnp.take_along_axis(
        jnn.log_softmax(predictions.length_logits), batch.length[..., None], -1
    )[..., 0]
    return {
        "loss": h_mse + jnp.mean(boundary_loss) + jnp.mean(length_loss),
        "h_mse": h_mse,
        "corrupt_h_mse": observed_mse,
        "b_f1": boundary_f1(predictions.boundary_logits, batch.b),
        "length_accuracy": jnp.mean(jnp.argmax(predictions.length_logits, axis=-1) == batch.length),
    }


def initial_map_report(config: NonlinearCoreConfig, params, key: jax.Array) -> dict[str, float]:
    state = jnp.linspace(-0.5, 0.5, config.width)
    trajectory = deterministic_core(config, params, state, return_trajectory=True)
    samples = jax.vmap(lambda sample_key: apply_core(config, params, state, sample_key))(
        jax.random.split(key, 64)
    )
    return {
        "state_error": float(jnp.linalg.norm(trajectory[-1] - state)),
        "nominal_pre_nonlinearity_variance": config.total_variance,
        "spectral_radius": float(jnp.max(jnp.abs(jnp.linalg.eigvals(jnp.eye(config.width))))),
        "post_output_variance": float(jnp.mean(jnp.var(samples, axis=0))),
    }


def nonlinearity_report(config: NonlinearCoreConfig, params) -> dict[str, float]:
    x = jnp.linspace(-0.7, 0.7, config.width)
    y = jnp.linspace(0.4, -0.4, config.width)
    mixed = 0.35 * x + 0.65 * y
    fx = deterministic_core(config, params, x)
    fy = deterministic_core(config, params, y)
    fm = deterministic_core(config, params, mixed)
    defect = jnp.linalg.norm(fm - (0.35 * fx + 0.65 * fy))
    jacobian_x = jax.jacobian(lambda z: deterministic_core(config, params, z))(x)
    jacobian_y = jax.jacobian(lambda z: deterministic_core(config, params, z))(y)
    return {
        "affine_defect": float(defect),
        "jacobian_state_dependence": float(jnp.linalg.norm(jacobian_x - jacobian_y)),
    }


def trajectory_metrics(
    config: NonlinearCoreConfig, params, batch, keys, composition: bool = True
) -> list[dict[str, float]]:
    states = jax.vmap(lambda state, key: apply_core(config, params, state, key, True))(
        batch.initial, keys
    )
    rows = []
    for q in range(config.q + 1):
        state = states[:, q]
        if composition:
            error = jnp.mean((state[:, :3] - batch.target[:, :3]) ** 2, axis=-1)
        else:
            error = jnp.mean((state - batch.target) ** 2, axis=-1)
        update = (
            0.0 if q == 0 else jnp.mean(jnp.linalg.norm(states[:, q] - states[:, q - 1], axis=-1))
        )
        prior = (
            0.0
            if q == 0
            else jnp.mean(
                error < (jnp.mean((states[:, q - 1, :3] - batch.target[:, :3]) ** 2, axis=-1))
            )
        )
        rows.append(
            {
                "q": float(q),
                "mse": float(jnp.mean(error)),
                "update_norm": float(update),
                "improve_fraction": float(prior),
            }
        )
    return rows


def fixed_output_variance(
    config: NonlinearCoreConfig, params, state: jax.Array, keys: jax.Array
) -> float:
    samples = jax.vmap(lambda key: apply_core(config, params, state, key))(keys)
    return float(jnp.mean(jnp.var(samples, axis=0)))
