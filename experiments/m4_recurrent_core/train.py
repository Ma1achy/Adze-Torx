"""Reproducible M4 training and metric collection."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, cast

import jax
import jax.numpy as jnp
import optax

from adze_t.model.carrier import BOUNDARY_UNKNOWN, StructureConfig
from experiments.m3_carrier_structure.data import StructuredCarrier
from experiments.m4_recurrent_core.model import M4Config, batch_loss, predict_batch


@dataclass(frozen=True)
class TrainConfig:
    steps: int = 60
    batch_size: int = 64
    learning_rate: float = 0.03
    alpha: float = 0.6
    sigma: float = 0.5
    rho_b: float = 0.5
    rho_length: float = 0.5
    seed: int = 0


def _keys(seed: int, step: int, n: int) -> jax.Array:
    return jax.random.split(jax.random.key(seed + 1009 * step), n)


def _tree_finite(tree: Any) -> bool:
    return bool(
        jnp.all(
            jnp.asarray(
                [
                    jnp.all(jnp.isfinite(cast(jax.Array, leaf)))
                    for leaf in jax.tree_util.tree_leaves(tree)
                ]
            )
        )
    )


def _boundary_metrics(
    predicted: jax.Array, target: jax.Array, observed: jax.Array
) -> dict[str, jax.Array]:
    nonforced = jnp.arange(target.shape[-1]) > 0
    unknown = (observed == BOUNDARY_UNKNOWN) & nonforced
    visible = (observed != BOUNDARY_UNKNOWN) & nonforced
    correct = predicted == target
    positive = target == 1
    predicted_positive = predicted == 1
    true_positive = jnp.sum(unknown & positive & predicted_positive)
    false_positive = jnp.sum(unknown & ~positive & predicted_positive)
    false_negative = jnp.sum(unknown & positive & ~predicted_positive)
    precision = true_positive / jnp.maximum(true_positive + false_positive, 1)
    recall = true_positive / jnp.maximum(true_positive + false_negative, 1)
    return {
        "overall_accuracy": jnp.mean(correct),
        "nonforced_accuracy": jnp.mean(jnp.where(nonforced, correct, 0.0))
        / jnp.maximum(jnp.mean(nonforced), 1e-8),
        "unknown_accuracy": jnp.sum(unknown & correct) / jnp.maximum(jnp.sum(unknown), 1),
        "observed_accuracy": jnp.sum(visible & correct) / jnp.maximum(jnp.sum(visible), 1),
        "unknown_precision": precision,
        "unknown_recall": recall,
        "unknown_f1": 2 * precision * recall / jnp.maximum(precision + recall, 1e-8),
    }


def _length_metrics(
    predicted: jax.Array, target: jax.Array, observed: jax.Array, config: StructureConfig
) -> dict[str, jax.Array]:
    unknown = observed == config.length_unknown
    visible = ~unknown
    correct = predicted == target
    zero = target == 0
    return {
        "overall_accuracy": jnp.mean(correct),
        "unknown_accuracy": jnp.sum(unknown & correct) / jnp.maximum(jnp.sum(unknown), 1),
        "observed_accuracy": jnp.sum(visible & correct) / jnp.maximum(jnp.sum(visible), 1),
        "unknown_zero_accuracy": jnp.sum(unknown & zero & correct)
        / jnp.maximum(jnp.sum(unknown & zero), 1),
    }


def evaluate(
    params: Any,
    data: StructuredCarrier,
    keys: jax.Array,
    config: M4Config,
    train_config: TrainConfig,
) -> dict[str, jax.Array]:
    predictions = predict_batch(
        params,
        data,
        keys,
        config,
        train_config.alpha,
        train_config.sigma,
        train_config.rho_b,
        train_config.rho_length,
    )
    metrics: dict[str, jax.Array] = {
        "h_mse": jnp.mean((predictions.h - data.h) ** 2),
        "h_corrupt_mse": jnp.mean((predictions.h_corrupt - data.h) ** 2),
    }
    if predictions.boundary_logits is not None:
        pred = jnp.argmax(predictions.boundary_logits, axis=-1)
        ce = -jnp.take_along_axis(
            jax.nn.log_softmax(predictions.boundary_logits), data.b[..., None], axis=-1
        )[..., 0]
        metrics["boundary_cross_entropy"] = jnp.mean(ce)
        metrics.update(
            {
                f"boundary_{k}": v
                for k, v in _boundary_metrics(pred, data.b, predictions.observed_b).items()
            }
        )
    if predictions.length_logits is not None:
        pred = jnp.argmax(predictions.length_logits, axis=-1)
        ce = -jnp.take_along_axis(
            jax.nn.log_softmax(predictions.length_logits), data.length[..., None], axis=-1
        )[..., 0]
        metrics["length_cross_entropy"] = jnp.mean(ce)
        metrics.update(
            {
                f"length_{k}": v
                for k, v in _length_metrics(
                    pred, data.length, predictions.observed_length, config.structure
                ).items()
            }
        )
    return metrics


def run(
    config: M4Config,
    train_config: TrainConfig,
    train_data: StructuredCarrier,
    val_data: StructuredCarrier,
    initialise_params,
):
    params: Any = initialise_params(config, jax.random.key(train_config.seed))
    optimizer = optax.adam(train_config.learning_rate)
    opt_state = optimizer.init(params)

    def loss_fn(p, batch, keys):
        return batch_loss(
            p,
            batch,
            keys,
            config,
            train_config.alpha,
            train_config.sigma,
            train_config.rho_b,
            train_config.rho_length,
        )

    step_fn = jax.jit(jax.value_and_grad(loss_fn))
    losses = []
    grad_norms = []
    start = time.perf_counter()
    for step in range(train_config.steps):
        indices = jax.random.randint(
            jax.random.key(train_config.seed + step),
            (train_config.batch_size,),
            0,
            train_data.h.shape[0],
        )
        batch = jax.tree_util.tree_map(lambda value, indices=indices: value[indices], train_data)
        value, grads = step_fn(
            params, batch, _keys(train_config.seed, step, train_config.batch_size)
        )
        if not bool(jnp.isfinite(value)) or not _tree_finite(grads):
            raise FloatingPointError("non-finite M4 loss or gradient")
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        losses.append(value)
        grad_norms.append(optax.global_norm(grads))
    elapsed = time.perf_counter() - start
    metrics: dict[str, jax.Array | float] = dict(
        evaluate(
            params,
            val_data,
            _keys(train_config.seed + 50000, 0, val_data.h.shape[0]),
            config,
            train_config,
        )
    )
    metrics["train_loss_initial"] = losses[0]
    metrics["train_loss_final"] = losses[-1]
    metrics["grad_norm_final"] = grad_norms[-1]
    metrics["step_seconds"] = elapsed / train_config.steps
    metrics["compile_seconds_included"] = elapsed
    metrics["parameter_norm"] = jnp.sqrt(
        sum(jnp.sum(cast(jax.Array, leaf) ** 2) for leaf in jax.tree_util.tree_leaves(params))
    )
    metrics["parameter_count"] = sum(
        int(cast(jax.Array, leaf).size) for leaf in jax.tree_util.tree_leaves(params)
    )
    return params, metrics
