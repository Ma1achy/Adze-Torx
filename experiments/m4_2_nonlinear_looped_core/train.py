"""Small reproducible M4.2 training runners."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import optax

from experiments.m3_carrier_structure.data import make_data as make_structure_data
from experiments.m4_2_nonlinear_looped_core.model import (
    M42Config,
    NonlinearCoreConfig,
    initialise_params,
    initialise_reconstruction_params,
    reconstruction_loss,
)
from experiments.m4_2_nonlinear_looped_core.tasks import (
    make_data as make_composition_data,
)
from experiments.m4_2_nonlinear_looped_core.tasks import (
    task_loss,
    task_metrics,
)


@dataclass(frozen=True)
class TrainConfig:
    steps: int = 60
    batch_size: int = 64
    learning_rate: float = 0.03
    seed: int = 0
    eval_interval: int = 10


def _tree_norm(tree: Any) -> jax.Array:
    leaves = jax.tree_util.tree_leaves(tree)
    return jnp.sqrt(sum(jnp.sum(jnp.square(x)) for x in leaves))


def _finite(tree: Any) -> bool:
    return all(bool(jnp.all(jnp.isfinite(x))) for x in jax.tree_util.tree_leaves(tree))


def _key(seed: int, step: int) -> jax.Array:
    return jax.random.PRNGKey(seed + 1009 * step)


def _metrics_to_float(metrics: dict[str, jax.Array]) -> dict[str, float]:
    return {name: float(value) for name, value in metrics.items()}


def _fit(
    params: Any,
    loss_fn: Callable[[Any, Any], jax.Array],
    metric_fn: Callable[[Any, jax.Array], dict[str, jax.Array]],
    batch_fn: Callable[[jax.Array], Any],
    config: TrainConfig,
) -> dict[str, Any]:
    optimiser = optax.adam(config.learning_rate)
    opt_state = optimiser.init(params)
    curves: list[dict[str, float]] = []
    validation: list[dict[str, float]] = []
    finite_failures = 0
    start = time.perf_counter()

    def step(current, state, key):
        batch = batch_fn(key)
        value, grads = jax.value_and_grad(loss_fn)(current, batch)
        updates, next_state = optimiser.update(grads, state, current)
        return optax.apply_updates(current, updates), next_state, value, grads, updates

    compiled_step = jax.jit(step)
    for step_index in range(1, config.steps + 1):
        step_start = time.perf_counter()
        params, opt_state, loss, grads, updates = compiled_step(
            params, opt_state, _key(config.seed, step_index)
        )
        loss.block_until_ready()
        elapsed = time.perf_counter() - step_start
        if not _finite(params) or not _finite(grads) or not bool(jnp.isfinite(loss)):
            finite_failures += 1
        curves.append(
            {
                "step": float(step_index),
                "loss": float(loss),
                "grad_norm": float(_tree_norm(grads)),
                "update_norm": float(_tree_norm(updates)),
                "step_seconds": elapsed,
            }
        )
        if step_index == 1 or step_index % config.eval_interval == 0 or step_index == config.steps:
            validation.append(
                {
                    "step": float(step_index),
                    **_metrics_to_float(metric_fn(params, _key(config.seed, 700000))),
                }
            )

    elapsed_total = time.perf_counter() - start
    best = min(validation, key=lambda row: row["loss"])
    return {
        "params": params,
        "curves": curves,
        "validation": validation,
        "best": best,
        "final": validation[-1],
        "finite_failures": finite_failures,
        "total_seconds": elapsed_total,
        "compile_seconds": curves[0]["step_seconds"],
        "steady_step_seconds": float(
            jnp.median(jnp.asarray([row["step_seconds"] for row in curves[2:]]))
        ),
        "parameter_count": sum(x.size for x in jax.tree_util.tree_leaves(params)),
    }


def run_composition(
    core: NonlinearCoreConfig,
    train_config: TrainConfig,
    depth: int,
    validation_size: int = 512,
) -> dict[str, Any]:
    from experiments.m4_2_nonlinear_looped_core.model import apply_core

    params = initialise_params(core, jax.random.PRNGKey(train_config.seed))
    val_batch = make_composition_data(
        jax.random.PRNGKey(900000 + depth), validation_size, depth, False, core.width
    )

    def batch_fn(key):
        return make_composition_data(key, train_config.batch_size, depth, True, core.width)

    def loss_fn(current, batch):
        keys = jax.random.split(
            jax.random.fold_in(jax.random.PRNGKey(55), batch.initial.shape[0]),
            batch.initial.shape[0],
        )
        return task_loss(current, batch, keys, lambda p, x, k: apply_core(core, p, x, k))

    def metric_fn(current, key):
        keys = jax.random.split(key, val_batch.initial.shape[0])
        return task_metrics(current, val_batch, keys, lambda p, x, k: apply_core(core, p, x, k))

    return _fit(params, loss_fn, metric_fn, batch_fn, train_config) | {
        "depth": depth,
        "core": core,
        "validation_batch": val_batch,
    }


def run_reconstruction(
    config: M42Config,
    train_config: TrainConfig,
    validation_size: int = 256,
) -> dict[str, Any]:
    params = initialise_reconstruction_params(config, jax.random.PRNGKey(train_config.seed))
    val_batch = make_structure_data(jax.random.PRNGKey(910000), validation_size, config.structure)

    def batch_fn(key):
        return make_structure_data(key, train_config.batch_size, config.structure)

    def loss_fn(current, batch):
        keys = jax.random.split(
            jax.random.fold_in(jax.random.PRNGKey(56), batch.h.shape[0]), batch.h.shape[0]
        )
        return reconstruction_loss(current, batch, keys, config)

    def metric_fn(current, key):
        from experiments.m4_2_nonlinear_looped_core.diagnostics import reconstruction_metrics

        keys = jax.random.split(key, val_batch.h.shape[0])
        return reconstruction_metrics(current, val_batch, keys, config)

    return _fit(params, loss_fn, metric_fn, batch_fn, train_config) | {
        "config": config,
        "validation_batch": val_batch,
    }
