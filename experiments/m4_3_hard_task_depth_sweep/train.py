"""Common-budget M4.3 training and validation."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import optax

from experiments.m4_2_nonlinear_looped_core.model import (
    NonlinearCoreConfig,
    initialise_params,
)
from experiments.m4_3_hard_task_depth_sweep.model import apply_core_trace
from experiments.m4_3_hard_task_depth_sweep.tasks import HardBatch, make_data


@dataclass(frozen=True)
class TrainConfig:
    steps: int = 120
    batch_size: int = 64
    learning_rate: float = 0.03
    seed: int = 0
    eval_interval: int = 10


def _norm(tree: Any) -> jax.Array:
    return jnp.sqrt(sum(jnp.sum(jnp.square(x)) for x in jax.tree_util.tree_leaves(tree)))


def _finite(tree: Any) -> bool:
    return all(bool(jnp.all(jnp.isfinite(x))) for x in jax.tree_util.tree_leaves(tree))


def _step_key(seed: int, step: int) -> jax.Array:
    return jax.random.PRNGKey(seed + 1009 * step)


def _loss(params, batch: HardBatch, keys, config: NonlinearCoreConfig):
    predictions = jax.vmap(lambda state, key: apply_core_trace(config, params, state, key)[0])(
        batch.initial, keys
    )
    return jnp.mean((predictions[:, :8] - batch.target[:, :8]) ** 2)


def _metrics(params, batch: HardBatch, keys, config: NonlinearCoreConfig):
    predictions = jax.vmap(lambda state, key: apply_core_trace(config, params, state, key)[0])(
        batch.initial, keys
    )
    per_example = jnp.mean((predictions[:, :8] - batch.target[:, :8]) ** 2, axis=-1)
    return {"loss": jnp.mean(per_example), "success_rate": jnp.mean(per_example < 0.01)}


def run(core: NonlinearCoreConfig, config: TrainConfig, depth: int, validation_size: int = 512):
    params = initialise_params(core, jax.random.PRNGKey(config.seed))
    val_batch = make_data(
        jax.random.PRNGKey(900000 + depth), validation_size, depth, False, core.width
    )
    optimiser = optax.adam(config.learning_rate)
    opt_state = optimiser.init(params)
    curves = []
    validation = []
    finite_failures = 0
    start = time.perf_counter()

    def step(current, state, key):
        batch_key, sample_key = jax.random.split(key)
        batch = make_data(batch_key, config.batch_size, depth, True, core.width)
        keys = jax.random.split(sample_key, config.batch_size)
        value, gradients = jax.value_and_grad(_loss)(current, batch, keys, core)
        updates, next_state = optimiser.update(gradients, state, current)
        return optax.apply_updates(current, updates), next_state, value, gradients, updates

    compiled_step = jax.jit(step)
    for step_index in range(1, config.steps + 1):
        step_start = time.perf_counter()
        params, opt_state, loss, gradients, updates = compiled_step(
            params, opt_state, _step_key(config.seed, step_index)
        )
        loss.block_until_ready()
        elapsed = time.perf_counter() - step_start
        if not _finite(params) or not _finite(gradients) or not bool(jnp.isfinite(loss)):
            finite_failures += 1
        curves.append(
            {
                "step": step_index,
                "loss": float(loss),
                "grad_norm": float(_norm(gradients)),
                "update_norm": float(_norm(updates)),
                "step_seconds": elapsed,
            }
        )
        if step_index == 1 or step_index % config.eval_interval == 0 or step_index == config.steps:
            keys = jax.random.split(_step_key(config.seed, 700000), validation_size)
            validation.append(
                {
                    "step": step_index,
                    **{k: float(v) for k, v in _metrics(params, val_batch, keys, core).items()},
                }
            )

    elapsed_total = time.perf_counter() - start
    best = min(validation, key=lambda row: row["loss"])
    return {
        "params": params,
        "validation_batch": val_batch,
        "curves": curves,
        "validation": validation,
        "best": best,
        "final": validation[-1],
        "finite_failures": finite_failures,
        "parameter_count": sum(leaf.size for leaf in jax.tree_util.tree_leaves(params)),
        "compile_seconds": curves[0]["step_seconds"],
        "steady_step_seconds": float(
            jnp.median(jnp.asarray([x["step_seconds"] for x in curves[2:]]))
        ),
        "total_seconds": elapsed_total,
        "depth": depth,
        "core": core,
    }
