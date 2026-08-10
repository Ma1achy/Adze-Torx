"""Small matched-budget M4.5 trainers."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import optax

from experiments.m4_5_algorithmic_reasoning.model import (
    AlgorithmicConfig,
    apply_state,
    initialise_params,
    logits,
)
from experiments.m4_5_algorithmic_reasoning.programs import mask_all_conditioning


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


def fit(
    config: AlgorithmicConfig,
    training: TrainConfig,
    batch_fn: Callable,
    validation_batch: Any,
    loss_fn: Callable,
    metric_fn: Callable,
) -> dict[str, Any]:
    params = initialise_params(config, jax.random.PRNGKey(training.seed))
    optimiser = optax.adam(training.learning_rate)
    opt_state = optimiser.init(params)
    curves: list[dict[str, float]] = []
    validation: list[dict[str, float]] = []
    finite_failures = 0

    def step(current, state, key):
        batch = batch_fn(key)
        sample_key, _data_key = jax.random.split(key)
        value, grads = jax.value_and_grad(loss_fn)(current, batch, sample_key)
        updates, next_state = optimiser.update(grads, state, current)
        return optax.apply_updates(current, updates), next_state, value, grads, updates

    compiled = jax.jit(step)
    start = time.perf_counter()
    for step_index in range(1, training.steps + 1):
        step_start = time.perf_counter()
        params, opt_state, loss, grads, updates = compiled(
            params, opt_state, jax.random.PRNGKey(training.seed + 1009 * step_index)
        )
        loss.block_until_ready()
        if not _finite(params) or not _finite(grads) or not bool(jnp.isfinite(loss)):
            finite_failures += 1
        curves.append(
            {
                "step": step_index,
                "loss": float(loss),
                "grad_norm": float(_norm(grads)),
                "update_norm": float(_norm(updates)),
                "step_seconds": time.perf_counter() - step_start,
            }
        )
        if (
            step_index == 1
            or step_index % training.eval_interval == 0
            or step_index == training.steps
        ):
            metrics = metric_fn(
                params, validation_batch, jax.random.PRNGKey(training.seed + 700000)
            )
            validation.append({"step": step_index, **{k: float(v) for k, v in metrics.items()}})
    return {
        "params": params,
        "curves": curves,
        "validation": validation,
        "best": min(validation, key=lambda row: row["loss"]),
        "final": validation[-1],
        "finite_failures": finite_failures,
        "parameter_count": sum(x.size for x in jax.tree_util.tree_leaves(params)),
        "compile_seconds": curves[0]["step_seconds"],
        "steady_step_seconds": float(
            jnp.median(jnp.asarray([x["step_seconds"] for x in curves[2:]]))
            if len(curves) > 2
            else curves[-1]["step_seconds"]
        ),
        "total_seconds": time.perf_counter() - start,
        "config": config,
    }


def make_loss(config: AlgorithmicConfig, task: str, mode: str = "all"):
    def _run(params, batch, key):
        keys = jax.random.split(key, batch.initial_dynamic.shape[0])
        raw = getattr(batch, f"{mode}_conditioning")
        if mode == "all":
            if task == "program":
                raw = mask_all_conditioning(raw)
            schedules = jnp.broadcast_to(raw[:, None, :], (raw.shape[0], config.q, raw.shape[-1]))
        else:
            if raw.shape[1] < config.q:
                raw = jnp.concatenate(
                    (
                        raw,
                        jnp.broadcast_to(
                            raw[:, -1:, :], (raw.shape[0], config.q - raw.shape[1], raw.shape[-1])
                        ),
                    ),
                    axis=1,
                )
            schedules = raw[:, : config.q]
        output = jax.vmap(
            lambda d, c, k: logits(config, params, apply_state(config, params, d, c, k))
        )(batch.initial_dynamic, schedules, keys)
        if task == "arithmetic":
            target = batch.result
            ce = -jnp.mean(
                jnp.take_along_axis(
                    jax.nn.log_softmax(output, axis=-1), target[..., None], axis=-1
                )[..., 0]
            )
            acc = jnp.mean(jnp.all(jnp.argmax(output, axis=-1) == target, axis=-1))
            digit = jnp.mean(jnp.argmax(output, axis=-1) == target)
        else:
            target = batch.final_registers
            ce = -jnp.mean(
                jnp.take_along_axis(
                    jax.nn.log_softmax(output, axis=-1), target[..., None], axis=-1
                )[..., 0]
            )
            acc = jnp.mean(jnp.all(jnp.argmax(output, axis=-1) == target, axis=-1))
            digit = jnp.mean(jnp.argmax(output, axis=-1) == target)
        return ce, {"loss": ce, "exact_accuracy": acc, "component_accuracy": digit}

    def loss_fn(params, batch, key):
        return _run(params, batch, key)[0]

    def metric_fn(params, batch, key):
        return _run(params, batch, key)[1]

    return loss_fn, metric_fn
