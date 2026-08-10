"""Common-budget M4.4 training runner."""

from __future__ import annotations

import time
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import optax

from experiments.m4_4_faithful_loop_core.model import (
    FaithfulConfig,
    apply_trace,
    initialise_params,
    minimal_params,
    minimal_trace,
)
from experiments.m4_4_faithful_loop_core.tasks import make_data


@dataclass(frozen=True)
class TrainConfig:
    steps: int = 80
    batch_size: int = 32
    learning_rate: float = 0.03
    seed: int = 0
    eval_interval: int = 10


def _norm(tree):
    return jnp.sqrt(sum(jnp.sum(x * x) for x in jax.tree_util.tree_leaves(tree)))


def _finite(tree):
    return all(bool(jnp.all(jnp.isfinite(x))) for x in jax.tree_util.tree_leaves(tree))


def run(
    config: FaithfulConfig,
    training: TrainConfig,
    depth: int,
    validation_size: int = 256,
    family: str = "faithful",
):
    initialise = initialise_params if family == "faithful" else minimal_params
    apply = apply_trace if family == "faithful" else minimal_trace
    params = initialise(config, jax.random.PRNGKey(training.seed))
    validation_batch = make_data(jax.random.PRNGKey(51000 + depth), validation_size, depth, False)
    optimiser = optax.adam(training.learning_rate)
    opt_state = optimiser.init(params)
    curves, validation = [], []
    finite_failures = 0
    start = time.perf_counter()

    def step(current, state, key):
        data_key, sample_key = jax.random.split(key)
        batch = make_data(data_key, training.batch_size, depth, True)
        sample_keys = jax.random.split(sample_key, training.batch_size)

        def loss_fn(candidate):
            output = jax.vmap(lambda x, k: apply(config, candidate, x, k)[0])(
                batch.initial, sample_keys
            )
            return jnp.mean((output[:, :8] - batch.target[:, :8]) ** 2)

        loss, gradients = jax.value_and_grad(loss_fn)(current)
        updates, next_state = optimiser.update(gradients, state, current)
        return optax.apply_updates(current, updates), next_state, loss, gradients, updates

    compiled = jax.jit(step)
    for step_index in range(1, training.steps + 1):
        started = time.perf_counter()
        params, opt_state, loss, gradients, updates = compiled(
            params, opt_state, jax.random.PRNGKey(training.seed + 1009 * step_index)
        )
        loss.block_until_ready()
        if not _finite(params) or not _finite(gradients) or not bool(jnp.isfinite(loss)):
            finite_failures += 1
        curves.append(
            {
                "step": step_index,
                "loss": float(loss),
                "grad_norm": float(_norm(gradients)),
                "update_norm": float(_norm(updates)),
                "step_seconds": time.perf_counter() - started,
            }
        )
        if (
            step_index == 1
            or step_index % training.eval_interval == 0
            or step_index == training.steps
        ):
            keys = jax.random.split(jax.random.PRNGKey(training.seed + 700000), validation_size)
            output = jax.vmap(lambda x, k, current=params: apply(config, current, x, k)[0])(
                validation_batch.initial, keys
            )
            error = jnp.mean((output[:, :8] - validation_batch.target[:, :8]) ** 2)
            validation.append({"step": step_index, "loss": float(error)})
    return {
        "params": params,
        "validation_batch": validation_batch,
        "curves": curves,
        "validation": validation,
        "best": min(validation, key=lambda x: x["loss"]),
        "final": validation[-1],
        "finite_failures": finite_failures,
        "parameter_count": sum(x.size for x in jax.tree_util.tree_leaves(params)),
        "compile_seconds": curves[0]["step_seconds"],
        "steady_step_seconds": (
            float(jnp.median(jnp.asarray([x["step_seconds"] for x in curves[2:]])))
            if len(curves) > 2
            else curves[-1]["step_seconds"]
        ),
        "total_seconds": time.perf_counter() - start,
        "config": config,
    }
