"""Small reproducible M2 training runner."""

from __future__ import annotations

import time
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import optax

from adze_t.model.direct_carrier import DirectCarrierConfig, initialise_params
from experiments.m2_direct_carrier.data import corrupt
from experiments.m2_direct_carrier.model import batch_loss, diagnostics


@dataclass(frozen=True)
class TrainConfig:
    steps: int = 80
    batch_size: int = 64
    learning_rate: float = 0.03
    alpha: float = 0.6
    sigma: float = 0.5
    seed: int = 0
    stochastic: bool = True


def _keys(seed: int, step: int, n: int) -> jax.Array:
    return jax.random.split(jax.random.key(seed + 1009 * step), n)


def run(
    config: DirectCarrierConfig,
    train_config: TrainConfig,
    train_data: jax.Array,
    val_data: jax.Array,
):
    params = initialise_params(config, jax.random.key(train_config.seed))
    optimizer = optax.adam(train_config.learning_rate)
    opt_state = optimizer.init(params)

    def loss_fn(p, targets, keys):
        return batch_loss(
            config,
            p,
            targets,
            keys,
            train_config.alpha,
            train_config.sigma,
            stochastic=train_config.stochastic,
        )

    step_fn = jax.jit(jax.value_and_grad(loss_fn))
    start = time.perf_counter()
    losses = []
    grad_norms = []
    for step in range(train_config.steps):
        indices = jax.random.randint(
            jax.random.key(train_config.seed + step),
            (train_config.batch_size,),
            0,
            train_data.shape[0],
        )
        batch = train_data[indices]
        value, grads = step_fn(
            params, batch, _keys(train_config.seed, step, train_config.batch_size)
        )
        if not bool(jnp.isfinite(value)) or not bool(
            jnp.all(
                jnp.asarray(
                    [jnp.all(jnp.isfinite(leaf)) for leaf in jax.tree_util.tree_leaves(grads)]
                )
            )
        ):
            raise FloatingPointError("non-finite M2 loss or gradient")
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        losses.append(value)
        grad_norms.append(optax.global_norm(grads))
    elapsed = time.perf_counter() - start
    val_keys = _keys(train_config.seed + 50000, 0, val_data.shape[0])
    metrics: dict[str, jax.Array | float] = dict(
        diagnostics(config, params, val_data, val_keys, train_config.alpha, train_config.sigma)
    )
    metrics["train_loss_final"] = losses[-1]
    metrics["train_loss_initial"] = losses[0]
    metrics["grad_norm_final"] = grad_norms[-1]
    metrics["step_seconds"] = elapsed / train_config.steps
    metrics["compile_seconds_included"] = elapsed
    metrics["parameter_norm"] = jnp.sqrt(
        sum(jnp.sum(leaf**2) for leaf in jax.tree_util.tree_leaves(params))
    )
    return params, metrics


def control_loss(clean: jax.Array, seed: int, alpha: float, sigma: float) -> float:
    key = jax.random.key(seed)
    return float(jnp.mean((corrupt(clean, key, alpha, sigma) - clean) ** 2))
