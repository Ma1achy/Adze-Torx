"""Small known-solvable affine toy control using T=exp(G)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import optax

from adze_t.model.core import RecurrentCoreConfig, deterministic_core, initialise_params


def run_toy(family: str, q: int, steps: int = 60, learning_rate: float = 0.03) -> float:
    width = 3
    generator = jnp.array([[-0.20, 0.08, 0.0], [0.0, -0.15, 0.05], [0.02, 0.0, -0.10]])
    target = jax.scipy.linalg.expm(generator)
    states = jnp.arange(48.0, dtype=jnp.float32).reshape((16, 3)) / 10.0 - 2.0
    targets = states @ target.T
    config = RecurrentCoreConfig(
        width=width,
        q=q,
        family=family,
        eta=0.25,
        total_variance=0.01,
        noise_mode="fixed_total",
    )
    params = initialise_params(config, jax.random.key(700 + q))
    optimizer = optax.adam(learning_rate)
    state = optimizer.init(params)

    def loss(p):
        predicted = jax.vmap(lambda x: deterministic_core(config, p, x))(states)
        return jnp.mean((predicted - targets) ** 2)

    for _ in range(steps):
        value, grads = jax.value_and_grad(loss)(params)
        del value
        updates, state = optimizer.update(grads, state, params)
        params = optax.apply_updates(params, updates)
    return float(loss(params))
