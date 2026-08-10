"""Public-Torx mixed trajectory experiment and M1.5 score bridge."""

from __future__ import annotations

from typing import NamedTuple, cast

import jax
import jax.numpy as jnp
from torx.psc import PNOT, HybridPCircuit, MixtureGaussianGate

from adze_t.train.score_bridge import score_corrected_loss

from .oracles import mixed_value_and_grad


class Trajectory(NamedTuple):
    loss: jax.Array
    log_prob_sum: jax.Array
    discrete_final: jax.Array
    continuous_final: jax.Array


def build(depth: int = 1) -> HybridPCircuit:
    return HybridPCircuit(
        [
            PNOT(0),
            MixtureGaussianGate(sites=([0], [0]), dims=(1,), num_components=2),
        ],
        reps=depth,
    )


def forward_sample(depth: int, params, key):
    circuit = build(depth)
    inputs = {
        "discrete": jnp.array([0], dtype=circuit.gates[0].input_ports["in"].dtype),
        "continuous": jnp.array([0.0], dtype=circuit.input_ports["continuous"].dtype),
    }
    return circuit.sample(key, inputs, params)


def oracle(depth: int, params):
    return mixed_value_and_grad(params, depth)


def _gates() -> tuple[PNOT, MixtureGaussianGate]:
    return PNOT(0), MixtureGaussianGate(sites=([0], [0]), dims=(1,), num_components=2)


def sample_trajectory(params: jax.Array, key, depth: int) -> Trajectory:
    """Sample one mixed trajectory using public Torx factor methods."""
    theta, alpha, beta, log_var = params
    pnot, mixture = _gates()
    theta_param = jnp.reshape(theta, (1,))
    mixture_params = {
        "means": jnp.stack((jnp.zeros_like(beta), beta)).reshape(2, 1),
        "log_vars": jnp.stack((log_var, log_var)).reshape(2, 1),
    }
    x = jnp.zeros((1,), dtype=pnot.input_ports["in"].dtype)
    h = jnp.zeros((1,), dtype=mixture.input_ports["continuous"].dtype)
    log_prob_sum = jnp.asarray(0.0, dtype=params.dtype)
    step_keys = jax.random.split(key, 2 * depth)

    for step in range(depth):
        step_key_disc = step_keys[2 * step]
        step_key_cont = step_keys[2 * step + 1]
        previous_x = x
        x = cast(jax.Array, pnot.sample(step_key_disc, {"in": previous_x}, theta_param))
        log_prob = cast(jax.Array, pnot.log_probability({"in": previous_x}, x, theta_param))
        log_prob_sum = log_prob_sum + log_prob
        h = cast(
            jax.Array,
            mixture.sample(
                step_key_cont,
                {
                    "discrete": x.astype(mixture.input_ports["discrete"].dtype),
                    "continuous": alpha * h,
                },
                mixture_params,
            ),
        )

    loss = 0.4 * x[0].astype(h.dtype) + 0.3 * h[0] + 0.8 * h[0] ** 2
    return Trajectory(loss, log_prob_sum, x, h)


def uncorrected_batch_loss(params: jax.Array, keys: jax.Array, depth: int) -> jax.Array:
    losses = jax.vmap(lambda key: sample_trajectory(params, key, depth).loss)(keys)
    return jnp.mean(losses)


def corrected_batch_loss(params: jax.Array, keys: jax.Array, depth: int) -> jax.Array:
    trajectories = jax.vmap(lambda key: sample_trajectory(params, key, depth))(keys)
    corrected = score_corrected_loss(trajectories.loss, trajectories.log_prob_sum)
    return jnp.mean(corrected)


def estimate_bridge(params: jax.Array, keys: jax.Array, depth: int):
    return jax.value_and_grad(corrected_batch_loss)(params, keys, depth)


def estimate_uncorrected(params: jax.Array, keys: jax.Array, depth: int):
    return jax.value_and_grad(uncorrected_batch_loss)(params, keys, depth)


def make_keys(seed: int, n_samples: int) -> jax.Array:
    return jax.random.split(jax.random.key(seed), n_samples)


def public_gradient_route():
    return (
        "No unified public mixed stochastic-gradient simulator found: "
        "BranchingSimulator accepts DiscretePCircuit, while "
        "AffineGaussianSimulator accepts only affine-Gaussian HybridPCircuit gates."
    )
