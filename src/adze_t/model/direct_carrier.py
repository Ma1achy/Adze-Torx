"""M2 fixed-capacity direct-carrier model components."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import jax
import jax.numpy as jnp
from torx import ChainFactor
from torx.psc import AffineGaussianGate


@dataclass(frozen=True)
class DirectCarrierConfig:
    """Static shape and recurrent-compute configuration for M2."""

    capacity: int = 4
    latent_dim: int = 3
    q: int = 1
    tied: bool = True
    fixed_total_noise: bool = True

    def __post_init__(self) -> None:
        if min(self.capacity, self.latent_dim, self.q) < 1:
            raise ValueError("capacity, latent_dim, and q must be positive")

    @property
    def width(self) -> int:
        return self.capacity * self.latent_dim


def make_gate(config: DirectCarrierConfig) -> AffineGaussianGate:
    """Create the public Torx affine-Gaussian transition gate."""
    return AffineGaussianGate(sites=[0], dims=(config.width,))


def initialise_params(config: DirectCarrierConfig, key: jax.Array, noise_var: float = 0.01):
    """Initialise stable affine transitions with explicit JAX randomness."""
    gate = make_gate(config)
    base = gate.init_params(key)
    identity = 0.6 * jnp.eye(config.width, dtype=base["A"].dtype)
    base = {
        "A": identity,
        "b": jnp.zeros_like(base["b"]),
        "log_var": jnp.full_like(base["log_var"], jnp.log(noise_var)),
    }
    if config.tied:
        return base
    return [dict(base) for _ in range(config.q)]


def _step(gate, params, state, key, discrete_dtype):
    return cast(
        jax.Array,
        gate.sample(
            key,
            {"continuous": state, "discrete": jnp.empty((0,), dtype=discrete_dtype)},
            params,
        ),
    )


def apply_core(config: DirectCarrierConfig, params, state: jax.Array, key: jax.Array) -> jax.Array:
    """Apply Q public Torx transitions with explicit independent subkeys."""
    gate = make_gate(config)
    keys = jnp.expand_dims(key, 0) if config.q == 1 else jax.random.split(key, config.q)
    states = state
    for index, step_key in enumerate(keys):
        step_params = params if config.tied else params[index]
        states = _step(gate, step_params, states, step_key, gate.input_ports["discrete"].dtype)
    return states


def deterministic_core(config: DirectCarrierConfig, params, state: jax.Array) -> jax.Array:
    """Apply the same affine mean dynamics without stochastic sampling."""
    states = state
    for index in range(config.q):
        step_params = params if config.tied else params[index]
        states = step_params["A"] @ states + step_params["b"]
    return states


def apply_chain(config: DirectCarrierConfig, params, state: jax.Array, key: jax.Array) -> jax.Array:
    """Apply the public ChainFactor for a tied manual-unroll comparison."""
    if not config.tied:
        raise ValueError("public ChainFactor comparison requires tied parameters")
    gate = make_gate(config)
    chain = ChainFactor(gate, config.q, "continuous", weight_tied=True)
    inputs = {
        "continuous": state,
        "discrete": jnp.empty((0,), dtype=gate.input_ports["discrete"].dtype),
    }
    return cast(jax.Array, chain.sample(key, inputs, params))


def effective_log_var(config: DirectCarrierConfig, params):
    """Return the per-step variance parameter under the fixed-total-noise policy."""
    divisor = config.q if config.fixed_total_noise else 1
    if config.tied:
        return params["log_var"] - jnp.log(divisor)
    return jnp.stack([p["log_var"] - jnp.log(divisor) for p in params])
