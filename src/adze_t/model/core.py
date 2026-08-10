"""Small public-Torx recurrent cores for M4."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import jax
import jax.numpy as jnp
from torx.psc import AffineGaussianGate


@dataclass(frozen=True)
class RecurrentCoreConfig:
    """Static configuration for a manually unrolled public Torx core."""

    width: int = 18
    q: int = 1
    tied: bool = True
    family: str = "current"
    eta: float = 0.25
    total_variance: float = 0.01
    noise_mode: str = "fixed_total"
    cycle_conditioning: bool = False

    def __post_init__(self) -> None:
        if min(self.width, self.q) < 1:
            raise ValueError("width and q must be positive")
        if self.family not in {
            "current",
            "residual",
            "identity_residual",
            "q_normalized_residual",
        }:
            raise ValueError(
                "family must be current, residual, identity_residual, or q_normalized_residual"
            )
        if not 0.0 <= self.eta <= 1.0:
            raise ValueError("eta must be in [0, 1]")
        if self.total_variance <= 0.0:
            raise ValueError("total_variance must be positive")
        if self.noise_mode not in {"fixed_total", "fixed_per_cycle"}:
            raise ValueError("noise_mode must be fixed_total or fixed_per_cycle")


def make_gate(config: RecurrentCoreConfig) -> AffineGaussianGate:
    """Construct the public Torx affine-Gaussian gate used by the core."""
    return AffineGaussianGate(sites=[0], dims=(config.width,))


def per_step_variance(config: RecurrentCoreConfig) -> float:
    """Return the nominal independent Gaussian variance injected per cycle."""
    if config.noise_mode == "fixed_total":
        return config.total_variance / config.q
    return config.total_variance


def nominal_accumulated_variance(config: RecurrentCoreConfig) -> float:
    """Return the sum of independent per-cycle variances before affine transport."""
    return config.q * per_step_variance(config)


def _initial_gate_params(config: RecurrentCoreConfig, key: jax.Array) -> dict[str, jax.Array]:
    gate = make_gate(config)
    base = gate.init_params(key)
    if config.family == "current":
        matrix = 0.6 * jnp.eye(config.width, dtype=base["A"].dtype)
        bias = jnp.zeros_like(base["b"])
    elif config.family == "residual":
        # A small negative diagonal delta starts as a near-identity refinement.
        matrix = -0.4 * jnp.eye(config.width, dtype=base["A"].dtype)
        bias = jnp.zeros_like(base["b"])
    else:
        # M4.1 variants start at an exact identity mean map for every Q.
        matrix = jnp.zeros_like(base["A"])
        bias = jnp.zeros_like(base["b"])
    params = {
        "A": matrix,
        "b": bias,
        "log_var": jnp.full_like(base["log_var"], jnp.log(per_step_variance(config))),
    }
    if config.cycle_conditioning:
        params["cycle_bias"] = jnp.zeros_like(base["b"])
    return params


def initialise_params(config: RecurrentCoreConfig, key: jax.Array) -> Any:
    """Initialise tied or occurrence-specific parameters with equal values."""
    base = _initial_gate_params(config, key)
    if config.tied:
        return base
    return [dict(base) for _ in range(config.q)]


def _gate_params(
    config: RecurrentCoreConfig, params: dict[str, jax.Array], cycle: int
) -> dict[str, jax.Array]:
    if config.family == "current":
        matrix = params["A"]
        bias = params["b"]
    elif config.family == "residual":
        matrix = jnp.eye(config.width, dtype=params["A"].dtype) + config.eta * params["A"]
        bias = config.eta * params["b"]
    else:
        scale = config.eta
        if config.family == "q_normalized_residual":
            scale = scale / config.q
        matrix = jnp.eye(config.width, dtype=params["A"].dtype) + scale * params["A"]
        bias = scale * params["b"]
    if config.cycle_conditioning:
        bias = bias + (cycle / max(config.q, 1)) * params["cycle_bias"]
    return {"A": matrix, "b": bias, "log_var": params["log_var"]}


def _sample_step(
    config: RecurrentCoreConfig,
    params: dict[str, jax.Array],
    state: jax.Array,
    key: jax.Array,
    cycle: int,
) -> jax.Array:
    gate = make_gate(config)
    inputs = {
        "continuous": state,
        "discrete": jnp.empty((0,), dtype=gate.input_ports["discrete"].dtype),
    }
    return cast(jax.Array, gate.sample(key, inputs, _gate_params(config, params, cycle)))


def apply_core(
    config: RecurrentCoreConfig,
    params: Any,
    state: jax.Array,
    key: jax.Array,
    return_trajectory: bool = False,
) -> jax.Array:
    """Apply Q public Torx transitions with explicit independent subkeys."""
    keys = jax.random.split(key, config.q)
    states = state
    trajectory = [state]
    for cycle, step_key in enumerate(keys):
        step_params = params if config.tied else params[cycle]
        states = _sample_step(config, step_params, states, step_key, cycle)
        trajectory.append(states)
    if return_trajectory:
        return jnp.stack(trajectory)
    return states


def deterministic_core(
    config: RecurrentCoreConfig, params: Any, state: jax.Array, return_trajectory: bool = False
) -> jax.Array:
    """Apply the analytically corresponding affine mean recurrence."""
    states = state
    trajectory = [state]
    for cycle in range(config.q):
        step_params = params if config.tied else params[cycle]
        effective = _gate_params(config, step_params, cycle)
        states = effective["A"] @ states + effective["b"]
        trajectory.append(states)
    if return_trajectory:
        return jnp.stack(trajectory)
    return states


def prefix_config(config: RecurrentCoreConfig, q: int) -> RecurrentCoreConfig:
    """Create a same-family config for a trajectory prefix."""
    return RecurrentCoreConfig(
        width=config.width,
        q=q,
        tied=config.tied,
        family=config.family,
        eta=config.eta,
        total_variance=config.total_variance,
        noise_mode=config.noise_mode,
        cycle_conditioning=config.cycle_conditioning,
    )


def effective_linear_map(config: RecurrentCoreConfig, params: Any) -> jax.Array:
    """Return the deterministic Q-step linear map for affine diagnostics."""
    effective = jnp.eye(config.width)
    for cycle in range(config.q):
        step_params = params if config.tied else params[cycle]
        matrix = _gate_params(config, step_params, cycle)["A"]
        effective = matrix @ effective
    return effective
