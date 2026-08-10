"""Higher-fidelity recurrent-state proxy using public Torx gates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import jax
import jax.numpy as jnp
from torx.psc import AffineGaussianGate

TASK = 8
SCRATCH = 24
OPS = 48
MASK = 12
WIDTH = TASK + SCRATCH + OPS + MASK


@dataclass(frozen=True)
class FaithfulConfig:
    q: int = 1
    blocks: int = 2
    eta: float = 0.15
    residual_scale: float = 0.10
    total_variance: float = 0.04
    fixed_horizon: bool = True
    nonlinear: bool = True
    scratch_width: int = SCRATCH
    progress: bool = False
    fixed_total_noise: bool = True

    def __post_init__(self):
        if min(self.q, self.blocks, self.scratch_width + TASK) < 1:
            raise ValueError("q, blocks, and dynamic state width must be positive")
        if self.scratch_width > SCRATCH:
            raise ValueError("scratch_width cannot exceed the fixed layout")
        if self.eta <= 0 or self.residual_scale <= 0 or self.total_variance <= 0:
            raise ValueError("eta, residual_scale, and total_variance must be positive")

    @property
    def width(self) -> int:
        return WIDTH

    @property
    def step_scale(self) -> float:
        return self.eta / self.q if self.fixed_horizon else self.eta

    @property
    def variance_per_gate(self) -> float:
        divisor = 2 * self.blocks * self.q if self.fixed_total_noise else 2 * self.blocks
        return self.total_variance / divisor

    @property
    def dynamic_width(self) -> int:
        return TASK + self.scratch_width


def gate(config: FaithfulConfig) -> AffineGaussianGate:
    return AffineGaussianGate(sites=[0], dims=(config.width,))


def initialise_params(
    config: FaithfulConfig, key: jax.Array
) -> list[dict[str, dict[str, jax.Array]]]:
    keys = jax.random.split(key, config.blocks * 2)
    result = []
    for block in range(config.blocks):
        gates = []
        for gate_index in range(2):
            base = gate(config).init_params(keys[2 * block + gate_index])
            gates.append(
                {
                    "A": 0.01 * jnp.eye(config.width, dtype=base["A"].dtype),
                    "b": jnp.zeros_like(base["b"]),
                    "log_var": jnp.full_like(base["log_var"], jnp.log(config.variance_per_gate)),
                }
            )
        result.append({"gate1": gates[0], "gate2": gates[1]})
    return result


def _conditioned(state: jax.Array) -> jax.Array:
    operators = state[TASK + SCRATCH : TASK + SCRATCH + OPS].reshape((12, 4))
    valid = state[TASK + SCRATCH + OPS :]
    masked = (operators * valid[:, None]).reshape((OPS,))
    return state.at[TASK + SCRATCH : TASK + SCRATCH + OPS].set(masked)


def _sample(config, params, state, key):
    current = _conditioned(state)
    inputs = {
        "continuous": current,
        "discrete": jnp.empty((0,), dtype=gate(config).input_ports["discrete"].dtype),
    }
    return cast(jax.Array, gate(config).sample(key, inputs, params))


def _residual(config, value, cycle):
    value = jnp.tanh(value) if config.nonlinear else value
    if config.progress:
        value = value.at[: config.dynamic_width].add((cycle / max(config.q, 1)) * 0.01)
    mask = jnp.concatenate(
        (jnp.ones((config.dynamic_width,)), jnp.zeros((WIDTH - config.dynamic_width,)))
    )
    return config.residual_scale * mask * value


def apply_trace(config: FaithfulConfig, params, state, key):
    keys = jax.random.split(key, config.blocks * config.q * 2)
    current = state
    block_states = [state]
    cycle_states = [state]
    index = 0
    for cycle in range(config.q):
        for block in range(config.blocks):
            first = _sample(config, params[block]["gate1"], current, keys[index])
            second = _sample(config, params[block]["gate2"], first, keys[index + 1])
            current = current + config.step_scale * _residual(config, second, cycle)
            block_states.append(current)
            index += 2
        cycle_states.append(current)
    return current, jnp.stack(block_states), jnp.stack(cycle_states)


def deterministic_trace(config: FaithfulConfig, params, state):
    current = state
    block_states = [state]
    cycle_states = [state]
    for cycle in range(config.q):
        for block in range(config.blocks):
            first = (
                params[block]["gate1"]["A"] @ _conditioned(current) + params[block]["gate1"]["b"]
            )
            second = params[block]["gate2"]["A"] @ _conditioned(first) + params[block]["gate2"]["b"]
            current = current + config.step_scale * _residual(config, second, cycle)
            block_states.append(current)
        cycle_states.append(current)
    return current, jnp.stack(block_states), jnp.stack(cycle_states)


def minimal_params(config: FaithfulConfig, key):
    keys = jax.random.split(key, config.blocks)
    result = []
    variance = config.total_variance / (
        config.blocks * config.q if config.fixed_total_noise else config.blocks
    )
    for item in keys:
        base = gate(config).init_params(item)
        result.append(
            {
                "A": 0.01 * jnp.eye(config.width, dtype=base["A"].dtype),
                "b": jnp.zeros_like(base["b"]),
                "log_var": jnp.full_like(base["log_var"], jnp.log(variance)),
            }
        )
    return result


def minimal_trace(config: FaithfulConfig, params, state, key):
    keys = jax.random.split(key, config.blocks * config.q)
    current = state
    blocks = [state]
    cycles = [state]
    index = 0
    for _cycle in range(config.q):
        for block in range(config.blocks):
            value = _sample(config, params[block], current, keys[index])
            value = jnp.tanh(value) if config.nonlinear else value
            mask = jnp.concatenate(
                (jnp.ones((config.dynamic_width,)), jnp.zeros((WIDTH - config.dynamic_width,)))
            )
            current = current + config.step_scale * config.residual_scale * mask * value
            blocks.append(current)
            index += 1
        cycles.append(current)
    return current, jnp.stack(blocks), jnp.stack(cycles)
