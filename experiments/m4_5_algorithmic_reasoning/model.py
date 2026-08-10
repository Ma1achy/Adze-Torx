"""Fixed-shape recurrent model using public Torx Gaussian gates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import jax
import jax.numpy as jnp
from torx.psc import AffineGaussianGate


@dataclass(frozen=True)
class AlgorithmicConfig:
    q: int
    conditioning_width: int
    output_shape: tuple[int, int]
    dynamic_width: int = 32
    blocks: int = 2
    eta: float = 0.15
    residual_scale: float = 0.10
    total_variance: float = 0.04
    fixed_horizon: bool = False

    @property
    def width(self) -> int:
        return self.dynamic_width + self.conditioning_width

    @property
    def variance_per_gate(self) -> float:
        return self.total_variance / (2 * self.blocks * self.q)

    @property
    def step_scale(self) -> float:
        return self.eta / self.q if self.fixed_horizon else self.eta

    def __post_init__(self) -> None:
        if min(self.q, self.conditioning_width, self.dynamic_width, self.blocks) < 1:
            raise ValueError("all recurrent dimensions must be positive")
        if self.eta <= 0 or self.residual_scale <= 0 or self.total_variance <= 0:
            raise ValueError("eta, residual_scale, and total_variance must be positive")


def _gate(config: AlgorithmicConfig) -> AffineGaussianGate:
    return AffineGaussianGate(sites=[0], dims=(config.width,))


def initialise_params(config: AlgorithmicConfig, key: jax.Array) -> dict[str, Any]:
    keys = jax.random.split(key, config.blocks * 2)
    blocks = []
    for index in range(config.blocks):
        gates = []
        for gate_index in range(2):
            base = _gate(config).init_params(keys[index * 2 + gate_index])
            gates.append(
                {
                    "A": 0.01 * jnp.eye(config.width, dtype=base["A"].dtype),
                    "b": jnp.zeros_like(base["b"]),
                    "log_var": jnp.full_like(base["log_var"], jnp.log(config.variance_per_gate)),
                }
            )
        blocks.append({"gate1": gates[0], "gate2": gates[1]})
    classes = config.output_shape[0] * config.output_shape[1]
    return {
        "blocks": blocks,
        "head_weight": jnp.zeros((config.dynamic_width, classes)),
        "head_bias": jnp.zeros((classes,)),
    }


def _sample(config, params, full_state, key):
    gate = _gate(config)
    inputs = {
        "continuous": full_state,
        "discrete": jnp.empty((0,), dtype=gate.input_ports["discrete"].dtype),
    }
    return cast(jax.Array, gate.sample(key, inputs, params))


def apply_state(
    config: AlgorithmicConfig,
    params: Any,
    initial_dynamic: jax.Array,
    conditioning_schedule: jax.Array,
    key: jax.Array,
    return_trajectory: bool = False,
) -> jax.Array:
    if conditioning_schedule.shape != (config.q, config.conditioning_width):
        raise ValueError("conditioning schedule must have shape (q, conditioning_width)")
    keys = jax.random.split(key, config.q * config.blocks * 2)
    dynamic = initial_dynamic
    trajectory = [dynamic]
    index = 0
    for cycle in range(config.q):
        conditioning = conditioning_schedule[cycle]
        for block in range(config.blocks):
            full = jnp.concatenate((dynamic, conditioning))
            first = _sample(config, params["blocks"][block]["gate1"], full, keys[index])
            second = _sample(config, params["blocks"][block]["gate2"], first, keys[index + 1])
            dynamic = dynamic + config.step_scale * config.residual_scale * jnp.tanh(
                second[: config.dynamic_width]
            )
            index += 2
        trajectory.append(dynamic)
    return jnp.stack(trajectory) if return_trajectory else dynamic


def logits(config: AlgorithmicConfig, params: Any, dynamic: jax.Array) -> jax.Array:
    flat = dynamic @ params["head_weight"] + params["head_bias"]
    return flat.reshape(config.output_shape)


def deterministic_state(
    config: AlgorithmicConfig,
    params: Any,
    initial_dynamic: jax.Array,
    conditioning_schedule: jax.Array,
    return_trajectory: bool = False,
) -> jax.Array:
    # Deterministic mean path: use the public gate's affine mean, without forcing invalid variance.
    keys = jnp.arange(config.q * config.blocks * 2)
    del keys
    dynamic = initial_dynamic
    trajectory = [dynamic]
    for cycle in range(config.q):
        conditioning = conditioning_schedule[cycle]
        for block in range(config.blocks):
            block_params = params["blocks"][block]
            first = (
                block_params["gate1"]["A"] @ jnp.concatenate((dynamic, conditioning))
                + block_params["gate1"]["b"]
            )
            second = block_params["gate2"]["A"] @ first + block_params["gate2"]["b"]
            dynamic = dynamic + config.step_scale * config.residual_scale * jnp.tanh(
                second[: config.dynamic_width]
            )
        trajectory.append(dynamic)
    return jnp.stack(trajectory) if return_trajectory else dynamic
