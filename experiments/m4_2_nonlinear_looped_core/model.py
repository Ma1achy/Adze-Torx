"""Public-Torx stochastic affine blocks with explicit JAX nonlinear residuals."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, NamedTuple, cast

import jax
import jax.nn as jnn
import jax.numpy as jnp
from torx.psc import AffineGaussianGate

from adze_t.model.carrier import StructureConfig
from adze_t.model.corruption import corrupt_content, corrupt_structure
from experiments.m3_carrier_structure.data import StructuredCarrier


@dataclass(frozen=True)
class NonlinearCoreConfig:
    width: int = 18
    blocks: int = 2
    q: int = 1
    eta: float = 0.25
    total_variance: float = 0.01
    nonlinear: bool = True
    active_mask: tuple[float, ...] | None = None

    def __post_init__(self) -> None:
        if min(self.width, self.blocks, self.q) < 1:
            raise ValueError("width, blocks, and q must be positive")
        if not 0.0 <= self.eta <= 1.0:
            raise ValueError("eta must be in [0, 1]")
        if self.total_variance <= 0.0:
            raise ValueError("total_variance must be positive")
        if self.active_mask is not None and len(self.active_mask) != self.width:
            raise ValueError("active_mask must have length width")

    @property
    def mask(self) -> jax.Array:
        if self.active_mask is None:
            return jnp.ones((self.width,))
        return jnp.asarray(self.active_mask)

    @property
    def per_block_variance(self) -> float:
        return self.total_variance / (self.blocks * self.q)

    @property
    def per_block_step(self) -> float:
        return self.eta / (self.blocks * self.q)


def make_gate(config: NonlinearCoreConfig) -> AffineGaussianGate:
    return AffineGaussianGate(sites=[0], dims=(config.width,))


def initialise_params(config: NonlinearCoreConfig, key: jax.Array) -> list[dict[str, jax.Array]]:
    keys = jax.random.split(key, config.blocks)
    params = []
    for block_key in keys:
        base = make_gate(config).init_params(block_key)
        params.append(
            {
                "A": jnp.zeros_like(base["A"]),
                "b": jnp.zeros_like(base["b"]),
                "log_var": jnp.full_like(base["log_var"], jnp.log(config.per_block_variance)),
            }
        )
    return params


def _gate_step(
    config: NonlinearCoreConfig,
    params: dict[str, jax.Array],
    state: jax.Array,
    key: jax.Array,
) -> jax.Array:
    gate = make_gate(config)
    inputs = {
        "continuous": state,
        "discrete": jnp.empty((0,), dtype=gate.input_ports["discrete"].dtype),
    }
    return cast(jax.Array, gate.sample(key, inputs, params))


def _proposal(config: NonlinearCoreConfig, value: jax.Array) -> jax.Array:
    return jnp.tanh(value) if config.nonlinear else value


def apply_core(
    config: NonlinearCoreConfig,
    params: list[dict[str, jax.Array]],
    state: jax.Array,
    key: jax.Array,
    return_trajectory: bool = False,
) -> jax.Array:
    keys = jax.random.split(key, config.blocks * config.q)
    states = state
    trajectory = [state]
    index = 0
    for _cycle in range(config.q):
        for block in range(config.blocks):
            proposal = _proposal(config, _gate_step(config, params[block], states, keys[index]))
            states = states + config.per_block_step * config.mask * proposal
            index += 1
        trajectory.append(states)
    return jnp.stack(trajectory) if return_trajectory else states


def deterministic_core(
    config: NonlinearCoreConfig,
    params: list[dict[str, jax.Array]],
    state: jax.Array,
    return_trajectory: bool = False,
) -> jax.Array:
    states = state
    trajectory = [state]
    for _cycle in range(config.q):
        for block in range(config.blocks):
            affine = params[block]["A"] @ states + params[block]["b"]
            states = states + config.per_block_step * config.mask * _proposal(config, affine)
        trajectory.append(states)
    return jnp.stack(trajectory) if return_trajectory else states


class ReconstructionPrediction(NamedTuple):
    h: jax.Array
    boundary_logits: jax.Array
    length_logits: jax.Array
    observed_b: jax.Array
    observed_length: jax.Array
    h_corrupt: jax.Array
    trajectory: jax.Array | None


@dataclass(frozen=True)
class M42Config:
    structure: StructureConfig = field(default_factory=StructureConfig)
    core: NonlinearCoreConfig = field(default_factory=NonlinearCoreConfig)

    def __post_init__(self) -> None:
        if self.core.width != self.structure.capacity * self.structure.latent_dim:
            raise ValueError("core width must match carrier width")

    @property
    def feature_dim(self) -> int:
        return 3 * self.structure.latent_dim + 1 + 3 + self.structure.length_observed_classes


def initialise_reconstruction_params(config: M42Config, key: jax.Array) -> dict[str, Any]:
    core_key, _ = jax.random.split(key)
    return {
        "core": initialise_params(config.core, core_key),
        "boundary": {
            "weight": jnp.zeros((config.feature_dim, 2)),
            "bias": jnp.zeros((2,)),
        },
        "length": {
            "weight": jnp.zeros((config.feature_dim, config.structure.max_length + 1)),
            "bias": jnp.zeros((config.structure.max_length + 1,)),
        },
    }


def _features(h, observed_b, observed_length, config: M42Config):
    previous = jnp.concatenate((h[:1], h[:-1]), axis=0)
    difference = h - previous
    norm = jnp.sqrt(jnp.sum(difference**2, axis=-1, keepdims=True) + 1e-12)
    return jnp.concatenate(
        (
            h,
            previous,
            difference,
            norm,
            jnn.one_hot(observed_b, 3),
            jnn.one_hot(observed_length, config.structure.length_observed_classes),
        ),
        axis=-1,
    )


def predict_reconstruction(
    params,
    target: StructuredCarrier,
    key: jax.Array,
    config: M42Config,
    return_trajectory: bool = False,
) -> ReconstructionPrediction:
    h_key, structure_key, core_key = jax.random.split(key, 3)
    h_corrupt = corrupt_content(target.h, h_key, 0.6, 0.5)
    observed_b, observed_length = corrupt_structure(
        target.b, target.length, structure_key, 0.5, 0.5, config.structure
    )
    output = apply_core(
        config.core, params["core"], h_corrupt.reshape(-1), core_key, return_trajectory
    )
    if return_trajectory:
        trajectory = output.reshape((config.core.q + 1, *target.h.shape))
        h = trajectory[-1]
    else:
        trajectory = None
        h = output.reshape(target.h.shape)
    features = _features(h, observed_b, observed_length, config)
    return ReconstructionPrediction(
        h,
        features @ params["boundary"]["weight"] + params["boundary"]["bias"],
        features @ params["length"]["weight"] + params["length"]["bias"],
        observed_b,
        observed_length,
        h_corrupt,
        trajectory,
    )


def reconstruction_loss(params, batch: StructuredCarrier, keys, config: M42Config):
    def one(target, key):
        prediction = predict_reconstruction(params, target, key, config)
        boundary = -jnp.take_along_axis(
            jnn.log_softmax(prediction.boundary_logits), target.b[..., None], -1
        )[..., 0]
        length = -jnp.take_along_axis(
            jnn.log_softmax(prediction.length_logits), target.length[..., None], -1
        )[..., 0]
        return jnp.mean((prediction.h - target.h) ** 2) + jnp.mean(boundary) + jnp.mean(length)

    return jnp.mean(jax.vmap(one)(batch, keys))
