"""M4 model: M3 carrier heads over a configurable Torx-native recurrent core."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, NamedTuple

import jax
import jax.nn as jnn
import jax.numpy as jnp

from adze_t.model.carrier import StructureConfig
from adze_t.model.core import (
    RecurrentCoreConfig,
    apply_core,
)
from adze_t.model.core import (
    initialise_params as initialise_core_params,
)
from adze_t.model.corruption import corrupt_content, corrupt_structure
from experiments.m3_carrier_structure.data import StructuredCarrier


@dataclass(frozen=True)
class M4Config:
    structure: StructureConfig = field(default_factory=StructureConfig)
    core: RecurrentCoreConfig = field(default_factory=RecurrentCoreConfig)
    predict_boundary: bool = True
    predict_length: bool = True

    def __post_init__(self) -> None:
        expected_width = self.structure.capacity * self.structure.latent_dim
        if self.core.width != expected_width:
            raise ValueError("core width must equal fixed carrier width")

    @property
    def feature_dim(self) -> int:
        return 3 * self.structure.latent_dim + 1 + 3 + self.structure.length_observed_classes


class Prediction(NamedTuple):
    h: jax.Array
    boundary_logits: jax.Array | None
    length_logits: jax.Array | None
    observed_b: jax.Array
    observed_length: jax.Array
    h_corrupt: jax.Array
    trajectory: jax.Array | None


def _head(features: jax.Array, params: dict[str, jax.Array], name: str) -> jax.Array:
    return features @ params[f"{name}_weight"] + params[f"{name}_bias"]


def initialise_params(config: M4Config, key: jax.Array) -> Any:
    core_key, _ = jax.random.split(key)
    params: dict[str, object] = {"core": initialise_core_params(config.core, core_key)}
    if config.predict_boundary:
        params["boundary"] = {
            "boundary_weight": jnp.zeros((config.feature_dim, 2)),
            "boundary_bias": jnp.zeros((2,)),
        }
    if config.predict_length:
        params["length"] = {
            "length_weight": jnp.zeros((config.feature_dim, config.structure.max_length + 1)),
            "length_bias": jnp.zeros((config.structure.max_length + 1,)),
        }
    return params


def local_features(
    h: jax.Array,
    observed_b: jax.Array,
    observed_length: jax.Array,
    config: M4Config,
) -> jax.Array:
    previous = jnp.concatenate((h[:1], h[:-1]), axis=0)
    difference = h - previous
    # The first site's difference is exactly zero; the epsilon keeps its
    # deterministic readout gradient finite.
    jump_norm = jnp.sqrt(jnp.sum(difference**2, axis=-1, keepdims=True) + 1e-12)
    return jnp.concatenate(
        (
            h,
            previous,
            difference,
            jump_norm,
            jnn.one_hot(observed_b, 3),
            jnn.one_hot(observed_length, config.structure.length_observed_classes),
        ),
        axis=-1,
    )


def predict_one(
    params: Any,
    target: StructuredCarrier,
    key: jax.Array,
    config: M4Config,
    alpha: float,
    sigma: float,
    rho_b: float,
    rho_length: float,
    return_trajectory: bool = False,
) -> Prediction:
    h_key, structure_key, core_key = jax.random.split(key, 3)
    h_corrupt = corrupt_content(target.h, h_key, alpha, sigma)
    observed_b, observed_length = corrupt_structure(
        target.b, target.length, structure_key, rho_b, rho_length, config.structure
    )
    core_output = apply_core(
        config.core,
        params["core"],
        h_corrupt.reshape(-1),
        core_key,
        return_trajectory=return_trajectory,
    )
    if return_trajectory:
        trajectory = core_output.reshape((config.core.q + 1, *target.h.shape))
        h_pred = trajectory[-1]
    else:
        trajectory = None
        h_pred = core_output.reshape(target.h.shape)
    features = local_features(h_pred, observed_b, observed_length, config)
    boundary_logits = (
        _head(features, params["boundary"], "boundary") if config.predict_boundary else None
    )
    length_logits = _head(features, params["length"], "length") if config.predict_length else None
    return Prediction(
        h_pred,
        boundary_logits,
        length_logits,
        observed_b,
        observed_length,
        h_corrupt,
        trajectory,
    )


def cross_entropy(logits: jax.Array, labels: jax.Array) -> jax.Array:
    return -jnp.take_along_axis(jnn.log_softmax(logits), labels[..., None], axis=-1)[..., 0]


def loss_one(
    params: Any,
    target: StructuredCarrier,
    key: jax.Array,
    config: M4Config,
    alpha: float,
    sigma: float,
    rho_b: float,
    rho_length: float,
) -> jax.Array:
    prediction = predict_one(params, target, key, config, alpha, sigma, rho_b, rho_length)
    loss = jnp.mean((prediction.h - target.h) ** 2)
    if prediction.boundary_logits is not None:
        loss = loss + jnp.mean(cross_entropy(prediction.boundary_logits, target.b))
    if prediction.length_logits is not None:
        loss = loss + jnp.mean(cross_entropy(prediction.length_logits, target.length))
    return loss


def batch_loss(
    params: Any,
    batch: StructuredCarrier,
    keys: jax.Array,
    config: M4Config,
    alpha: float,
    sigma: float,
    rho_b: float,
    rho_length: float,
) -> jax.Array:
    return jnp.mean(
        jax.vmap(
            lambda target, key: loss_one(
                params, target, key, config, alpha, sigma, rho_b, rho_length
            )
        )(batch, keys)
    )


def predict_batch(
    params: Any,
    batch: StructuredCarrier,
    keys: jax.Array,
    config: M4Config,
    alpha: float,
    sigma: float,
    rho_b: float,
    rho_length: float,
    return_trajectory: bool = False,
) -> Prediction:
    return jax.vmap(
        lambda target, key: predict_one(
            params,
            target,
            key,
            config,
            alpha,
            sigma,
            rho_b,
            rho_length,
            return_trajectory,
        )
    )(batch, keys)
