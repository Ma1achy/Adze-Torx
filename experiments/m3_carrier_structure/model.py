"""M3 fixed-topology carrier model with lightweight structural readouts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, NamedTuple, cast

import jax
import jax.nn as jnn
import jax.numpy as jnp

from adze_t.model.carrier import BOUNDARY_UNKNOWN, StructureConfig
from adze_t.model.corruption import corrupt_content, corrupt_structure
from adze_t.model.direct_carrier import (
    DirectCarrierConfig,
    apply_core,
)
from adze_t.model.direct_carrier import (
    initialise_params as initialise_core_params,
)
from experiments.m3_carrier_structure.data import StructuredCarrier


@dataclass(frozen=True)
class M3Config:
    structure: StructureConfig = field(default_factory=StructureConfig)
    q: int = 1
    predict_boundary: bool = True
    predict_length: bool = True

    def __post_init__(self) -> None:
        if self.q not in (1, 2):
            raise ValueError("M3 supports Q=1 primary and Q=2 regression only")

    @property
    def core(self) -> DirectCarrierConfig:
        return DirectCarrierConfig(
            capacity=self.structure.capacity,
            latent_dim=self.structure.latent_dim,
            q=self.q,
            tied=True,
        )

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


def _head(features: jax.Array, params: dict[str, jax.Array], name: str) -> jax.Array:
    return features @ params[f"{name}_weight"] + params[f"{name}_bias"]


def initialise_params(config: M3Config, key: jax.Array) -> Any:
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
    config: M3Config,
) -> jax.Array:
    previous = jnp.concatenate((h[:1], h[:-1]), axis=0)
    boundary_one_hot = jnn.one_hot(observed_b, 3)
    length_one_hot = jnn.one_hot(observed_length, config.structure.length_observed_classes)
    difference = h - previous
    jump_norm = jnp.linalg.norm(difference, axis=-1, keepdims=True)
    return jnp.concatenate(
        (h, previous, difference, jump_norm, boundary_one_hot, length_one_hot), axis=-1
    )


def predict_one(
    params,
    target: StructuredCarrier,
    key: jax.Array,
    config: M3Config,
    alpha: float,
    sigma: float,
    rho_b: float,
    rho_length: float,
) -> Prediction:
    h_key, structure_key, core_key = jax.random.split(key, 3)
    h_corrupt = corrupt_content(target.h, h_key, alpha, sigma)
    observed_b, observed_length = corrupt_structure(
        target.b, target.length, structure_key, rho_b, rho_length, config.structure
    )
    h_flat = cast(
        jax.Array, apply_core(config.core, params["core"], h_corrupt.reshape(-1), core_key)
    )
    h_pred = h_flat.reshape(target.h.shape)
    features = local_features(h_corrupt, observed_b, observed_length, config)
    boundary_logits = (
        _head(features, params["boundary"], "boundary") if config.predict_boundary else None
    )
    length_logits = _head(features, params["length"], "length") if config.predict_length else None
    return Prediction(
        h_pred, boundary_logits, length_logits, observed_b, observed_length, h_corrupt
    )


def _cross_entropy(logits: jax.Array, labels: jax.Array) -> jax.Array:
    return -jnp.take_along_axis(jnn.log_softmax(logits), labels[..., None], axis=-1)[..., 0]


def loss_one(
    params,
    target: StructuredCarrier,
    key: jax.Array,
    config: M3Config,
    alpha: float,
    sigma: float,
    rho_b: float,
    rho_length: float,
) -> jax.Array:
    prediction = predict_one(params, target, key, config, alpha, sigma, rho_b, rho_length)
    loss = jnp.mean((prediction.h - target.h) ** 2)
    if prediction.boundary_logits is not None:
        loss = loss + jnp.mean(_cross_entropy(prediction.boundary_logits, target.b))
    if prediction.length_logits is not None:
        loss = loss + jnp.mean(_cross_entropy(prediction.length_logits, target.length))
    return loss


def batch_loss(
    params,
    batch: StructuredCarrier,
    keys: jax.Array,
    config: M3Config,
    alpha,
    sigma,
    rho_b,
    rho_length,
):
    return jnp.mean(
        jax.vmap(
            lambda target, key: loss_one(
                params, target, key, config, alpha, sigma, rho_b, rho_length
            )
        )(batch, keys)
    )


def predict_batch(params, batch, keys, config, alpha, sigma, rho_b, rho_length):
    return jax.vmap(
        lambda target, key: predict_one(
            params, target, key, config, alpha, sigma, rho_b, rho_length
        )
    )(batch, keys)


def accuracy_metrics(
    predicted: jax.Array, target: jax.Array, observed: jax.Array, forced_first: bool = False
):
    unknown = observed == BOUNDARY_UNKNOWN
    observed_mask = ~unknown
    if forced_first:
        nonforced = jnp.arange(target.shape[-1]) > 0
        unknown = unknown & nonforced
        observed_mask = observed_mask & nonforced
    correct = predicted == target
    return {
        "overall_accuracy": jnp.mean(correct),
        "unknown_accuracy": jnp.sum(jnp.where(unknown, correct, 0))
        / jnp.maximum(jnp.sum(unknown), 1),
        "observed_accuracy": jnp.sum(jnp.where(observed_mask, correct, 0))
        / jnp.maximum(jnp.sum(observed_mask), 1),
    }
