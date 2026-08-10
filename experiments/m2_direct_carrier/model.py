"""M2 reconstruction pipeline and controls."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from adze_t.model.direct_carrier import (
    apply_core,
    deterministic_core,
    effective_log_var,
)
from experiments.m2_direct_carrier.data import corrupt


def reconstruction_loss(pred: jax.Array, target: jax.Array) -> jax.Array:
    return jnp.mean((pred - target) ** 2)


def stochastic_prediction(config, params, clean, key, alpha, sigma):
    noise_key, core_key = jax.random.split(key)
    h0 = corrupt(clean, noise_key, alpha, sigma)
    return apply_core(config, params, h0, core_key), h0


def deterministic_prediction(config, params, clean, key, alpha, sigma):
    h0 = corrupt(clean, key, alpha, sigma)
    return deterministic_core(config, params, h0), h0


def no_update_prediction(clean, key, alpha, sigma):
    return corrupt(clean, key, alpha, sigma)


def batch_loss(config, params, clean, keys, alpha, sigma, stochastic=True):
    def one(target, key):
        if stochastic:
            pred, _ = stochastic_prediction(config, params, target, key, alpha, sigma)
        else:
            pred, _ = deterministic_prediction(config, params, target, key, alpha, sigma)
        return reconstruction_loss(pred, target)

    return jnp.mean(jax.vmap(one)(clean, keys))


def diagnostics(config, params, clean, keys, alpha, sigma):
    predictions, initial = jax.vmap(
        lambda target, key: stochastic_prediction(config, params, target, key, alpha, sigma)
    )(clean, keys)
    return {
        "loss": reconstruction_loss(predictions, clean),
        "no_update_loss": reconstruction_loss(initial, clean),
        "output_variance": jnp.mean(jnp.var(predictions, axis=0)),
        "effective_log_var": jnp.mean(effective_log_var(config, params)),
    }
