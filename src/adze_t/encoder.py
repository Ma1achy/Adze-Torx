"""Shared byte frontend plus deterministic context/target encoders."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp

from .config import ReferenceConfig


def _linear(key: jax.Array, in_dim: int, out_dim: int) -> dict[str, jax.Array]:
    return {
        "weight": jax.random.normal(key, (in_dim, out_dim)) / jnp.sqrt(in_dim),
        "bias": jnp.zeros((out_dim,)),
    }


def _apply(x: jax.Array, p: dict[str, jax.Array]) -> jax.Array:
    return x @ p["weight"] + p["bias"]


def init_encoder_params(key: jax.Array, config: ReferenceConfig) -> dict[str, Any]:
    m = config.model
    keys = iter(jax.random.split(key, 8))
    return {
        "byte_embed": jax.random.normal(next(keys), (m.byte_vocab, m.d_front))
        / jnp.sqrt(m.d_front),
        "frontend_in": _linear(next(keys), m.d_front, m.d_front),
        "frontend_state": _linear(next(keys), m.d_front, m.d_front),
        "frontend_gate": _linear(next(keys), m.d_front, m.d_front),
        "context_proj": _linear(next(keys), m.d_front, m.d_ctx),
        "target_h": _linear(next(keys), m.d_front, config.carrier.h_dim),
        "target_b": _linear(next(keys), m.d_front, 2),
        "target_l": _linear(next(keys), m.d_front, config.carrier.L_max + 1),
    }


def shared_byte_frontend(
    byte_ids: jax.Array, byte_mask: jax.Array, params: dict[str, Any]
) -> jax.Array:
    x: jax.Array = params["byte_embed"][byte_ids.astype(jnp.int32)]
    x = jnp.where(byte_mask[..., None], x, 0.0)
    initial = jnp.zeros((x.shape[0], x.shape[-1]))

    def step(state: jax.Array, token: jax.Array) -> tuple[jax.Array, jax.Array]:
        candidate = jnp.tanh(
            _apply(token, params["frontend_in"]) + _apply(state, params["frontend_state"])
        )
        gate = jax.nn.sigmoid(_apply(token, params["frontend_gate"]))
        new_state = gate * candidate + (1.0 - gate) * state
        return new_state, new_state

    _, hidden = jax.lax.scan(step, initial, jnp.swapaxes(x, 0, 1))
    hidden = jnp.asarray(hidden)
    return jnp.swapaxes(hidden, 0, 1)


def masked_mean(hidden: jax.Array, mask: jax.Array) -> jax.Array:
    weights = mask.astype(hidden.dtype)[..., None]
    return jnp.sum(hidden * weights, axis=1) / jnp.maximum(jnp.sum(weights, axis=1), 1.0)


def encode_context(
    byte_ids: jax.Array, byte_mask: jax.Array, params: dict[str, Any]
) -> tuple[jax.Array, jax.Array]:
    hidden = _apply(shared_byte_frontend(byte_ids, byte_mask, params), params["context_proj"])
    return hidden, masked_mean(hidden, byte_mask)


def encode_target(
    byte_ids: jax.Array,
    byte_mask: jax.Array,
    params: dict[str, Any],
    config: ReferenceConfig,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    hidden = shared_byte_frontend(byte_ids, byte_mask, params)
    pooled = masked_mean(hidden, byte_mask)
    carrier = jnp.broadcast_to(
        pooled[:, None, :], (pooled.shape[0], config.carrier.C, pooled.shape[-1])
    )
    h0 = _apply(carrier, params["target_h"])
    b0 = _apply(carrier, params["target_b"])
    l0 = _apply(carrier, params["target_l"])
    b0 = b0.at[:, -1, 1].set(10.0)
    b0 = b0.at[:, -1, 0].set(-10.0)
    return h0, b0, l0
