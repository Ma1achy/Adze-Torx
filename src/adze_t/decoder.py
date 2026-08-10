"""Deterministic fixed-slot Mamba-style byte decoder."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp

from .config import ReferenceConfig


def init_decoder_params(key: jax.Array, config: ReferenceConfig) -> dict[str, Any]:
    m = config.model
    keys = jax.random.split(key, 4)

    def linear(k: jax.Array, i: int, o: int) -> dict[str, jax.Array]:
        return {"weight": jax.random.normal(k, (i, o)) / jnp.sqrt(i), "bias": jnp.zeros((o,))}

    return {
        "h": linear(keys[0], config.carrier.h_dim, m.d_dec),
        "state": linear(keys[1], m.d_dec, m.d_dec),
        "gate": linear(keys[2], m.d_dec, m.d_dec),
        "out": linear(keys[3], m.d_dec, config.model.byte_vocab),
        "slot_embed": jax.random.normal(keys[0], (config.carrier.L_max, m.d_dec))
        / jnp.sqrt(m.d_dec),
        "carrier_embed": jax.random.normal(keys[1], (config.carrier.C, m.d_dec))
        / jnp.sqrt(m.d_dec),
    }


def apply_decoder(
    h: jax.Array, length: jax.Array, params: dict[str, Any], config: ReferenceConfig
) -> tuple[jax.Array, jax.Array]:
    batch, capacity = h.shape[:2]
    slots = config.carrier.L_max
    base = h @ params["h"]["weight"] + params["h"]["bias"]
    base = (
        base[:, :, None, :]
        + params["carrier_embed"][None, :, None, :]
        + params["slot_embed"][None, None, :, :]
    )
    state = jnp.zeros((batch, base.shape[-1]))
    flat = base.reshape(batch, capacity * slots, -1)

    def step(carry: jax.Array, token: jax.Array) -> tuple[jax.Array, jax.Array]:
        proposal = jnp.tanh(token @ params["state"]["weight"] + carry @ params["gate"]["weight"])
        return proposal, proposal

    _, hidden = jax.lax.scan(step, state, jnp.swapaxes(flat, 0, 1))
    hidden = jnp.swapaxes(hidden, 0, 1).reshape(batch, capacity, slots, -1)
    logits = hidden @ params["out"]["weight"] + params["out"]["bias"]
    emit = jnp.arange(slots)[None, None, :] < length[:, :, None]
    return logits, emit
