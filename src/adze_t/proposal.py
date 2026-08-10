"""Deterministic SSM-style carrier proposal."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp

from .config import ReferenceConfig


def init_proposal_params(key: jax.Array, config: ReferenceConfig) -> dict[str, Any]:
    m = config.model
    keys = jax.random.split(key, 5)

    def linear(k: jax.Array, i: int, o: int) -> dict[str, jax.Array]:
        return {"weight": jax.random.normal(k, (i, o)) / jnp.sqrt(i), "bias": jnp.zeros((o,))}

    return {
        "context": linear(keys[0], m.d_ctx, m.proposal_hidden_dim),
        "position": jax.random.normal(keys[1], (config.carrier.C, m.proposal_hidden_dim))
        / jnp.sqrt(m.proposal_hidden_dim),
        "state": linear(keys[2], m.proposal_hidden_dim, m.proposal_hidden_dim),
        "h": linear(keys[3], m.proposal_hidden_dim, config.carrier.h_dim),
        "b": linear(keys[4], m.proposal_hidden_dim, 2),
        "l": linear(keys[0], m.proposal_hidden_dim, config.carrier.L_max + 1),
    }


def apply_proposal(
    context_global: jax.Array, params: dict[str, Any], config: ReferenceConfig
) -> tuple[jax.Array, jax.Array, jax.Array]:
    state = jnp.tanh(context_global @ params["context"]["weight"] + params["context"]["bias"])
    positions = params["position"]

    def step(
        carry: jax.Array, pos: jax.Array
    ) -> tuple[jax.Array, tuple[jax.Array, jax.Array, jax.Array]]:
        carry = jnp.tanh(carry @ params["state"]["weight"] + pos)
        h = carry @ params["h"]["weight"] + params["h"]["bias"]
        b = carry @ params["b"]["weight"] + params["b"]["bias"]
        length_logits = carry @ params["l"]["weight"] + params["l"]["bias"]
        return carry, (h, b, length_logits)

    _, (h, b, length_logits) = jax.lax.scan(step, state, positions)
    h, b, length_logits = (jnp.swapaxes(x, 0, 1) for x in (h, b, length_logits))
    b = b.at[:, -1, 1].set(10.0)
    b = b.at[:, -1, 0].set(-10.0)
    return h, b, length_logits
