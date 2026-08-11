"""Backend-driven fixed-slot Mamba byte decoder."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp

from .backends.deterministic import DeterministicOps
from .backends.protocol import LearnedOps
from .config import ReferenceConfig
from .mamba import MambaConfig, apply_mamba_stack, init_mamba_stack


def _config(config: ReferenceConfig) -> MambaConfig:
    return MambaConfig(
        width=config.model.d_dec,
        layers=config.model.decoder_layers,
        expand=config.model.mamba_expand,
        state_dim=config.model.mamba_state_dim,
        conv_kernel=config.model.mamba_conv_kernel,
    )


def init_decoder_params(
    key: jax.Array, config: ReferenceConfig, ops: LearnedOps | None = None
) -> dict[str, Any]:
    ops = ops or DeterministicOps()
    keys = iter(jax.random.split(key, 5))
    return {
        "h": ops.init_linear(next(keys), config.carrier.h_dim, config.model.d_dec),
        "slot_embed": ops.init_embedding(next(keys), config.carrier.L_max, config.model.d_dec),
        "carrier_embed": ops.init_embedding(next(keys), config.carrier.C, config.model.d_dec),
        "stack": init_mamba_stack(next(keys), _config(config), ops, name="decoder"),
        "out": ops.init_linear(next(keys), config.model.d_dec, config.model.byte_vocab),
    }


def apply_decoder(
    h: jax.Array,
    length: jax.Array,
    params: dict[str, Any],
    config: ReferenceConfig,
    ops: LearnedOps | None = None,
    *,
    name: str = "decoder",
) -> tuple[jax.Array, jax.Array]:
    ops = ops or DeterministicOps()
    batch = h.shape[0]
    slots = config.carrier.L_max
    base = ops.linear(h, params["h"], name=f"{name}.h")[:, :, None, :]
    carrier_ids = jnp.arange(config.carrier.C)[None, :, None]
    slot_ids = jnp.arange(slots)[None, None, :]
    candidate = (
        base
        + ops.embedding(carrier_ids, params["carrier_embed"], name=f"{name}.carrier_embed")
        + ops.embedding(slot_ids, params["slot_embed"], name=f"{name}.slot_embed")
    )
    emit = slot_ids < length[:, :, None]
    flat = candidate.reshape(batch, config.carrier.C * slots, config.model.d_dec)
    flat_mask = emit.reshape(batch, config.carrier.C * slots)
    hidden = apply_mamba_stack(
        flat, params["stack"], _config(config), ops, name=f"{name}.stack", mask=flat_mask
    )
    logits = ops.categorical_logits(hidden, params["out"], name=f"{name}.out")
    logits = logits.reshape(batch, config.carrier.C, slots, config.model.byte_vocab)
    return logits, emit
