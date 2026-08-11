"""Shared Mamba frontend plus context and clean-target analysis branches."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp

from .backends.deterministic import DeterministicOps
from .backends.protocol import LearnedOps
from .config import ReferenceConfig
from .mamba import MambaConfig, apply_mamba_stack, init_mamba_stack
from .teacher import canonical_teacher_structure


def _mamba_config(width: int, layers: int, config: ReferenceConfig) -> MambaConfig:
    return MambaConfig(
        width=width,
        layers=layers,
        expand=config.model.mamba_expand,
        state_dim=config.model.mamba_state_dim,
        conv_kernel=config.model.mamba_conv_kernel,
    )


def init_encoder_params(
    key: jax.Array, config: ReferenceConfig, ops: LearnedOps | None = None
) -> dict[str, Any]:
    ops = ops or DeterministicOps()
    m = config.model
    keys = iter(jax.random.split(key, 11))
    return {
        "byte_embed": ops.init_embedding(next(keys), m.byte_vocab, m.d_front),
        "frontend": init_mamba_stack(
            next(keys), _mamba_config(m.d_front, m.frontend_layers, config), ops, name="frontend"
        ),
        "context_in": ops.init_linear(next(keys), m.d_front, m.d_ctx),
        "context": init_mamba_stack(
            next(keys), _mamba_config(m.d_ctx, m.context_layers, config), ops, name="context"
        ),
        "target": init_mamba_stack(
            next(keys), _mamba_config(m.d_front, m.target_layers, config), ops, name="target"
        ),
        "target_slot_embed": ops.init_embedding(next(keys), config.carrier.L_max, m.d_front),
        "target_carrier_embed": ops.init_embedding(next(keys), config.carrier.C, m.d_front),
        "target_pool": ops.init_linear(next(keys), config.carrier.L_max * m.d_front, m.d_front),
        "target_h": ops.init_linear(next(keys), m.d_front, config.carrier.h_dim),
        "target_b": ops.init_linear(next(keys), config.carrier.h_dim, 2),
        "target_l": ops.init_linear(next(keys), config.carrier.h_dim, config.carrier.L_max + 1),
    }


def shared_byte_frontend(
    byte_ids: jax.Array,
    byte_mask: jax.Array,
    params: dict[str, Any],
    config: ReferenceConfig,
    ops: LearnedOps | None = None,
) -> jax.Array:
    ops = ops or DeterministicOps()
    hidden = ops.embedding(byte_ids, params["byte_embed"], name="frontend.byte_embed")
    hidden = jnp.where(byte_mask[..., None], hidden, 0.0)
    frontend_config = _mamba_config(config.model.d_front, config.model.frontend_layers, config)
    return apply_mamba_stack(
        hidden, params["frontend"], frontend_config, ops, name="frontend", mask=byte_mask
    )


def masked_mean(hidden: jax.Array, mask: jax.Array) -> jax.Array:
    weights = mask.astype(hidden.dtype)[..., None]
    return jnp.sum(hidden * weights, axis=1) / jnp.maximum(jnp.sum(weights, axis=1), 1.0)


def encode_context_from_hidden(
    prompt_hidden: jax.Array,
    prompt_mask: jax.Array,
    params: dict[str, Any],
    config: ReferenceConfig,
    ops: LearnedOps,
) -> tuple[jax.Array, jax.Array]:
    hidden = ops.linear(prompt_hidden, params["context_in"], name="context.input_proj")
    context_config = _mamba_config(config.model.d_ctx, config.model.context_layers, config)
    hidden = apply_mamba_stack(
        hidden, params["context"], context_config, ops, name="context", mask=prompt_mask
    )
    return hidden, masked_mean(hidden, prompt_mask)


def encode_context(
    byte_ids: jax.Array,
    byte_mask: jax.Array,
    params: dict[str, Any],
    config: ReferenceConfig,
    ops: LearnedOps | None = None,
) -> tuple[jax.Array, jax.Array]:
    ops = ops or DeterministicOps()
    frontend = shared_byte_frontend(byte_ids, byte_mask, params, config, ops)
    return encode_context_from_hidden(frontend, byte_mask, params, config, ops)


def encode_target_from_hidden(
    target_hidden: jax.Array,
    target_bytes: jax.Array,
    target_mask: jax.Array,
    params: dict[str, Any],
    config: ReferenceConfig,
    ops: LearnedOps,
) -> dict[str, Any]:
    target_config = _mamba_config(config.model.d_front, config.model.target_layers, config)
    analysed = apply_mamba_stack(
        target_hidden, params["target"], target_config, ops, name="target", mask=target_mask
    )
    teacher = canonical_teacher_structure(target_bytes, target_mask, config)
    batch = analysed.shape[0]
    total_slots = config.carrier.C * config.carrier.L_max
    padded = jnp.pad(
        analysed,
        ((0, 0), (0, max(0, total_slots - analysed.shape[1])), (0, 0)),
    )[:, :total_slots]
    slots = padded.reshape(batch, config.carrier.C, config.carrier.L_max, config.model.d_front)
    slot_ids = jnp.arange(config.carrier.L_max)[None, None, :]
    slots = slots + ops.embedding(slot_ids, params["target_slot_embed"], name="target.slot_embed")
    slots = jnp.where(teacher.slot_mask[..., None], slots, 0.0)
    pooled = ops.linear(
        slots.reshape(batch, config.carrier.C, -1),
        params["target_pool"],
        name="target.pool",
    )
    carrier_ids = jnp.arange(config.carrier.C)[None, :]
    pooled = pooled + ops.embedding(
        carrier_ids, params["target_carrier_embed"], name="target.carrier_embed"
    )
    h0 = ops.linear(pooled, params["target_h"], name="target.h")
    b_logits = ops.categorical_logits(h0, params["target_b"], name="target.b")
    l_logits = ops.categorical_logits(h0, params["target_l"], name="target.l")
    return {
        "h0": h0,
        "b_logits": b_logits,
        "l_logits": l_logits,
        "teacher": teacher,
    }


def encode_target(
    byte_ids: jax.Array,
    byte_mask: jax.Array,
    params: dict[str, Any],
    config: ReferenceConfig,
    ops: LearnedOps | None = None,
) -> dict[str, Any]:
    ops = ops or DeterministicOps()
    frontend = shared_byte_frontend(byte_ids, byte_mask, params, config, ops)
    return encode_target_from_hidden(frontend, byte_ids, byte_mask, params, config, ops)
