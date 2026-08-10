"""Faithful deterministic looped Transformer/DiT core."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp

from .conditioning import build_conditioning
from .masking import draft_attention_mask, refine_attention_mask
from .packing import PackMetadata


@dataclass(frozen=True)
class DiTConfig:
    d_model: int = 128
    heads: int = 4
    head_dim: int = 32
    ffn_hidden: int = 256
    physical_blocks: int = 4
    cycles: int = 3
    carrier_capacity: int = 32
    d_context: int = 128


def _linear(key: jax.Array, in_dim: int, out_dim: int, scale: float = 1.0) -> dict[str, jax.Array]:
    weight = jax.random.normal(key, (in_dim, out_dim)) * (scale / jnp.sqrt(in_dim))
    return {"weight": weight, "bias": jnp.zeros((out_dim,))}


def init_dit_params(key: jax.Array, config: DiTConfig) -> dict[str, Any]:
    keys = iter(jax.random.split(key, 4 + config.physical_blocks * 7))
    params: dict[str, Any] = {
        "input_proj": _linear(next(keys), config.d_model, config.d_model),
        "output_proj": _linear(next(keys), config.d_model, config.d_model),
        "carrier_embed": jax.random.normal(next(keys), (config.carrier_capacity, config.d_model))
        / jnp.sqrt(config.d_model),
        "conditioning": _linear(next(keys), config.d_context + 5 * 16, 6 * config.d_model),
        "blocks": [],
    }
    for _ in range(config.physical_blocks):
        params["blocks"].append(
            {
                "q": _linear(next(keys), config.d_model, config.heads * config.head_dim),
                "k": _linear(next(keys), config.d_model, config.heads * config.head_dim),
                "v": _linear(next(keys), config.d_model, config.heads * config.head_dim),
                "o": _linear(next(keys), config.heads * config.head_dim, config.d_model),
                "up": _linear(next(keys), config.d_model, config.ffn_hidden),
                "gate": _linear(next(keys), config.d_model, config.ffn_hidden),
                "down": _linear(next(keys), config.ffn_hidden, config.d_model),
            }
        )
    return params


def _apply_linear(x: jax.Array, params: dict[str, jax.Array]) -> jax.Array:
    return x @ params["weight"] + params["bias"]


def _layer_norm(x: jax.Array, eps: float = 1.0e-5) -> jax.Array:
    mean = jnp.mean(x, axis=-1, keepdims=True)
    variance = jnp.mean((x - mean) ** 2, axis=-1, keepdims=True)
    return (x - mean) / jnp.sqrt(variance + eps)


def _attention(
    x: jax.Array, block: dict[str, Any], mask: jax.Array, config: DiTConfig
) -> jax.Array:
    batch, positions, _ = x.shape
    q = _apply_linear(x, block["q"]).reshape(batch, positions, config.heads, config.head_dim)
    k = _apply_linear(x, block["k"]).reshape(batch, positions, config.heads, config.head_dim)
    v = _apply_linear(x, block["v"]).reshape(batch, positions, config.heads, config.head_dim)
    scores = jnp.einsum("bthd,bshd->bhts", q, k) / jnp.sqrt(config.head_dim)
    # A large finite sentinel keeps fully padded query rows differentiable;
    # their outputs are explicitly zeroed by the caller.
    scores = jnp.where(mask[:, None, :, :], scores, -1.0e9)
    weights = jax.nn.softmax(scores, axis=-1)
    attended = jnp.einsum("bhts,bshd->bthd", weights, v).reshape(batch, positions, -1)
    return _apply_linear(attended, block["o"])


def _block(
    x: jax.Array, block: dict[str, Any], modulation: jax.Array, mask: jax.Array, config: DiTConfig
) -> jax.Array:
    shift_a, scale_a, gate_a, shift_f, scale_f, gate_f = jnp.split(modulation, 6, axis=-1)
    normed = _layer_norm(x) * (1.0 + scale_a[:, None, :]) + shift_a[:, None, :]
    x = x + 0.1 * gate_a[:, None, :] * _attention(normed, block, mask, config)
    normed = _layer_norm(x) * (1.0 + scale_f[:, None, :]) + shift_f[:, None, :]
    up = _apply_linear(normed, block["up"])
    gate = jax.nn.silu(_apply_linear(normed, block["gate"]))
    x = x + 0.1 * gate_f[:, None, :] * _apply_linear(up * gate, block["down"])
    return x


def apply_dit(
    packed: jax.Array,
    metadata: PackMetadata,
    prompt_global: jax.Array,
    params: dict[str, Any],
    config: DiTConfig,
    *,
    mode: str = "draft",
    noise: jax.Array | float = 0.0,
    denoise_step: jax.Array | int = 0,
    refinement_step: jax.Array | int = 0,
) -> tuple[jax.Array, dict[str, Any]]:
    """Apply the L-physical-block stack repeatedly for Q tied cycles."""
    batch, blocks, slots, _ = packed.shape
    positions = blocks * slots
    x = packed.reshape(batch, positions, -1)
    carrier_ids = jnp.maximum(metadata.carrier_id.reshape(batch, positions), 0)
    x = _apply_linear(x, params["input_proj"]) + params["carrier_embed"][carrier_ids]
    query = metadata.query_mask.reshape(batch, positions)
    kv = metadata.kv_mask.reshape(batch, positions)
    block_id = metadata.block_id.reshape(batch, positions)
    mask = (
        draft_attention_mask(block_id, query, kv)
        if mode == "draft"
        else refine_attention_mask(query, kv)
    )
    cond = build_conditioning(
        prompt_global, noise, 0 if mode == "draft" else 1, denoise_step, refinement_step, 0
    )
    modulation = _apply_linear(cond, params["conditioning"])
    trajectories = []
    for cycle in range(config.cycles):
        for block in params["blocks"]:
            depth = cycle * config.physical_blocks + len(trajectories) % config.physical_blocks
            del depth
            x = _block(x, block, modulation, mask, config)
            x = jnp.where(query[..., None], x, 0.0)
        trajectories.append(x)
    x = _apply_linear(x, params["output_proj"])
    return x.reshape(batch, blocks, slots, -1), {
        "trajectory": jnp.stack(trajectories),
        "mask": mask,
    }
