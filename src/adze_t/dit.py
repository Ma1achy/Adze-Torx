"""Faithful backend-driven looped Transformer/DiT core."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp

from .backends.deterministic import DeterministicOps
from .backends.protocol import LearnedOps
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
    max_blocks: int = 32
    max_slots: int = 8
    max_extent: int = 4
    residual_gate_init: float = 0.1


def init_dit_params(
    key: jax.Array, config: DiTConfig, ops: LearnedOps | None = None
) -> dict[str, Any]:
    """Initialise L physical blocks; no parameter is indexed by Q cycle."""
    ops = ops or DeterministicOps()
    keys = iter(jax.random.split(key, 9 + config.physical_blocks * 8))
    params: dict[str, Any] = {
        "input_proj": ops.init_linear(next(keys), config.d_model, config.d_model),
        "output_proj": ops.init_linear(next(keys), config.d_model, config.d_model),
        "carrier_embed": ops.init_embedding(next(keys), config.carrier_capacity, config.d_model),
        "block_embed": ops.init_embedding(next(keys), config.max_blocks, config.d_model),
        "within_embed": ops.init_embedding(next(keys), config.max_slots, config.d_model),
        "length_embed": ops.init_embedding(next(keys), config.max_extent + 1, config.d_model),
        "boundary_left_embed": ops.init_embedding(next(keys), 3, config.d_model),
        "boundary_right_embed": ops.init_embedding(next(keys), 2, config.d_model),
        "conditioning_trunk": ops.init_linear(
            next(keys), config.d_context + 5 * 16, config.d_model
        ),
        "blocks": [],
    }
    for _ in range(config.physical_blocks):
        modulation = ops.init_linear(next(keys), config.d_model, 6 * config.d_model, scale=0.02)
        bias = modulation["bias"]
        bias = bias.at[2 * config.d_model : 3 * config.d_model].set(config.residual_gate_init)
        bias = bias.at[5 * config.d_model : 6 * config.d_model].set(config.residual_gate_init)
        modulation = {**modulation, "bias": bias}
        params["blocks"].append(
            {
                "modulation": modulation,
                "q": ops.init_linear(next(keys), config.d_model, config.heads * config.head_dim),
                "k": ops.init_linear(next(keys), config.d_model, config.heads * config.head_dim),
                "v": ops.init_linear(next(keys), config.d_model, config.heads * config.head_dim),
                "o": ops.init_linear(next(keys), config.heads * config.head_dim, config.d_model),
                "up": ops.init_linear(next(keys), config.d_model, config.ffn_hidden),
                "gate": ops.init_linear(next(keys), config.d_model, config.ffn_hidden),
                "down": ops.init_linear(next(keys), config.ffn_hidden, config.d_model),
            }
        )
    return params


def _layer_norm(x: jax.Array, eps: float = 1.0e-5) -> jax.Array:
    mean = jnp.mean(x, axis=-1, keepdims=True)
    variance = jnp.mean((x - mean) ** 2, axis=-1, keepdims=True)
    return (x - mean) / jnp.sqrt(variance + eps)


def apply_rope(q: jax.Array, k: jax.Array, carrier_id: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Apply deterministic RoPE using persistent carrier coordinates."""
    width = q.shape[-1]
    half = width // 2
    frequencies = jnp.exp(-jnp.log(10000.0) * jnp.arange(half, dtype=q.dtype) / max(half - 1, 1))
    phase = carrier_id.astype(q.dtype)[..., None] * frequencies
    cos = jnp.cos(phase)[:, :, None, :]
    sin = jnp.sin(phase)[:, :, None, :]

    def rotate(x):
        first, second = x[..., :half], x[..., half : 2 * half]
        rotated = jnp.concatenate([first * cos - second * sin, first * sin + second * cos], -1)
        if width % 2:
            rotated = jnp.concatenate([rotated, x[..., -1:]], -1)
        return rotated

    return rotate(q), rotate(k)


def masked_softmax(scores: jax.Array, mask: jax.Array) -> jax.Array:
    """Softmax over allowed keys, returning exact zeros for empty rows."""
    mask = jnp.broadcast_to(mask, scores.shape)
    has_value = jnp.any(mask, axis=-1, keepdims=True)
    masked_scores = jnp.where(mask, scores, -jnp.inf)
    row_max = jnp.where(has_value, jnp.max(masked_scores, axis=-1, keepdims=True), 0.0)
    unnormalised = jnp.where(mask, jnp.exp(masked_scores - row_max), 0.0)
    denominator = jnp.sum(unnormalised, axis=-1, keepdims=True)
    return unnormalised / jnp.where(has_value, denominator, 1.0)


def apply_attention(
    x: jax.Array,
    block: dict[str, Any],
    mask: jax.Array,
    carrier_id: jax.Array,
    config: DiTConfig,
    ops: LearnedOps,
    *,
    name: str,
) -> jax.Array:
    batch, positions, _ = x.shape
    q = ops.linear(x, block["q"], name=f"{name}.q").reshape(
        batch, positions, config.heads, config.head_dim
    )
    k = ops.linear(x, block["k"], name=f"{name}.k").reshape(
        batch, positions, config.heads, config.head_dim
    )
    v = ops.linear(x, block["v"], name=f"{name}.v").reshape(
        batch, positions, config.heads, config.head_dim
    )
    q, k = apply_rope(q, k, carrier_id)
    scores = jnp.einsum("bthd,bshd->bhts", q, k) / jnp.sqrt(config.head_dim)
    expanded_mask = mask[:, None, :, :]
    weights = masked_softmax(scores, expanded_mask)
    attended = jnp.einsum("bhts,bshd->bthd", weights, v).reshape(batch, positions, -1)
    projected = ops.linear(attended, block["o"], name=f"{name}.o")
    has_kv = jnp.any(mask, axis=-1)
    return jnp.where(has_kv[..., None], projected, 0.0)


def apply_physical_block(
    x: jax.Array,
    block: dict[str, Any],
    conditioning: jax.Array,
    mask: jax.Array,
    carrier_id: jax.Array,
    query_mask: jax.Array,
    config: DiTConfig,
    ops: LearnedOps,
    *,
    block_index: int,
) -> jax.Array:
    prefix = f"dit.block_{block_index}"
    modulation = ops.linear(conditioning, block["modulation"], name=f"{prefix}.modulation")
    shift_a, scale_a, gate_a, shift_f, scale_f, gate_f = jnp.split(modulation, 6, axis=-1)
    normed = _layer_norm(x) * (1.0 + scale_a[:, None, :]) + shift_a[:, None, :]
    x = x + gate_a[:, None, :] * apply_attention(
        normed, block, mask, carrier_id, config, ops, name=prefix
    )
    normed = _layer_norm(x) * (1.0 + scale_f[:, None, :]) + shift_f[:, None, :]
    up = ops.linear(normed, block["up"], name=f"{prefix}.ffn_up")
    gate = jax.nn.silu(ops.linear(normed, block["gate"], name=f"{prefix}.ffn_gate"))
    x = x + gate_f[:, None, :] * ops.linear(up * gate, block["down"], name=f"{prefix}.ffn_down")
    return jnp.where(query_mask[..., None], x, 0.0)


def apply_dit_cycle(
    x: jax.Array,
    params: dict[str, Any],
    prompt_global: jax.Array,
    mask: jax.Array,
    carrier_id: jax.Array,
    query_mask: jax.Array,
    config: DiTConfig,
    *,
    cycle_index: int,
    ops: LearnedOps | None = None,
    mode: str = "draft",
    noise: jax.Array | float = 0.0,
    denoise_step: jax.Array | int = 0,
    refinement_step: jax.Array | int = 0,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Apply one tied physical stack at recurrence position ``cycle_index``."""
    ops = ops or DeterministicOps()
    block_rms = []
    block_states = []
    depths = []
    for block_index, block in enumerate(params["blocks"]):
        depth = cycle_index * config.physical_blocks + block_index
        block_ops = ops.with_occurrence(
            recurrence_cycle=cycle_index,
            physical_layer=block_index,
            denoise_step=denoise_step,
            refinement_step=refinement_step,
        )
        raw_conditioning = build_conditioning(
            prompt_global,
            noise,
            0 if mode == "draft" else 1,
            denoise_step,
            refinement_step,
            depth,
        )
        conditioning = jax.nn.silu(
            block_ops.linear(
                raw_conditioning,
                params["conditioning_trunk"],
                name="dit.conditioning_trunk",
            )
        )
        x = apply_physical_block(
            x,
            block,
            conditioning,
            mask,
            carrier_id,
            query_mask,
            config,
            block_ops,
            block_index=block_index,
        )
        depths.append(depth)
        block_rms.append(jnp.sqrt(jnp.mean(x**2)))
        block_states.append(x)
    return x, jnp.stack(block_rms), jnp.asarray(depths, dtype=jnp.int32), jnp.stack(block_states)


def apply_dit(
    packed: jax.Array,
    metadata: PackMetadata,
    prompt_global: jax.Array,
    params: dict[str, Any],
    config: DiTConfig,
    *,
    ops: LearnedOps | None = None,
    mode: str = "draft",
    noise: jax.Array | float = 0.0,
    denoise_step: jax.Array | int = 0,
    refinement_step: jax.Array | int = 0,
    cycles: int | None = None,
    observed_b: jax.Array | None = None,
    observed_l: jax.Array | None = None,
) -> tuple[jax.Array, dict[str, Any]]:
    """Apply (B_L ... B_1)^Q with block parameters tied across Q."""
    ops = ops or DeterministicOps()
    ops = ops.with_scope(f"mode:{mode}")
    cycle_count = config.cycles if cycles is None else cycles
    batch, blocks, slots, _ = packed.shape
    positions = blocks * slots
    query = metadata.query_mask.reshape(batch, positions)
    kv = metadata.kv_mask.reshape(batch, positions)
    block_id = metadata.block_id.reshape(batch, positions)
    carrier_id = metadata.carrier_id.reshape(batch, positions)
    safe_carrier = jnp.maximum(carrier_id, 0)
    within = metadata.within_block_pos.reshape(batch, positions)
    capacity = metadata.carrier_to_m.shape[1]
    if observed_b is None:
        observed_b = jnp.zeros((batch, capacity), dtype=jnp.int32)
    if observed_l is None:
        observed_l = jnp.zeros((batch, capacity), dtype=jnp.int32)
    right_boundary = observed_b[jnp.arange(batch)[:, None], safe_carrier].astype(jnp.int32)
    previous_carrier = jnp.maximum(safe_carrier - 1, 0)
    left_boundary = observed_b[jnp.arange(batch)[:, None], previous_carrier].astype(jnp.int32)
    left_boundary = jnp.where(safe_carrier == 0, 2, left_boundary)
    packed_length = observed_l[jnp.arange(batch)[:, None], safe_carrier].astype(jnp.int32)

    x = packed.reshape(batch, positions, -1)
    x = ops.linear(x, params["input_proj"], name="dit.input_proj")
    x = (
        x
        + ops.embedding(safe_carrier, params["carrier_embed"], name="dit.carrier_embed")
        + ops.embedding(block_id, params["block_embed"], name="dit.block_embed")
        + ops.embedding(within, params["within_embed"], name="dit.within_embed")
        + ops.embedding(packed_length, params["length_embed"], name="dit.length_embed")
        + ops.embedding(
            left_boundary, params["boundary_left_embed"], name="dit.boundary_left_embed"
        )
        + ops.embedding(
            right_boundary, params["boundary_right_embed"], name="dit.boundary_right_embed"
        )
    )
    x = jnp.where(query[..., None], x, 0.0)
    mask = (
        draft_attention_mask(block_id, query, kv)
        if mode == "draft"
        else refine_attention_mask(query, kv)
    )

    cycle_states = []
    block_rms = []
    depths = []
    block_states = []
    for cycle in range(cycle_count):
        x, cycle_block_rms, cycle_depths, cycle_block_states = apply_dit_cycle(
            x,
            params,
            prompt_global,
            mask,
            safe_carrier,
            query,
            config,
            cycle_index=cycle,
            ops=ops,
            mode=mode,
            noise=noise,
            denoise_step=denoise_step,
            refinement_step=refinement_step,
        )
        block_rms.append(cycle_block_rms)
        depths.append(cycle_depths)
        block_states.append(cycle_block_states)
        cycle_states.append(x)
    x = ops.linear(x, params["output_proj"], name="dit.output_proj")
    x = jnp.where(query[..., None], x, 0.0)
    return x.reshape(batch, blocks, slots, -1), {
        "trajectory": jnp.stack(cycle_states),
        "block_trajectory": jnp.concatenate(block_states, axis=0),
        "mask": mask,
        "effective_depths": jnp.concatenate(depths),
        "block_rms": jnp.concatenate(block_rms),
        "cycle_rms": jnp.stack([jnp.sqrt(jnp.mean(state**2)) for state in cycle_states]),
    }
