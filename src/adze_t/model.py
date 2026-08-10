"""Shared high-level deterministic/Torx-ready Adze model topology."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp

from .config import REFERENCE_SMALL_V0, ReferenceConfig
from .decoder import apply_decoder, init_decoder_params
from .dit import DiTConfig, apply_dit, init_dit_params
from .encoder import encode_context, encode_target, init_encoder_params
from .packing import build_pack_metadata_core, pack_values
from .proposal import apply_proposal, init_proposal_params
from .unpool import unpool_values


def _linear(key: jax.Array, in_dim: int, out_dim: int) -> dict[str, jax.Array]:
    return {
        "weight": jax.random.normal(key, (in_dim, out_dim)) / jnp.sqrt(in_dim),
        "bias": jnp.zeros((out_dim,)),
    }


def init_model_params(
    key: jax.Array, config: ReferenceConfig = REFERENCE_SMALL_V0
) -> dict[str, Any]:
    keys = jax.random.split(key, 8)
    m = config.model
    dit_config = DiTConfig(
        d_model=m.d_model,
        heads=m.heads,
        head_dim=m.head_dim,
        ffn_hidden=m.ffn_hidden,
        physical_blocks=m.physical_blocks_L,
        cycles=m.cycles_Q,
        carrier_capacity=config.carrier.C,
        d_context=m.d_ctx,
    )
    return {
        "encoder": init_encoder_params(keys[0], config),
        "proposal": init_proposal_params(keys[1], config),
        "dit": init_dit_params(keys[2], dit_config),
        "decoder": init_decoder_params(keys[3], config),
        "carrier_in": _linear(keys[4], config.carrier.h_dim, m.d_model),
        "carrier_out": _linear(keys[5], m.d_model, config.carrier.h_dim),
        "h_head": _linear(keys[6], m.d_model, config.carrier.h_dim),
        "b_head": _linear(keys[7], m.d_model, 2),
        "l_head": _linear(keys[6], m.d_model, config.carrier.L_max + 1),
    }


def _apply_linear(x: jax.Array, params: dict[str, jax.Array]) -> jax.Array:
    return x @ params["weight"] + params["bias"]


def _default_structure(
    target_mask: jax.Array, config: ReferenceConfig
) -> tuple[jax.Array, jax.Array, jax.Array]:
    batch = target_mask.shape[0]
    carrier_positions = jnp.arange(config.carrier.C)
    base_boundaries = ((carrier_positions + 1) % config.packing.K == 0).astype(jnp.int32)
    base_boundaries = base_boundaries.at[-1].set(1)
    c_b = jnp.broadcast_to(base_boundaries, (batch, config.carrier.C))
    length = jnp.zeros((batch, config.carrier.C), dtype=jnp.int32)
    n = jnp.minimum(jnp.sum(target_mask, axis=1), config.carrier.C)
    length = (jnp.arange(config.carrier.C)[None, :] < n[:, None]).astype(jnp.int32)
    activity = length > 0
    return c_b, activity, length


def apply_model(
    params: dict[str, Any],
    prompt: jax.Array,
    prompt_mask: jax.Array,
    target: jax.Array,
    target_mask: jax.Array,
    *,
    config: ReferenceConfig = REFERENCE_SMALL_V0,
    committed_c_b: jax.Array | None = None,
    committed_activity: jax.Array | None = None,
    committed_length: jax.Array | None = None,
    mode: str = "draft",
) -> dict[str, Any]:
    """Run S=1,R=0 deterministic Adze with teacher/fixed routing."""
    context_seq, context_global = encode_context(prompt, prompt_mask, params["encoder"])
    h0, b0, l0 = encode_target(target, target_mask, params["encoder"], config)
    default_b, default_a, default_l = _default_structure(target_mask, config)
    c_b = default_b if committed_c_b is None else committed_c_b
    activity = default_a if committed_activity is None else committed_activity
    length = default_l if committed_length is None else committed_length
    proposal_h, proposal_b, proposal_l = apply_proposal(context_global, params["proposal"], config)
    metadata = build_pack_metadata_core(
        c_b, activity, M_max=config.packing.M_max, K=config.packing.K
    )
    packed = pack_values(_apply_linear(proposal_h, params["carrier_in"]), metadata)
    dit_config = DiTConfig(
        d_model=config.model.d_model,
        heads=config.model.heads,
        head_dim=config.model.head_dim,
        ffn_hidden=config.model.ffn_hidden,
        physical_blocks=config.model.physical_blocks_L,
        cycles=config.model.cycles_Q,
        carrier_capacity=config.carrier.C,
        d_context=config.model.d_ctx,
    )
    packed_out, dit_aux = apply_dit(
        packed, metadata, context_global, params["dit"], dit_config, mode=mode
    )
    carrier_model = unpool_values(packed_out, metadata, C=config.carrier.C)
    carrier_delta = _apply_linear(carrier_model, params["carrier_out"])
    h_updated = proposal_h + carrier_delta
    h_hat = _apply_linear(carrier_model, params["h_head"])
    b_logits = _apply_linear(carrier_model, params["b_head"])
    l_logits = _apply_linear(carrier_model, params["l_head"])
    byte_logits, emit_mask = apply_decoder(h_updated, length, params["decoder"], config)
    return {
        "context_seq": context_seq,
        "context_global": context_global,
        "target": (h0, b0, l0),
        "proposal": (proposal_h, proposal_b, proposal_l),
        "carrier": h_updated,
        "prediction": (h_hat, b_logits, l_logits),
        "byte_logits": byte_logits,
        "emit_mask": emit_mask,
        "metadata": metadata,
        "dit_aux": dit_aux,
    }
