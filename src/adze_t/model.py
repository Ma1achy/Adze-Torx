"""One backend-driven Adze topology shared by deterministic and future Torx ops."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp

from .backends.deterministic import DeterministicOps
from .backends.protocol import LearnedOps
from .config import REFERENCE_SMALL_V0, ReferenceConfig
from .decoder import apply_decoder, init_decoder_params
from .dit import DiTConfig, apply_dit, init_dit_params
from .encoder import (
    encode_context_from_hidden,
    encode_target_from_hidden,
    init_encoder_params,
    shared_byte_frontend,
)
from .packing import build_pack_metadata_core, pack_values, trim_padding_blocks
from .proposal import apply_proposal, init_proposal_params
from .unpool import unpool_values


CLEAN_TARGET_MEAN_TEACHER_V0 = "CLEAN_TARGET_MEAN_TEACHER_V0"


def _dit_config(config: ReferenceConfig) -> DiTConfig:
    m = config.model
    return DiTConfig(
        d_model=m.d_model,
        heads=m.heads,
        head_dim=m.head_dim,
        ffn_hidden=m.ffn_hidden,
        physical_blocks=m.physical_blocks_L,
        cycles=m.cycles_Q,
        carrier_capacity=config.carrier.C,
        d_context=m.d_ctx,
        max_blocks=config.packing.M_max,
        max_slots=config.packing.K,
        max_extent=config.carrier.L_max,
        residual_gate_init=m.residual_gate_init,
        effective_depth_conditioning=m.effective_depth_conditioning,
    )


def init_model_params(
    key: jax.Array,
    config: ReferenceConfig = REFERENCE_SMALL_V0,
    ops: LearnedOps | None = None,
) -> dict[str, Any]:
    ops = ops or DeterministicOps()
    keys = iter(jax.random.split(key, 11))
    m = config.model
    return {
        "encoder": init_encoder_params(next(keys), config, ops),
        "proposal": init_proposal_params(next(keys), config, ops),
        "dit": init_dit_params(next(keys), _dit_config(config), ops),
        "decoder": init_decoder_params(next(keys), config, ops),
        "carrier_prior": ops.init_embedding(next(keys), config.carrier.C, config.carrier.h_dim),
        "carrier_in": ops.init_linear(next(keys), config.carrier.h_dim, m.d_model),
        "carrier_out": ops.init_linear(next(keys), m.d_model, config.carrier.h_dim),
        "h_head": ops.init_linear(next(keys), config.carrier.h_dim, config.carrier.h_dim),
        "b_head": ops.init_linear(next(keys), config.carrier.h_dim, 2),
        "l_head": ops.init_linear(next(keys), config.carrier.h_dim, config.carrier.L_max + 1),
    }


def apply_target_codec(
    params: dict[str, Any],
    target: jax.Array,
    target_mask: jax.Array,
    *,
    config: ReferenceConfig = REFERENCE_SMALL_V0,
    ops: LearnedOps | None = None,
) -> dict[str, Any]:
    """Run only the training-time target codec used before B3.

    Keeping this path separate avoids compiling the heavy DiT while the clean
    carrier representation is being established. The target-side parameters
    are frozen before ordinary Phase-B model training.
    """
    ops = ops or DeterministicOps()
    if target.shape[1] > config.carrier.C * config.carrier.L_max:
        raise ValueError("target sequence width exceeds carrier emission capacity")
    target_analysis = apply_clean_target_teacher(
        params, target, target_mask, config=config, ops=ops
    )
    target_codec = target_analysis["target"]
    codec_logits, codec_emit_mask = apply_decoder(
        target_codec["h0"],
        target_codec["teacher"].length,
        params["decoder"],
        config,
        ops.with_scope("target_codec_decoder"),
        name="decoder",
    )
    return {
        "target": target_codec,
        "codec_logits": codec_logits,
        "codec_emit_mask": codec_emit_mask,
    }


def apply_clean_target_teacher(
    params: dict[str, Any],
    target: jax.Array,
    target_mask: jax.Array,
    *,
    config: ReferenceConfig = REFERENCE_SMALL_V0,
    ops: LearnedOps | None = None,
) -> dict[str, Any]:
    """Prepare the frozen clean target analysis used only by losses and corruption."""
    ops = (ops or DeterministicOps()).with_scope("target")
    target_frontend = shared_byte_frontend(target, target_mask, params["encoder"], config, ops)
    target_codec = encode_target_from_hidden(
        target_frontend, target, target_mask, params["encoder"], config, ops
    )
    return {"target_frontend": target_frontend, "target": target_codec}


def apply_model(
    params: dict[str, Any],
    prompt: jax.Array,
    prompt_mask: jax.Array,
    target: jax.Array,
    target_mask: jax.Array,
    *,
    config: ReferenceConfig = REFERENCE_SMALL_V0,
    ops: LearnedOps | None = None,
    target_ops: LearnedOps | None = None,
    target_analysis: dict[str, Any] | None = None,
    carrier_h_input: jax.Array | None = None,
    noise_level: jax.Array | float = 0.0,
    denoise_step: jax.Array | int = 0,
    committed_c_b: jax.Array | None = None,
    committed_activity: jax.Array | None = None,
    committed_length: jax.Array | None = None,
    mode: str = "draft",
    dit_cycles: int | None = None,
    depth_code_override: str = "correct",
    suppress_cycle: int | None = None,
    shuffle_cycle: int | None = None,
    stop_gradient_after_cycle: int | None = None,
    capture_diagnostics: bool = False,
) -> dict[str, Any]:
    """Run the shared S=1,R=0 graph with deterministic teacher routing."""
    ops = ops or DeterministicOps()
    if target.shape[1] > config.carrier.C * config.carrier.L_max:
        raise ValueError("target sequence width exceeds carrier emission capacity")
    prompt_ops = ops.with_scope("prompt")
    prompt_frontend = shared_byte_frontend(
        prompt, prompt_mask, params["encoder"], config, prompt_ops
    )
    context_seq, context_global = encode_context_from_hidden(
        prompt_frontend, prompt_mask, params["encoder"], config, prompt_ops
    )
    if target_analysis is None:
        target_analysis = apply_clean_target_teacher(
            params,
            target,
            target_mask,
            config=config,
            ops=target_ops or ops,
        )
    target_frontend = target_analysis["target_frontend"]
    target_codec = target_analysis["target"]
    teacher = target_codec["teacher"]
    c_b = teacher.boundaries if committed_c_b is None else committed_c_b
    length = teacher.length if committed_length is None else committed_length
    activity = length > 0
    if committed_length is None and committed_activity is not None:
        # Phase-A compatibility only. New Phase-B paths derive activity from length.
        activity = committed_activity

    carrier_ids = jnp.arange(config.carrier.C)[None, :]
    carrier_prior = ops.embedding(
        carrier_ids, params["carrier_prior"], name="proposal.carrier_prior"
    )
    carrier_prior = jnp.broadcast_to(
        carrier_prior, (prompt.shape[0], config.carrier.C, config.carrier.h_dim)
    )
    proposal_h, proposal_b, proposal_l = apply_proposal(
        context_global, carrier_prior, params["proposal"], config, ops
    )
    current_carrier = proposal_h if carrier_h_input is None else carrier_h_input
    if current_carrier.shape != proposal_h.shape:
        raise ValueError("carrier_h_input must match the proposal carrier shape")
    metadata = build_pack_metadata_core(
        c_b, activity, M_max=config.packing.M_max, K=config.packing.K
    )
    if committed_c_b is None:
        # PROVISIONAL_PHASE_B_TEACHER has a static K-bucket partition. Its
        # generated M is known exactly; M_max remains the validated capacity.
        teacher_blocks = (config.carrier.C + config.packing.K - 1) // config.packing.K
        if teacher_blocks > config.packing.M_max:
            raise ValueError("Phase-B teacher block count exceeds M_max")
        metadata = trim_padding_blocks(metadata, teacher_blocks)
    carrier_input = ops.linear(current_carrier, params["carrier_in"], name="model.carrier_input")
    packed = pack_values(carrier_input, metadata)
    packed_out, dit_aux = apply_dit(
        packed,
        metadata,
        context_global,
        params["dit"],
        _dit_config(config),
        ops=ops,
        mode=mode,
        noise=noise_level,
        denoise_step=denoise_step,
        observed_b=c_b,
        observed_l=length,
        cycles=dit_cycles,
        depth_code_override=depth_code_override,
        suppress_cycle=suppress_cycle,
        shuffle_cycle=shuffle_cycle,
        stop_gradient_after_cycle=stop_gradient_after_cycle,
        capture_diagnostics=capture_diagnostics,
    )
    unpooled = unpool_values(packed_out, metadata, C=config.carrier.C)
    carrier_delta = ops.linear(unpooled, params["carrier_out"], name="model.carrier_output")
    pre_head_carrier = current_carrier + carrier_delta
    h_hat = ops.linear(pre_head_carrier, params["h_head"], name="model.h_head")
    b_logits = ops.categorical_logits(pre_head_carrier, params["b_head"], name="model.b_head")
    l_logits = ops.categorical_logits(pre_head_carrier, params["l_head"], name="model.l_head")

    # Phase-B fixed clean commit: predicted clean content reaches the decoder;
    # teacher structure controls routing/emission and is not rewritten in-step.
    h_final = h_hat
    byte_logits, emit_mask = apply_decoder(
        h_final,
        length,
        params["decoder"],
        config,
        ops.with_scope("output"),
        name="decoder",
    )
    return {
        "prompt_frontend": prompt_frontend,
        "target_frontend": target_frontend,
        "context_seq": context_seq,
        "context_global": context_global,
        "target": target_codec,
        "proposal": (proposal_h, proposal_b, proposal_l),
        "current_carrier_input": current_carrier,
        "pre_head_carrier": pre_head_carrier,
        "carrier": h_final,
        "prediction": (h_hat, b_logits, l_logits),
        "byte_logits": byte_logits,
        "emit_mask": emit_mask,
        "metadata": metadata,
        "packed_carrier": packed,
        "packed_output": packed_out,
        "unpooled_carrier": unpooled,
        "dit_aux": dit_aux,
        "activation_rms": {
            "packed_input": jnp.sqrt(jnp.mean(packed**2)),
            "unpooled_carrier": jnp.sqrt(jnp.mean(unpooled**2)),
        },
    }
