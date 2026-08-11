"""Corrected deterministic Phase B codec and model training utilities."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp

from .backends.deterministic import DeterministicOps
from .config import REFERENCE_SMALL_V0, ReferenceConfig
from .model import apply_model, apply_target_codec, init_model_params
from .objectives import (
    adamw_init,
    adamw_step,
    codec_loss_components,
    emitted_metrics,
    loss_components,
    total_loss,
)
from .teacher import canonical_teacher_structure


_CODEC_ENCODER_NAMES = (
    "byte_embed",
    "frontend",
    "target",
    "target_slot_embed",
    "target_carrier_embed",
    "target_pool",
    "target_h",
    "target_b",
    "target_l",
)


def _constant_mask(tree: Any, enabled: bool) -> Any:
    return jax.tree_util.tree_map(lambda _: jnp.asarray(enabled), tree)


def codec_update_mask(params: dict[str, Any]) -> dict[str, Any]:
    """Select only target-codec leaves for codec pretraining."""
    mask = _constant_mask(params, False)
    encoder = dict(mask["encoder"])
    for name in _CODEC_ENCODER_NAMES:
        encoder[name] = _constant_mask(params["encoder"][name], True)
    return {**mask, "encoder": encoder, "decoder": _constant_mask(params["decoder"], True)}


def model_update_mask(params: dict[str, Any]) -> dict[str, Any]:
    """Freeze the established target codec during ordinary Phase-B training."""
    mask = _constant_mask(params, True)
    encoder = dict(mask["encoder"])
    for name in _CODEC_ENCODER_NAMES:
        encoder[name] = _constant_mask(params["encoder"][name], False)
    return {**mask, "encoder": encoder}


def _norm(*trees: Any) -> jax.Array:
    leaves = [leaf for tree in trees for leaf in jax.tree_util.tree_leaves(tree)]
    return jnp.sqrt(jnp.sum(jnp.stack([jnp.sum(leaf * leaf) for leaf in leaves])))


def _gradient_metrics(grads: dict[str, Any]) -> dict[str, jax.Array]:
    encoder = grads["encoder"]
    dit = grads["dit"]
    qkvo = [block[name] for block in dit["blocks"] for name in ("q", "k", "v", "o")]
    ffn = [block[name] for block in dit["blocks"] for name in ("up", "gate", "down")]
    modulation = [block["modulation"] for block in dit["blocks"]]
    return {
        "grad_frontend": _norm(encoder["byte_embed"], encoder["frontend"]),
        "grad_context_encoder": _norm(encoder["context_in"], encoder["context"]),
        "grad_target_encoder": _norm(
            encoder["target"],
            encoder["target_pool"],
            encoder["target_h"],
            encoder["target_b"],
            encoder["target_l"],
        ),
        "grad_proposal": _norm(grads["proposal"]),
        "grad_dit_qkvo": _norm(qkvo),
        "grad_dit_ffn": _norm(ffn),
        "grad_conditioning": _norm(dit["conditioning_trunk"], modulation),
        "grad_output_heads": _norm(grads["h_head"], grads["b_head"], grads["l_head"]),
        "grad_decoder": _norm(grads["decoder"]),
    }


def train_step(
    params: Any,
    moments: tuple[Any, Any],
    step: jax.Array | int,
    batch: dict[str, jax.Array],
    *,
    config: ReferenceConfig = REFERENCE_SMALL_V0,
) -> tuple[Any, tuple[Any, Any], dict[str, jax.Array]]:
    ops = DeterministicOps()

    def objective(p):
        outputs = apply_model(
            p,
            batch["prompt"],
            batch["prompt_mask"],
            batch["target"],
            batch["target_mask"],
            config=config,
            ops=ops,
        )
        components = loss_components(outputs)
        return total_loss(components, config), (components, outputs)

    (loss, (components, outputs)), grads = jax.value_and_grad(objective, has_aux=True)(params)
    gradient_metrics = _gradient_metrics(grads)
    params, moments, grad_norm = adamw_step(
        params,
        grads,
        moments,
        step,
        learning_rate=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
        clip_norm=config.training.grad_clip_norm,
        update_mask=model_update_mask(params),
    )
    teacher = outputs["target"]["teacher"]
    byte_accuracy, sequence_accuracy = emitted_metrics(
        outputs["byte_logits"], teacher.slot_bytes, teacher.slot_mask
    )
    _, b_logits, l_logits = outputs["prediction"]
    boundary_accuracy = jnp.mean(
        jnp.argmax(b_logits[:, :-1], axis=-1) == teacher.boundaries[:, :-1]
    )
    length_accuracy = jnp.mean(jnp.argmax(l_logits, axis=-1) == teacher.length)
    return (
        params,
        moments,
        {
            "loss": loss,
            **components,
            "byte_accuracy": byte_accuracy,
            "sequence_accuracy": sequence_accuracy,
            "boundary_accuracy": boundary_accuracy,
            "length_accuracy": length_accuracy,
            "grad_norm": grad_norm,
            **gradient_metrics,
            "activation_packed_input": outputs["activation_rms"]["packed_input"],
            "activation_unpooled_carrier": outputs["activation_rms"]["unpooled_carrier"],
            "activation_block_rms": outputs["dit_aux"]["block_rms"],
            "activation_cycle_rms": outputs["dit_aux"]["cycle_rms"],
        },
    )


def codec_pretrain_step(
    params: Any,
    moments: tuple[Any, Any],
    step: jax.Array | int,
    batch: dict[str, jax.Array],
    *,
    config: ReferenceConfig = REFERENCE_SMALL_V0,
) -> tuple[Any, tuple[Any, Any], dict[str, jax.Array]]:
    def objective(p):
        outputs = apply_target_codec(
            p,
            batch["target"],
            batch["target_mask"],
            config=config,
        )
        components = codec_loss_components(outputs)
        loss = jnp.sum(jnp.stack(list(components.values())))
        return loss, (components, outputs)

    (loss, (components, outputs)), grads = jax.value_and_grad(objective, has_aux=True)(params)
    params, moments, grad_norm = adamw_step(
        params,
        grads,
        moments,
        step,
        learning_rate=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
        clip_norm=config.training.grad_clip_norm,
        update_mask=codec_update_mask(params),
    )
    teacher = outputs["target"]["teacher"]
    byte_accuracy, sequence_accuracy = emitted_metrics(
        outputs["codec_logits"], teacher.slot_bytes, teacher.slot_mask
    )
    return (
        params,
        moments,
        {
            "loss": loss,
            **components,
            "byte_accuracy": byte_accuracy,
            "sequence_accuracy": sequence_accuracy,
            "grad_norm": grad_norm,
        },
    )


def make_fixed_structure_batch(
    prompt: jax.Array,
    target: jax.Array,
    *,
    prompt_mask: jax.Array | None = None,
    target_mask: jax.Array | None = None,
    config: ReferenceConfig = REFERENCE_SMALL_V0,
) -> dict[str, jax.Array]:
    """Build a validated batch; byte value zero is ordinary data, not padding."""
    if prompt_mask is None:
        prompt_mask = jnp.ones_like(prompt, dtype=bool)
    if target_mask is None:
        target_mask = jnp.ones_like(target, dtype=bool)
    if prompt_mask.shape != prompt.shape or target_mask.shape != target.shape:
        raise ValueError("byte masks must match their corresponding byte arrays")
    teacher = canonical_teacher_structure(target, target_mask, config)
    return {
        "prompt": prompt,
        "prompt_mask": prompt_mask,
        "target": target,
        "target_mask": target_mask,
        "committed_c_b": teacher.boundaries,
        "committed_length": teacher.length,
        "committed_activity": teacher.activity,
        "boundary_target": teacher.boundaries,
        "length_target": teacher.length,
    }


def initialise_training(
    key: jax.Array, config: ReferenceConfig = REFERENCE_SMALL_V0
) -> tuple[Any, tuple[Any, Any]]:
    params = init_model_params(key, config)
    zeros = adamw_init(params)
    return params, (zeros, zeros)
