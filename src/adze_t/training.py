"""Corrected deterministic Phase B codec and model training utilities."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp

from .backends.deterministic import DeterministicOps
from .backends.torx import TorxOperatorConfig, TorxOps
from .config import REFERENCE_SMALL_V0, ReferenceConfig
from .model import (
    apply_clean_target_teacher,
    apply_model,
    apply_target_codec,
    init_model_params,
)
from .objectives import (
    adamw_init,
    adamw_step,
    codec_loss_components,
    emitted_metrics,
    loss_components,
    total_loss,
)
from .teacher import canonical_teacher_structure
from .phase_f_1 import make_initial_corruption


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
B3_INITIALIZATION_SEED = 700


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


def stochastic_model_update_mask(params: dict[str, Any]) -> dict[str, Any]:
    """Retain the Phase-B freeze boundary and disable every rho leaf."""
    phase_b = model_update_mask(params)
    return jax.tree_util.tree_map_with_path(
        lambda path, enabled: (
            jnp.asarray(enabled) & jnp.asarray("['rho']" not in jax.tree_util.keystr(path))
        ),
        phase_b,
    )


def _norm(*trees: Any) -> jax.Array:
    leaves = [leaf for tree in trees for leaf in jax.tree_util.tree_leaves(tree)]
    return jnp.sqrt(jnp.sum(jnp.stack([jnp.sum(leaf * leaf) for leaf in leaves])))


def _path_norm(tree: Any, predicate: Any) -> jax.Array:
    leaves = [
        leaf
        for path, leaf in jax.tree_util.tree_leaves_with_path(tree)
        if predicate(jax.tree_util.keystr(path))
    ]
    if not leaves:
        return jnp.asarray(0.0, dtype=jnp.float32)
    return jnp.sqrt(jnp.sum(jnp.stack([jnp.sum(leaf * leaf) for leaf in leaves])))


def _masked_gradients(grads: Any, update_mask: Any) -> Any:
    return jax.tree_util.tree_map(
        lambda grad, enabled: jnp.where(enabled, grad, jnp.zeros_like(grad)),
        grads,
        update_mask,
    )


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


def stochastic_train_step(
    params: Any,
    moments: tuple[Any, Any],
    step: jax.Array | int,
    batch: dict[str, jax.Array],
    training_root: jax.Array,
    *,
    config: ReferenceConfig = REFERENCE_SMALL_V0,
    lambda_op: float | jax.Array = 1.0,
) -> tuple[Any, tuple[Any, Any], dict[str, jax.Array]]:
    """Run one pathwise Phase-D trajectory with frozen rho and a clean teacher."""
    noisy_ops = TorxOps.create(
        training_root,
        config=TorxOperatorConfig(operator_stochasticity=True, lambda_op=lambda_op),
        optimizer_step=step,
    )

    clean_target_ops = TorxOps.create(
        training_root,
        config=TorxOperatorConfig(operator_stochasticity=True, lambda_op=0.0),
        optimizer_step=step,
    )

    def objective(p):
        outputs = apply_model(
            p,
            batch["prompt"],
            batch["prompt_mask"],
            batch["target"],
            batch["target_mask"],
            config=config,
            ops=noisy_ops,
            target_ops=clean_target_ops,
        )
        components = loss_components(outputs)
        return total_loss(components, config), (components, outputs)

    (loss, (components, outputs)), grads = jax.value_and_grad(objective, has_aux=True)(params)
    update_mask = stochastic_model_update_mask(params)
    permitted_grads = _masked_gradients(grads, update_mask)
    raw_norm = _norm(grads)
    permitted_norm = _norm(permitted_grads)
    clipped_norm = jnp.minimum(permitted_norm, config.training.grad_clip_norm)
    gradient_metrics = _gradient_metrics(grads)
    rho_norm = _path_norm(grads, lambda path: "['rho']" in path)
    direct_norm = _path_norm(
        grads,
        lambda path: any(
            f"['{name}']" in path for name in ("a_log", "d_skip", "delta_bias", "layer_scale")
        ),
    )
    direct_permitted_norm = _path_norm(
        permitted_grads,
        lambda path: any(
            f"['{name}']" in path for name in ("a_log", "d_skip", "delta_bias", "layer_scale")
        ),
    )
    clip_scale = jnp.minimum(
        1.0, config.training.grad_clip_norm / jnp.maximum(permitted_norm, 1.0e-8)
    )
    params, moments, optimizer_grad_norm = adamw_step(
        params,
        grads,
        moments,
        step,
        learning_rate=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
        clip_norm=config.training.grad_clip_norm,
        update_mask=update_mask,
    )
    teacher = outputs["target"]["teacher"]
    byte_accuracy, sequence_accuracy = emitted_metrics(
        outputs["byte_logits"], teacher.slot_bytes, teacher.slot_mask
    )
    return (
        params,
        moments,
        {
            "loss": loss,
            **components,
            "byte_accuracy": byte_accuracy,
            "sequence_accuracy": sequence_accuracy,
            "grad_raw_norm": raw_norm,
            "grad_permitted_norm": permitted_norm,
            "grad_clipped_applied_norm": clipped_norm,
            "grad_optimizer_reported_norm": optimizer_grad_norm,
            "grad_direct_ssm_norm": direct_norm,
            "grad_direct_ssm_applied_norm": direct_permitted_norm * clip_scale,
            "grad_rho_raw_norm": rho_norm,
            "grad_rho_applied_norm": jnp.asarray(0.0, dtype=rho_norm.dtype),
            **gradient_metrics,
            "activation_packed_input": outputs["activation_rms"]["packed_input"],
            "activation_unpooled_carrier": outputs["activation_rms"]["unpooled_carrier"],
            "activation_block_rms": outputs["dit_aux"]["block_rms"],
            "activation_cycle_rms": outputs["dit_aux"]["cycle_rms"],
        },
    )


def stochastic_denoise_train_step(
    params: Any,
    moments: tuple[Any, Any],
    step: jax.Array | int,
    batch: dict[str, jax.Array],
    operator_root: jax.Array,
    diffusion_root: jax.Array,
    *,
    config: ReferenceConfig = REFERENCE_SMALL_V0,
    lambda_op: float | jax.Array = 1.0,
) -> tuple[Any, tuple[Any, Any], dict[str, jax.Array]]:
    """Train one faithful S=1 x0 prediction from an occurrence-fresh corrupted carrier."""
    if config.training.proposal_weight != 0.0:
        raise ValueError("Phase-F denoising benchmarks require proposal_weight=0")
    noisy_ops = TorxOps.create(
        operator_root,
        config=TorxOperatorConfig(operator_stochasticity=True, lambda_op=lambda_op),
        optimizer_step=step,
    )
    clean_target_ops = TorxOps.create(
        operator_root,
        config=TorxOperatorConfig(operator_stochasticity=True, lambda_op=0.0),
        optimizer_step=step,
    )

    def objective(p):
        target_analysis = apply_clean_target_teacher(
            p,
            batch["target"],
            batch["target_mask"],
            config=config,
            ops=clean_target_ops,
        )
        clean_h = jax.lax.stop_gradient(target_analysis["target"]["h0"])
        carrier_h_input, epsilon = make_initial_corruption(
            clean_h,
            batch["nu"],
            diffusion_root,
            batch["global_example_id"],
            optimizer_step=batch["diffusion_occurrence"],
        )
        outputs = apply_model(
            p,
            batch["prompt"],
            batch["prompt_mask"],
            batch["target"],
            batch["target_mask"],
            config=config,
            ops=noisy_ops,
            target_ops=clean_target_ops,
            target_analysis=target_analysis,
            carrier_h_input=carrier_h_input,
            noise_level=batch["nu"],
            denoise_step=0,
        )
        components = loss_components(outputs)
        return total_loss(components, config), (components, outputs, epsilon)

    (loss, (components, outputs, epsilon)), grads = jax.value_and_grad(objective, has_aux=True)(
        params
    )
    update_mask = stochastic_model_update_mask(params)
    permitted_grads = _masked_gradients(grads, update_mask)
    raw_norm = _norm(grads)
    permitted_norm = _norm(permitted_grads)
    clipped_norm = jnp.minimum(permitted_norm, config.training.grad_clip_norm)
    gradient_metrics = _gradient_metrics(grads)
    rho_norm = _path_norm(grads, lambda path: "['rho']" in path)
    direct_names = ("a_log", "d_skip", "delta_bias", "layer_scale")
    direct_norm = _path_norm(
        grads, lambda path: any(f"['{name}']" in path for name in direct_names)
    )
    direct_permitted_norm = _path_norm(
        permitted_grads,
        lambda path: any(f"['{name}']" in path for name in direct_names),
    )
    clip_scale = jnp.minimum(
        1.0, config.training.grad_clip_norm / jnp.maximum(permitted_norm, 1.0e-8)
    )
    params, moments, optimizer_grad_norm = adamw_step(
        params,
        grads,
        moments,
        step,
        learning_rate=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
        clip_norm=config.training.grad_clip_norm,
        update_mask=update_mask,
    )
    teacher = outputs["target"]["teacher"]
    byte_accuracy, sequence_accuracy = emitted_metrics(
        outputs["byte_logits"], teacher.slot_bytes, teacher.slot_mask
    )
    return (
        params,
        moments,
        {
            "loss": loss,
            **components,
            "byte_accuracy": byte_accuracy,
            "sequence_accuracy": sequence_accuracy,
            "grad_raw_norm": raw_norm,
            "grad_permitted_norm": permitted_norm,
            "grad_clipped_applied_norm": clipped_norm,
            "grad_optimizer_reported_norm": optimizer_grad_norm,
            "grad_direct_ssm_norm": direct_norm,
            "grad_direct_ssm_applied_norm": direct_permitted_norm * clip_scale,
            "grad_rho_raw_norm": rho_norm,
            "grad_rho_applied_norm": jnp.asarray(0.0, dtype=rho_norm.dtype),
            **gradient_metrics,
            "diffusion_epsilon_rms": jnp.sqrt(jnp.mean(epsilon**2)),
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


def accepted_b3_scratch_initialization(
    target_codec_params: Any,
    config: ReferenceConfig = REFERENCE_SMALL_V0,
) -> Any:
    """Reproduce B3: fresh seed-700 generative state plus accepted clean codec.

    This deliberately does not consume task-trained COPY/REVERSE parameters.
    The codec update mask identifies the exact leaves copied from
    ``target_codec_b1.pkl``; every other leaf comes from a fresh deterministic
    initialization using the same seed/procedure as the accepted B3 run.
    """
    fresh = init_model_params(jax.random.PRNGKey(B3_INITIALIZATION_SEED), config)
    mask = codec_update_mask(fresh)
    return jax.tree_util.tree_map(
        lambda initialized, codec, use_codec: jnp.where(use_codec, codec, initialized),
        fresh,
        target_codec_params,
        mask,
    )
