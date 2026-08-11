"""Corrected Phase B clean-state, structure, codec, and byte objectives."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp

from .config import ReferenceConfig


def cross_entropy(logits: jax.Array, labels: jax.Array, mask: jax.Array) -> jax.Array:
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    selected = jnp.take_along_axis(log_probs, labels[..., None], axis=-1)[..., 0]
    weights = mask.astype(logits.dtype)
    return -jnp.sum(selected * weights) / jnp.maximum(jnp.sum(weights), 1.0)


def loss_components(outputs: dict[str, Any]) -> dict[str, jax.Array]:
    """Use frozen target-codec h0 and canonical teacher b0/l0 as clean targets."""
    h_hat, b_logits, l_logits = outputs["prediction"]
    target = outputs["target"]
    teacher = target["teacher"]
    clean_h = jax.lax.stop_gradient(target["h0"])
    proposal_h, proposal_b, proposal_l = outputs["proposal"]
    ones_b = jnp.ones_like(teacher.boundaries[:, :-1], dtype=bool)
    ones_l = jnp.ones_like(teacher.length, dtype=bool)
    return {
        "h": jnp.mean((h_hat - clean_h) ** 2),
        "b": cross_entropy(b_logits[:, :-1], teacher.boundaries[:, :-1], ones_b),
        "l": cross_entropy(l_logits, teacher.length, ones_l),
        "byte": cross_entropy(outputs["byte_logits"], teacher.slot_bytes, teacher.slot_mask),
        "proposal": (
            jnp.mean((proposal_h - clean_h) ** 2)
            + cross_entropy(proposal_b[:, :-1], teacher.boundaries[:, :-1], ones_b)
            + cross_entropy(proposal_l, teacher.length, ones_l)
        ),
    }


def codec_loss_components(outputs: dict[str, Any]) -> dict[str, jax.Array]:
    target = outputs["target"]
    teacher = target["teacher"]
    return {
        "codec_byte": cross_entropy(outputs["codec_logits"], teacher.slot_bytes, teacher.slot_mask),
        "codec_b": cross_entropy(
            target["b_logits"][:, :-1],
            teacher.boundaries[:, :-1],
            jnp.ones_like(teacher.boundaries[:, :-1], dtype=bool),
        ),
        "codec_l": cross_entropy(
            target["l_logits"],
            teacher.length,
            jnp.ones_like(teacher.length, dtype=bool),
        ),
    }


def total_loss(components: dict[str, jax.Array], config: ReferenceConfig) -> jax.Array:
    weights = config.training
    return (
        weights.h_weight * components["h"]
        + weights.boundary_weight * components["b"]
        + weights.extent_weight * components["l"]
        + weights.byte_weight * components["byte"]
        + weights.proposal_weight * components["proposal"]
    )


def emitted_metrics(
    logits: jax.Array, labels: jax.Array, mask: jax.Array
) -> tuple[jax.Array, jax.Array]:
    predicted = jnp.argmax(logits, axis=-1)
    correct = (predicted == labels) & mask
    byte_accuracy = jnp.sum(correct) / jnp.maximum(jnp.sum(mask), 1)
    sequence_accuracy = jnp.mean(jnp.all(correct | ~mask, axis=(1, 2)))
    return byte_accuracy, sequence_accuracy


def adamw_init(params: Any) -> Any:
    return jax.tree_util.tree_map(jnp.zeros_like, params)


def global_norm(tree: Any) -> jax.Array:
    return jnp.sqrt(jnp.sum(jnp.stack([jnp.sum(x * x) for x in jax.tree_util.tree_leaves(tree)])))


def adamw_step(
    params: Any,
    grads: Any,
    moments: tuple[Any, Any],
    step: jax.Array | int,
    *,
    learning_rate: float,
    weight_decay: float,
    clip_norm: float,
    update_mask: Any | None = None,
) -> tuple[Any, tuple[Any, Any], jax.Array]:
    """Apply AdamW, leaving masked parameters and moments bitwise unchanged."""
    m, v = moments
    if update_mask is None:
        update_mask = jax.tree_util.tree_map(lambda _: jnp.asarray(True), params)
    masked_grads = jax.tree_util.tree_map(
        lambda g, enabled: jnp.where(enabled, g, jnp.zeros_like(g)), grads, update_mask
    )
    grad_norm = global_norm(masked_grads)
    scale = jnp.minimum(1.0, clip_norm / jnp.maximum(grad_norm, 1.0e-8))
    masked_grads = jax.tree_util.tree_map(lambda g: g * scale, masked_grads)
    beta1, beta2 = 0.9, 0.999
    m = jax.tree_util.tree_map(
        lambda old, g, enabled: jnp.where(enabled, beta1 * old + (1 - beta1) * g, old),
        m,
        masked_grads,
        update_mask,
    )
    v = jax.tree_util.tree_map(
        lambda old, g, enabled: jnp.where(enabled, beta2 * old + (1 - beta2) * g * g, old),
        v,
        masked_grads,
        update_mask,
    )
    m_hat = jax.tree_util.tree_map(lambda x: x / (1 - beta1**step), m)
    v_hat = jax.tree_util.tree_map(lambda x: x / (1 - beta2**step), v)
    params = jax.tree_util.tree_map(
        lambda p, mh, vh, enabled: jnp.where(
            enabled,
            p - learning_rate * (mh / (jnp.sqrt(vh) + 1.0e-8) + weight_decay * p),
            p,
        ),
        params,
        m_hat,
        v_hat,
        update_mask,
    )
    return params, (m, v), grad_norm
