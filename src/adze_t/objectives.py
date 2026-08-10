"""Phase B deterministic clean-state and byte objectives."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp


def _cross_entropy(logits: jax.Array, labels: jax.Array, mask: jax.Array) -> jax.Array:
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    selected = jnp.take_along_axis(log_probs, labels[..., None], axis=-1)[..., 0]
    weights = mask.astype(logits.dtype)
    return -jnp.sum(selected * weights) / jnp.maximum(jnp.sum(weights), 1.0)


def loss_components(
    outputs: dict[str, Any],
    *,
    clean_h: jax.Array,
    boundary_target: jax.Array,
    length_target: jax.Array,
    target_bytes: jax.Array,
) -> dict[str, jax.Array]:
    """Return separately reported Phase B loss components."""
    h_hat, b_logits, l_logits = outputs["prediction"]
    slot_count = h_hat.shape[1] * outputs["emit_mask"].shape[2]
    target_slots = jnp.pad(target_bytes, ((0, 0), (0, max(0, slot_count - target_bytes.shape[1]))))
    target_slots = target_slots[:, :slot_count].reshape(
        target_bytes.shape[0], h_hat.shape[1], outputs["emit_mask"].shape[2]
    )
    emit_mask = outputs["emit_mask"]
    proposal_h, proposal_b, proposal_l = outputs["proposal"]
    components = {
        "h": jnp.mean((h_hat - clean_h) ** 2),
        "b": _cross_entropy(
            b_logits[:, :-1], boundary_target[:, :-1], jnp.ones(boundary_target[:, :-1].shape)
        ),
        "l": _cross_entropy(l_logits, length_target, jnp.ones(length_target.shape)),
        "byte": _cross_entropy(outputs["byte_logits"], target_slots, emit_mask),
        "proposal": (
            jnp.mean((proposal_h - clean_h) ** 2)
            + _cross_entropy(
                proposal_b[:, :-1], boundary_target[:, :-1], jnp.ones(boundary_target[:, :-1].shape)
            )
            + _cross_entropy(proposal_l, length_target, jnp.ones(length_target.shape))
        ),
    }
    return components


def total_loss(
    components: dict[str, jax.Array], weights: dict[str, float] | None = None
) -> jax.Array:
    weights = weights or {"h": 1.0, "b": 1.0, "l": 1.0, "byte": 1.0, "proposal": 0.25}
    return jnp.sum(jnp.stack([weights[name] * components[name] for name in components]))


def adamw_init(params: Any) -> Any:
    return jax.tree_util.tree_map(jnp.zeros_like, params)


def adamw_step(
    params: Any,
    grads: Any,
    moments: tuple[Any, Any],
    step: int,
    *,
    learning_rate: float = 3.0e-4,
    weight_decay: float = 0.01,
    clip_norm: float = 1.0,
) -> tuple[Any, tuple[Any, Any]]:
    m, v = moments
    grad_norm = jnp.sqrt(sum(jnp.sum(g * g) for g in jax.tree_util.tree_leaves(grads)))
    scale = jnp.minimum(1.0, clip_norm / jnp.maximum(grad_norm, 1.0e-8))
    grads = jax.tree_util.tree_map(lambda g: g * scale, grads)
    beta1, beta2 = 0.9, 0.999
    m = jax.tree_util.tree_map(lambda old, g: beta1 * old + (1 - beta1) * g, m, grads)
    v = jax.tree_util.tree_map(lambda old, g: beta2 * old + (1 - beta2) * g * g, v, grads)
    m_hat = jax.tree_util.tree_map(lambda x: x / (1 - beta1**step), m)
    v_hat = jax.tree_util.tree_map(lambda x: x / (1 - beta2**step), v)
    params = jax.tree_util.tree_map(
        lambda p, mh, vh: p - learning_rate * (mh / (jnp.sqrt(vh) + 1.0e-8) + weight_decay * p),
        params,
        m_hat,
        v_hat,
    )
    return params, (m, v)
