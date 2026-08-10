"""Small deterministic Phase B training utilities."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp

from .config import REFERENCE_SMALL_V0, ReferenceConfig
from .model import apply_model
from .objectives import adamw_init, adamw_step, loss_components, total_loss


def train_step(
    params: Any,
    moments: tuple[Any, Any],
    step: int,
    batch: dict[str, jax.Array],
    *,
    config: ReferenceConfig = REFERENCE_SMALL_V0,
) -> tuple[Any, tuple[Any, Any], dict[str, jax.Array]]:
    def objective(p: Any) -> tuple[jax.Array, dict[str, jax.Array]]:
        outputs = apply_model(
            p,
            batch["prompt"],
            batch["prompt_mask"],
            batch["target"],
            batch["target_mask"],
            config=config,
            committed_c_b=batch.get("committed_c_b"),
            committed_activity=batch.get("committed_activity"),
            committed_length=batch.get("committed_length"),
        )
        components = loss_components(
            outputs,
            clean_h=batch["clean_h"],
            boundary_target=batch["boundary_target"],
            length_target=batch["length_target"],
            target_bytes=batch["target"],
        )
        return total_loss(components), components

    (loss, components), grads = jax.value_and_grad(objective, has_aux=True)(params)
    params, moments = adamw_step(
        params,
        grads,
        moments,
        step,
        learning_rate=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
        clip_norm=config.training.grad_clip_norm,
    )
    metrics = {"loss": loss, **components}
    return params, moments, metrics


def make_fixed_structure_batch(
    prompt: jax.Array, target: jax.Array, *, config: ReferenceConfig = REFERENCE_SMALL_V0
) -> dict[str, jax.Array]:
    """Create a teacher-structure batch for S=1/R=0 training."""
    batch, width = target.shape
    c_b = jnp.zeros((batch, config.carrier.C), dtype=jnp.int32)
    c_b = c_b.at[:, config.packing.K - 1 :: config.packing.K].set(1)
    c_b = c_b.at[:, -1].set(1)
    length = (jnp.arange(config.carrier.C)[None, :] < width).astype(jnp.int32)
    return {
        "prompt": prompt,
        "prompt_mask": jnp.ones(prompt.shape, dtype=bool),
        "target": target,
        "target_mask": jnp.ones(target.shape, dtype=bool),
        "committed_c_b": c_b,
        "committed_length": length,
        "committed_activity": length > 0,
        "clean_h": jnp.zeros((batch, config.carrier.C, config.carrier.h_dim)),
        "boundary_target": c_b,
        "length_target": length,
    }


def initialise_training(
    key: jax.Array, config: ReferenceConfig = REFERENCE_SMALL_V0
) -> tuple[Any, tuple[Any, Any]]:
    from .model import init_model_params

    params = init_model_params(key, config)
    zeros = adamw_init(params)
    return params, (zeros, zeros)
