"""Reusable paired Monte Carlo evaluation for finite-noise Adze execution."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import jax
import jax.numpy as jnp

from .backends.torx import TorxOperatorConfig, TorxOps
from .config import REFERENCE_SMALL_V0, ReferenceConfig
from .model import apply_model
from .objectives import emitted_metrics, loss_components, total_loss


_T_975 = {1: 12.7062, 2: 4.3027, 3: 3.1824, 7: 2.3646, 15: 2.1314, 31: 2.0395}


def phase_d_root(base_seed: int, index: int) -> jax.Array:
    """Derive the nested, deterministic root at ``index`` from ``base_seed``."""
    return jax.random.fold_in(jax.random.PRNGKey(base_seed), index)


def phase_d_stage_names(config: ReferenceConfig = REFERENCE_SMALL_V0) -> tuple[str, ...]:
    names = ["frontend", "proposal", "pack"]
    for cycle in range(config.model.cycles_Q):
        for layer in range(config.model.physical_blocks_L):
            names.append(f"dit.q{cycle}.block{layer}")
        names.append(f"dit.q{cycle}.cycle")
    return (*names, "unpool", "h_hat", "carrier", "decoder_logits")


def _stage_values(outputs: dict[str, Any], config: ReferenceConfig) -> tuple[Any, ...]:
    stages: list[Any] = [
        outputs["prompt_frontend"],
        outputs["proposal"],
        outputs["packed_carrier"],
    ]
    block_trajectory = outputs["dit_aux"]["block_trajectory"]
    cycle_trajectory = outputs["dit_aux"]["trajectory"]
    for cycle in range(config.model.cycles_Q):
        offset = cycle * config.model.physical_blocks_L
        stages.extend(
            block_trajectory[offset + layer] for layer in range(config.model.physical_blocks_L)
        )
        stages.append(cycle_trajectory[cycle])
    stages.extend(
        (
            outputs["unpooled_carrier"],
            outputs["prediction"][0],
            outputs["carrier"],
            outputs["byte_logits"],
        )
    )
    return tuple(stages)


def _tree_square_sum(tree: Any) -> jax.Array:
    return jnp.sum(jnp.stack([jnp.sum(leaf**2) for leaf in jax.tree_util.tree_leaves(tree)]))


def _tree_count(tree: Any) -> jax.Array:
    return jnp.sum(
        jnp.asarray([leaf.size for leaf in jax.tree_util.tree_leaves(tree)], dtype=jnp.int32)
    )


def _tree_nonfinite(tree: Any) -> jax.Array:
    counts = [
        jnp.sum(~jnp.isfinite(leaf))
        for leaf in jax.tree_util.tree_leaves(tree)
        if jnp.issubdtype(leaf.dtype, jnp.inexact)
    ]
    return jnp.sum(jnp.stack(counts))


def paired_chunk_statistics(
    params: Any,
    prompt: jax.Array,
    target: jax.Array,
    root: jax.Array,
    lambda_op: float | jax.Array,
    *,
    config: ReferenceConfig = REFERENCE_SMALL_V0,
) -> dict[str, jax.Array]:
    """Evaluate one validation chunk against its exact paired mean trajectory."""
    mask_prompt = jnp.ones_like(prompt, dtype=bool)
    mask_target = jnp.ones_like(target, dtype=bool)
    clean_ops = TorxOps.create(
        root, config=TorxOperatorConfig(operator_stochasticity=True, lambda_op=0.0)
    )
    noisy_ops = TorxOps.create(
        root,
        config=TorxOperatorConfig(operator_stochasticity=True, lambda_op=lambda_op),
    )
    clean = apply_model(
        params,
        prompt,
        mask_prompt,
        target,
        mask_target,
        config=config,
        ops=clean_ops,
        target_ops=clean_ops,
    )
    noisy = apply_model(
        params,
        prompt,
        mask_prompt,
        target,
        mask_target,
        config=config,
        ops=noisy_ops,
        target_ops=clean_ops,
    )
    components = loss_components(noisy)
    teacher = noisy["target"]["teacher"]
    byte_accuracy, exact_accuracy = emitted_metrics(
        noisy["byte_logits"], teacher.slot_bytes, teacher.slot_mask
    )
    clean_stages = _stage_values(clean, config)
    noisy_stages = _stage_values(noisy, config)
    signal_square_sums = jnp.stack([_tree_square_sum(stage) for stage in clean_stages])
    perturbation_square_sums = jnp.stack(
        [
            _tree_square_sum(
                jax.tree.map(lambda left, right: left - right, noisy_stage, clean_stage)
            )
            for noisy_stage, clean_stage in zip(noisy_stages, clean_stages, strict=True)
        ]
    )
    stage_counts = jnp.stack([_tree_count(stage) for stage in clean_stages])
    nonfinite_count = jnp.sum(jnp.stack([_tree_nonfinite(stage) for stage in noisy_stages]))
    nonfinite_total = jnp.sum(stage_counts)
    return {
        "loss": total_loss(components, config),
        "byte_accuracy": byte_accuracy,
        "exact_sequence_accuracy": exact_accuracy,
        "nonfinite_count": nonfinite_count,
        "nonfinite_total": nonfinite_total,
        "signal_square_sums": signal_square_sums,
        "perturbation_square_sums": perturbation_square_sums,
        "stage_counts": stage_counts,
    }


def aggregate_root_chunks(
    chunks: Iterable[dict[str, Any]], weights: Iterable[int]
) -> dict[str, Any]:
    """Combine fixed-size chunk statistics into one root-level observation."""
    chunk_list = list(chunks)
    weight_list = list(weights)
    total_weight = sum(weight_list)
    result = {
        name: sum(
            chunk[name] * weight for chunk, weight in zip(chunk_list, weight_list, strict=True)
        )
        / total_weight
        for name in ("loss", "byte_accuracy", "exact_sequence_accuracy")
    }
    signal_sum = sum(chunk["signal_square_sums"] for chunk in chunk_list)
    perturbation_sum = sum(chunk["perturbation_square_sums"] for chunk in chunk_list)
    stage_count = sum(chunk["stage_counts"] for chunk in chunk_list)
    nonfinite_count = sum(chunk["nonfinite_count"] for chunk in chunk_list)
    nonfinite_total = sum(chunk["nonfinite_total"] for chunk in chunk_list)
    return {
        **result,
        "nonfinite_rate": nonfinite_count / jnp.maximum(nonfinite_total, 1),
        "signal_rms": jnp.sqrt(signal_sum / stage_count),
        "perturbation_rms": jnp.sqrt(perturbation_sum / stage_count),
    }


def student_t_summary(values: list[float]) -> dict[str, Any]:
    """Summarize root observations with a two-sided Student-t 95% interval."""
    array = jnp.asarray(values, dtype=jnp.float32)
    count = int(array.size)
    mean = float(jnp.mean(array))
    if count == 1:
        return {"count": 1, "mean": mean, "sample_sd": 0.0, "ci95": [mean, mean]}
    sample_sd = float(jnp.std(array, ddof=1))
    degrees = count - 1
    if degrees not in _T_975:
        raise ValueError(f"no frozen Student-t critical value for {degrees} degrees of freedom")
    radius = _T_975[degrees] * sample_sd / count**0.5
    return {
        "count": count,
        "mean": mean,
        "sample_sd": sample_sd,
        "ci95": [mean - radius, mean + radius],
    }
