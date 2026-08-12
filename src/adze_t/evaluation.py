"""Reusable paired Monte Carlo evaluation for finite-noise Adze execution."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import jax
import jax.numpy as jnp

from .backends.torx import TorxOperatorConfig, TorxOps
from .config import REFERENCE_SMALL_V0, ReferenceConfig
from .model import apply_model
from .objectives import total_loss


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


def _component_numerators(outputs: dict[str, Any]) -> tuple[jax.Array, jax.Array]:
    """Return loss sums/counts in the exact normalization used by the objective."""
    h_hat, b_logits, l_logits = outputs["prediction"]
    target = outputs["target"]
    teacher = target["teacher"]
    clean_h = jax.lax.stop_gradient(target["h0"])
    proposal_h, proposal_b, proposal_l = outputs["proposal"]

    def ce_parts(
        logits: jax.Array, labels: jax.Array, mask: jax.Array
    ) -> tuple[jax.Array, jax.Array]:
        log_probs = jax.nn.log_softmax(logits, axis=-1)
        selected = jnp.take_along_axis(log_probs, labels[..., None], axis=-1)[..., 0]
        weights = mask.astype(logits.dtype)
        return -jnp.sum(selected * weights), jnp.sum(weights)

    h_sum = jnp.sum((h_hat - clean_h) ** 2)
    h_count = jnp.asarray(h_hat.size, dtype=h_hat.dtype)
    b_sum, b_count = ce_parts(
        b_logits[:, :-1],
        teacher.boundaries[:, :-1],
        jnp.ones_like(teacher.boundaries[:, :-1], bool),
    )
    l_sum, l_count = ce_parts(l_logits, teacher.length, jnp.ones_like(teacher.length, bool))
    byte_sum, byte_count = ce_parts(outputs["byte_logits"], teacher.slot_bytes, teacher.slot_mask)
    proposal_h_sum = jnp.sum((proposal_h - clean_h) ** 2)
    proposal_h_count = jnp.asarray(proposal_h.size, dtype=proposal_h.dtype)
    proposal_b_sum, proposal_b_count = ce_parts(
        proposal_b[:, :-1],
        teacher.boundaries[:, :-1],
        jnp.ones_like(teacher.boundaries[:, :-1], bool),
    )
    proposal_l_sum, proposal_l_count = ce_parts(
        proposal_l, teacher.length, jnp.ones_like(teacher.length, bool)
    )
    return (
        jnp.stack((h_sum, b_sum, l_sum, byte_sum, proposal_h_sum, proposal_b_sum, proposal_l_sum)),
        jnp.stack(
            (
                h_count,
                b_count,
                l_count,
                byte_count,
                proposal_h_count,
                proposal_b_count,
                proposal_l_count,
            )
        ),
    )


def _loss_from_parts(
    numerators: jax.Array, denominators: jax.Array, config: ReferenceConfig
) -> jax.Array:
    values = numerators / jnp.maximum(denominators, 1.0)
    return total_loss(
        {
            "h": values[0],
            "b": values[1],
            "l": values[2],
            "byte": values[3],
            "proposal": values[4] + values[5] + values[6],
        },
        config,
    )


def _paired_example_statistics(
    params: Any,
    prompt: jax.Array,
    target: jax.Array,
    root: jax.Array,
    global_example_id: jax.Array,
    lambda_op: float | jax.Array,
    *,
    config: ReferenceConfig = REFERENCE_SMALL_V0,
) -> dict[str, jax.Array]:
    """Evaluate one example in a root world with an explicit stable identity."""
    mask_prompt = jnp.ones_like(prompt, dtype=bool)
    mask_target = jnp.ones_like(target, dtype=bool)
    clean_ops = TorxOps.create(
        root,
        config=TorxOperatorConfig(operator_stochasticity=True, lambda_op=0.0),
        global_example_id=global_example_id,
    )
    noisy_ops = TorxOps.create(
        root,
        config=TorxOperatorConfig(operator_stochasticity=True, lambda_op=lambda_op),
        global_example_id=global_example_id,
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
    teacher = noisy["target"]["teacher"]
    correct = (jnp.argmax(noisy["byte_logits"], axis=-1) == teacher.slot_bytes) & teacher.slot_mask
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
    component_numerators, component_denominators = _component_numerators(noisy)
    return {
        "component_numerators": component_numerators,
        "component_denominators": component_denominators,
        "byte_correct": jnp.sum(correct),
        "byte_total": jnp.sum(teacher.slot_mask),
        "exact_correct": jnp.sum(jnp.all(correct | ~teacher.slot_mask, axis=(1, 2))),
        "exact_total": jnp.asarray(prompt.shape[0]),
        "nonfinite_count": nonfinite_count,
        "nonfinite_total": nonfinite_total,
        "signal_square_sums": signal_square_sums,
        "perturbation_square_sums": perturbation_square_sums,
        "stage_counts": stage_counts,
        "byte_logits": noisy["byte_logits"][0],
    }


def paired_chunk_statistics(
    params: Any,
    prompt: jax.Array,
    target: jax.Array,
    root: jax.Array,
    global_example_ids: jax.Array,
    lambda_op: float | jax.Array,
    *,
    config: ReferenceConfig = REFERENCE_SMALL_V0,
) -> dict[str, jax.Array]:
    """Evaluate a chunk as independently identified B=1 stochastic executions.

    The returned sums preserve each original objective component's denominator;
    callers must aggregate them rather than averaging per-example losses.
    """
    return jax.vmap(
        lambda sample_prompt, sample_target, sample_id: _paired_example_statistics(
            params,
            sample_prompt[None, :],
            sample_target[None, :],
            root,
            sample_id,
            lambda_op,
            config=config,
        )
    )(prompt, target, global_example_ids)


def aggregate_root_chunks(
    chunks: Iterable[dict[str, Any]],
    weights: Iterable[int],
    *,
    config: ReferenceConfig = REFERENCE_SMALL_V0,
) -> dict[str, Any]:
    """Combine fixed-size chunk statistics into one root-level observation."""
    chunk_list = list(chunks)
    weight_list = list(weights)
    total_weight = sum(weight_list)
    del weight_list, total_weight
    if not chunk_list:
        raise ValueError("cannot aggregate an empty evaluation")
    numerators = jnp.sum(
        jnp.stack([jnp.sum(chunk["component_numerators"], axis=0) for chunk in chunk_list]), axis=0
    )
    denominators = jnp.sum(
        jnp.stack([jnp.sum(chunk["component_denominators"], axis=0) for chunk in chunk_list]),
        axis=0,
    )
    byte_correct = jnp.sum(jnp.stack([jnp.sum(chunk["byte_correct"]) for chunk in chunk_list]))
    byte_total = jnp.sum(jnp.stack([jnp.sum(chunk["byte_total"]) for chunk in chunk_list]))
    exact_correct = jnp.sum(jnp.stack([jnp.sum(chunk["exact_correct"]) for chunk in chunk_list]))
    exact_total = jnp.sum(jnp.stack([jnp.sum(chunk["exact_total"]) for chunk in chunk_list]))
    signal_sum = jnp.sum(
        jnp.stack([jnp.sum(chunk["signal_square_sums"], axis=0) for chunk in chunk_list]), axis=0
    )
    perturbation_sum = jnp.sum(
        jnp.stack([jnp.sum(chunk["perturbation_square_sums"], axis=0) for chunk in chunk_list]),
        axis=0,
    )
    stage_count = jnp.sum(
        jnp.stack([jnp.sum(chunk["stage_counts"], axis=0) for chunk in chunk_list]), axis=0
    )
    nonfinite_count = jnp.sum(
        jnp.stack([jnp.sum(chunk["nonfinite_count"]) for chunk in chunk_list])
    )
    nonfinite_total = jnp.sum(
        jnp.stack([jnp.sum(chunk["nonfinite_total"]) for chunk in chunk_list])
    )
    return {
        "loss": _loss_from_parts(numerators, denominators, config),
        "byte_accuracy": byte_correct / jnp.maximum(byte_total, 1),
        "exact_sequence_accuracy": exact_correct / jnp.maximum(exact_total, 1),
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


def all_d3_runs_pass(results: Iterable[dict[str, Any]]) -> bool:
    """Return true only when every included authoritative D3 run clears both gates."""
    runs = list(results)
    return bool(runs) and all(
        bool(result["passed"])
        and float(result["lambda_zero"]["byte_accuracy"]["mean"]) >= 0.90
        and float(result["lambda_one"]["byte_accuracy"]["mean"]) >= 0.90
        for result in runs
    )
