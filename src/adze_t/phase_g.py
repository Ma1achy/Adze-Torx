"""Phase-G generated-structure losses, guarded replay, and optimizer step."""

from __future__ import annotations

from typing import Any, Literal, NamedTuple

import jax
import jax.numpy as jnp

from .backends.torx import TorxOperatorConfig, TorxOps
from .config import REFERENCE_SMALL_V0, ReferenceConfig
from .denoise import as_sanitized_model_target_analysis, make_sanitized_target_analysis
from .generated_structure import (
    CommitMasks,
    GeneratedRoutingResult,
    StructureEstimatorOwner,
    StructureKeyContext,
    commit_predicted_structure_core,
    generated_routing_core,
    score_corrected_structure_loss,
)
from .model import apply_clean_target_teacher, apply_model
from .objectives import adamw_step
from .phase_f_1 import make_initial_corruption
from .state import CommittedStructure, PredictedStructure
from .training import stochastic_model_update_mask


PHASE_G_TRAINING_PROTOCOL_V0 = "PHASE_G_TRAINING_PROTOCOL_V0"
STRUCTURE_TRUTH_LENGTH_MISMATCH = 12
STRUCTURE_TRUTH_INVALID = 13
TRAIN_NUMERICAL_INVALID = 14
GeneratedArm = Literal["teacher", "argmax", "sample"]


class GeneratedByteMetrics(NamedTuple):
    loss_per_example: jax.Array
    overlap_nll_per_example: jax.Array
    accuracy_per_example: jax.Array
    exact_per_example: jax.Array
    target_count: jax.Array
    predicted_count: jax.Array
    overlap_count: jax.Array
    missing_count: jax.Array
    extra_count: jax.Array


class GeneratedReplay(NamedTuple):
    executed: jax.Array
    error_code: jax.Array
    h_hat: jax.Array
    byte_logits: jax.Array
    b_logits: jax.Array
    l_logits: jax.Array


def _explicit_length(structure: CommittedStructure) -> jax.Array:
    if structure.length is None:
        raise ValueError("Phase G requires explicit committed lengths")
    return structure.length


def generated_structure_truth(
    slot_bytes: jax.Array,
    length: jax.Array,
    *,
    K: int,
) -> CommittedStructure:
    """Define the identifiable Phase-G task from observable codec content.

    Mandatory K-spaced cuts make every layout packable. Active carriers add a
    cut when their first emitted byte is in the lower half of the 1..32 task
    alphabet. The label is therefore a deterministic function of clean codec
    state, never of an independent random draw or hidden target metadata.
    """
    if slot_bytes.ndim != 3 or length.shape != slot_bytes.shape[:2]:
        raise ValueError("slot bytes and lengths must have [batch, carrier, slot] alignment")
    if K <= 0:
        raise ValueError("K must be positive")
    carrier = jnp.arange(length.shape[1])[None, :]
    mandatory = (carrier + 1) % K == 0
    active = length > 0
    content_cut = active & (slot_bytes[:, :, 0] >= 1) & (slot_bytes[:, :, 0] <= 16)
    boundary = (mandatory | content_cut).astype(jnp.int32).at[:, -1].set(1)
    return CommittedStructure(boundary, active, length.astype(jnp.int32))


def evaluate_committed_replay(
    params: Any,
    batch: dict[str, jax.Array],
    committed: CommittedStructure,
    operator_root: jax.Array,
    diffusion_root: jax.Array,
    occurrence: int,
    *,
    config: ReferenceConfig = REFERENCE_SMALL_V0,
) -> tuple[GeneratedByteMetrics, GeneratedReplay]:
    """Evaluate one committed structure with neutral observations and no mutation."""
    ops = TorxOps.create(
        operator_root,
        config=TorxOperatorConfig(operator_stochasticity=True, lambda_op=0.0),
        optimizer_step=occurrence,
    )
    clean = apply_clean_target_teacher(
        params, batch["target"], batch["target_mask"], config=config, ops=ops
    )
    clean = jax.tree.map(jax.lax.stop_gradient, clean)
    teacher = clean["target"]["teacher"]
    carrier_input, _ = make_initial_corruption(
        clean["target"]["h0"],
        batch["nu"],
        diffusion_root,
        batch["global_example_id"],
        optimizer_step=occurrence,
    )
    neutral = neutral_structure(carrier_input.shape[0], config)
    neutral_length = _explicit_length(neutral)
    committed_length = _explicit_length(committed)
    sanitized = make_sanitized_target_analysis(
        neutral.c_b,
        neutral_length,
        config=config,
        target_width=batch["target"].shape[1],
    )
    routing = generated_routing_core(
        committed,
        M_max=config.packing.M_max,
        K=config.packing.K,
        max_extent=config.carrier.L_max,
    )
    replay = guarded_generated_replay(
        params,
        batch,
        carrier_input,
        as_sanitized_model_target_analysis(sanitized),
        neutral,
        routing,
        ops,
        config=config,
    )
    metrics = generated_byte_metrics(
        replay.byte_logits, committed_length, teacher.slot_bytes, teacher.slot_mask
    )
    return metrics, replay


def neutral_structure(batch: int, config: ReferenceConfig) -> CommittedStructure:
    """Target-independent fixed K-block/all-active predictor structure."""
    carrier = jnp.arange(config.carrier.C)
    boundary = ((carrier + 1) % config.packing.K == 0).at[-1].set(True)
    boundary = jnp.broadcast_to(boundary, (batch, config.carrier.C)).astype(jnp.int32)
    length = jnp.full((batch, config.carrier.C), config.carrier.L_max, dtype=jnp.int32)
    return CommittedStructure(boundary, jnp.ones_like(length, dtype=bool), length)


def task_structure_commit_masks(batch: int, config: ReferenceConfig) -> CommitMasks:
    """Own only optional edges while preserving the mandatory packable scaffold."""
    carrier = jnp.arange(config.carrier.C)[None, :]
    mandatory = (carrier + 1) % config.packing.K == 0
    boundary_edge = jnp.broadcast_to(~mandatory, (batch, config.carrier.C)).at[:, -1].set(False)
    extent_site = jnp.ones((batch, config.carrier.C), dtype=bool)
    return CommitMasks(boundary_edge, extent_site)


def _compact_slots(values: jax.Array, mask: jax.Array) -> jax.Array:
    """Stable carrier/slot-major compaction into a static-width array."""
    batch = mask.shape[0]
    flat_mask = mask.reshape(batch, -1)
    flat = values.reshape((batch, flat_mask.shape[1], *values.shape[3:]))
    rank = jnp.cumsum(flat_mask, axis=1) - 1
    assignment = (
        jax.nn.one_hot(jnp.maximum(rank, 0), flat_mask.shape[1], dtype=values.dtype)
        * flat_mask[..., None]
    )
    if flat.ndim == 2:
        return jnp.einsum("bpn,bp->bn", assignment, flat)
    return jnp.einsum("bpn,bpv->bnv", assignment, flat)


def generated_byte_metrics(
    byte_logits: jax.Array,
    committed_length: jax.Array,
    target_bytes: jax.Array,
    target_mask: jax.Array,
) -> GeneratedByteMetrics:
    """Length-aware global compacted byte loss frozen for Phase G.

    Per example: ``(overlap CE + log(256)*(missing+extra))/max(n_target,1)``.
    Both-empty is exactly zero. Overlap NLL is reported separately.
    """
    slots = byte_logits.shape[2]
    predicted_mask = jnp.arange(slots)[None, None, :] < committed_length[..., None]
    compact_logits = _compact_slots(byte_logits, predicted_mask)
    compact_target = _compact_slots(target_bytes, target_mask).astype(jnp.int32)
    n_pred = jnp.sum(predicted_mask, axis=(1, 2))
    n_target = jnp.sum(target_mask, axis=(1, 2))
    overlap = jnp.minimum(n_pred, n_target)
    positions = jnp.arange(compact_target.shape[1])[None, :]
    overlap_mask = positions < overlap[:, None]
    log_probs = jax.nn.log_softmax(compact_logits, axis=-1)
    selected = jnp.take_along_axis(log_probs, compact_target[..., None], axis=-1)[..., 0]
    overlap_sum = -jnp.sum(jnp.where(overlap_mask, selected, 0.0), axis=1)
    missing = jnp.maximum(n_target - n_pred, 0)
    extra = jnp.maximum(n_pred - n_target, 0)
    loss = (overlap_sum + jnp.log(256.0) * (missing + extra)) / jnp.maximum(n_target, 1)
    loss = jnp.where((n_target == 0) & (n_pred == 0), 0.0, loss)
    correct = (jnp.argmax(compact_logits, axis=-1) == compact_target) & overlap_mask
    accuracy = jnp.sum(correct, axis=1) / jnp.maximum(jnp.maximum(n_target, n_pred), 1)
    accuracy = jnp.where((n_target == 0) & (n_pred == 0), 1.0, accuracy)
    exact = (n_target == n_pred) & jnp.all(correct | ~overlap_mask, axis=1)
    return GeneratedByteMetrics(
        loss,
        overlap_sum / jnp.maximum(overlap, 1),
        accuracy,
        exact,
        n_target,
        n_pred,
        overlap,
        missing,
        extra,
    )


def _cross_entropy_per_example(logits: jax.Array, labels: jax.Array) -> jax.Array:
    selected = jnp.take_along_axis(jax.nn.log_softmax(logits, axis=-1), labels[..., None], axis=-1)[
        ..., 0
    ]
    return -jnp.mean(selected, axis=1)


def guarded_generated_replay(
    params: Any,
    batch: dict[str, jax.Array],
    carrier_input: jax.Array,
    target_analysis: dict[str, Any],
    observed: CommittedStructure,
    routing: GeneratedRoutingResult,
    ops: Any,
    *,
    config: ReferenceConfig,
) -> GeneratedReplay:
    """Skip the complete replay model graph when generated routing is invalid."""
    batch_size = carrier_input.shape[0]
    observed_length = _explicit_length(observed)
    routed_length = _explicit_length(routing.structure)
    zero_h = jnp.zeros_like(carrier_input)
    zero_byte = jnp.zeros(
        (batch_size, config.carrier.C, config.carrier.L_max, config.model.byte_vocab),
        dtype=carrier_input.dtype,
    )
    zero_b = jnp.zeros((batch_size, config.carrier.C, 2), dtype=carrier_input.dtype)
    zero_l = jnp.zeros(
        (batch_size, config.carrier.C, config.carrier.L_max + 1), dtype=carrier_input.dtype
    )

    def execute(_):
        output = apply_model(
            params,
            batch["prompt"],
            batch["prompt_mask"],
            jnp.zeros_like(batch["target"]),
            jnp.zeros_like(batch["target_mask"]),
            config=config,
            ops=ops,
            target_analysis=target_analysis,
            carrier_h_input=carrier_input,
            noise_level=batch["nu"],
            actual_s_index=0,
            denoise_condition_index=0,
            committed_c_b=routing.structure.c_b,
            committed_activity=routing.structure.activity,
            committed_length=routed_length,
            observed_b=observed.c_b,
            observed_l=observed_length,
            routing_metadata=routing.metadata,
        )
        h_hat, b_logits, l_logits = output["prediction"]
        return GeneratedReplay(
            jnp.asarray(True), routing.error_code, h_hat, output["byte_logits"], b_logits, l_logits
        )

    return jax.lax.cond(
        routing.executed,
        execute,
        lambda _: GeneratedReplay(
            jnp.asarray(False), routing.error_code, zero_h, zero_byte, zero_b, zero_l
        ),
        operand=None,
    )


def generated_two_pass_objective(
    params: Any,
    batch: dict[str, jax.Array],
    operator_root: jax.Array,
    diffusion_root: jax.Array,
    structure_root: jax.Array,
    occurrence: jax.Array | int,
    *,
    arm: GeneratedArm,
    temperature: float | None = None,
    apply_score_correction: bool = True,
    inert_placeholder_value: int = 0,
    config: ReferenceConfig = REFERENCE_SMALL_V0,
) -> tuple[jax.Array, dict[str, jax.Array]]:
    """Predict once, commit at the boundary, then replay the same S1 input."""
    ops = TorxOps.create(
        operator_root,
        config=TorxOperatorConfig(operator_stochasticity=True, lambda_op=0.0),
        optimizer_step=occurrence,
    )
    clean = apply_clean_target_teacher(
        params, batch["target"], batch["target_mask"], config=config, ops=ops
    )
    clean = jax.tree.map(jax.lax.stop_gradient, clean)
    teacher = clean["target"]["teacher"]
    clean_h = clean["target"]["h0"]
    carrier_input, _ = make_initial_corruption(
        clean_h,
        batch["nu"],
        diffusion_root,
        batch["global_example_id"],
        optimizer_step=occurrence,
    )
    neutral = neutral_structure(carrier_input.shape[0], config)
    neutral_length = _explicit_length(neutral)
    sanitized = make_sanitized_target_analysis(
        neutral.c_b,
        neutral_length,
        config=config,
        target_width=batch["target"].shape[1],
        placeholder_value=inert_placeholder_value,
    )
    forward_target = as_sanitized_model_target_analysis(sanitized)
    predictor = apply_model(
        params,
        batch["prompt"],
        batch["prompt_mask"],
        sanitized.target_bytes_placeholder,
        sanitized.target_mask_placeholder,
        config=config,
        ops=ops,
        target_analysis=forward_target,
        carrier_h_input=carrier_input,
        noise_level=batch["nu"],
        actual_s_index=0,
        denoise_condition_index=0,
        committed_c_b=neutral.c_b,
        committed_activity=neutral.activity,
        committed_length=neutral_length,
        observed_b=neutral.c_b,
        observed_l=neutral_length,
    )
    h_predict, b_logits, l_logits = predictor["prediction"]
    predicted = PredictedStructure(b_logits, l_logits)
    raw_truth_boundary = batch["structure_boundary"]
    raw_truth_length = batch["structure_length"]
    truth_boundary = raw_truth_boundary.astype(jnp.int32)
    truth_length = raw_truth_length.astype(jnp.int32)
    truth_dtype_valid = jnp.issubdtype(raw_truth_boundary.dtype, jnp.integer) and jnp.issubdtype(
        raw_truth_length.dtype, jnp.integer
    )
    truth_domain_valid = (
        jnp.all((raw_truth_boundary == 0) | (raw_truth_boundary == 1), axis=1)
        & (raw_truth_boundary[:, -1] == 1)
        & jnp.all((raw_truth_length >= 0) & (raw_truth_length <= config.carrier.L_max), axis=1)
    )
    truth_matches_teacher = jnp.all(truth_length == teacher.length, axis=1)
    truth_error = jnp.where(
        truth_dtype_valid & truth_domain_valid,
        jnp.where(truth_matches_teacher, 0, STRUCTURE_TRUTH_LENGTH_MISMATCH),
        STRUCTURE_TRUTH_INVALID,
    )
    teacher_structure = CommittedStructure(truth_boundary, truth_length > 0, truth_length)
    key_context = StructureKeyContext(
        structure_root,
        occurrence,
        batch["global_example_id"],
        0,
        0,
        0,
    )
    if arm == "teacher":
        committed = teacher_structure
        commit = None
    else:
        commit = commit_predicted_structure_core(
            neutral,
            predicted,
            task_structure_commit_masks(neutral.c_b.shape[0], config),
            policy=arm,
            max_extent=config.carrier.L_max,
            key_context=key_context,
            temperature=temperature,
            estimator_owner=(
                StructureEstimatorOwner.LOCAL_SCORE_BRIDGE
                if arm == "sample"
                else StructureEstimatorOwner.NONE
            ),
            allow_reactivation=True,
        )
        committed = commit.committed
    commit_valid = jnp.ones_like(truth_error, dtype=bool) if commit is None else commit.valid
    commit_error = jnp.zeros_like(truth_error) if commit is None else commit.error_code
    upstream_valid = commit_valid & (truth_error == 0)
    upstream_error = jnp.where(commit_error != 0, commit_error, truth_error)
    routing = generated_routing_core(
        committed,
        M_max=config.packing.M_max,
        K=config.packing.K,
        max_extent=config.carrier.L_max,
        upstream_valid=upstream_valid,
        upstream_error=upstream_error,
    )
    replay = guarded_generated_replay(
        params, batch, carrier_input, forward_target, neutral, routing, ops, config=config
    )
    committed_length = _explicit_length(committed)

    def valid_loss(_):
        common = (
            _cross_entropy_per_example(b_logits[:, :-1], truth_boundary[:, :-1])
            + _cross_entropy_per_example(l_logits, truth_length)
            + 0.5 * jnp.mean((h_predict - clean_h) ** 2, axis=(1, 2))
        )
        byte = generated_byte_metrics(
            replay.byte_logits, committed_length, teacher.slot_bytes, teacher.slot_mask
        )
        descendant = (
            0.5 * jnp.mean((replay.h_hat - clean_h) ** 2, axis=(1, 2)) + byte.loss_per_example
        )
        if arm == "sample" and apply_score_correction:
            if commit is None:
                raise AssertionError("sample arm requires commit")
            descendant = score_corrected_structure_loss(descendant, commit)
        loss = jnp.mean(common + descendant)
        return loss, {
            "executed": jnp.asarray(True),
            "error_code_max": jnp.max(routing.error_code),
            "common_loss": jnp.mean(common),
            "descendant_loss": jnp.mean(descendant),
            "predictor_h_mse": jnp.mean((h_predict - clean_h) ** 2),
            "replay_h_mse": jnp.mean((replay.h_hat - clean_h) ** 2),
            "byte_loss": jnp.mean(byte.loss_per_example),
            "byte_overlap_nll": jnp.mean(byte.overlap_nll_per_example),
            "byte_accuracy": jnp.mean(byte.accuracy_per_example),
            "byte_exact": jnp.mean(byte.exact_per_example),
            "missing_bytes": jnp.sum(byte.missing_count),
            "extra_bytes": jnp.sum(byte.extra_count),
        }

    def invalid_loss(_):
        zero = jnp.asarray(0.0, carrier_input.dtype)
        return zero, {
            "executed": jnp.asarray(False),
            "error_code_max": jnp.max(routing.error_code),
            "common_loss": zero,
            "descendant_loss": zero,
            "predictor_h_mse": zero,
            "replay_h_mse": zero,
            "byte_loss": zero,
            "byte_overlap_nll": zero,
            "byte_accuracy": zero,
            "byte_exact": zero,
            "missing_bytes": zero,
            "extra_bytes": zero,
        }

    return jax.lax.cond(routing.executed, valid_loss, invalid_loss, operand=None)


def generated_train_step(
    params: Any,
    moments: tuple[Any, Any],
    step: jax.Array | int,
    batch: dict[str, jax.Array],
    operator_root: jax.Array,
    diffusion_root: jax.Array,
    structure_root: jax.Array,
    *,
    arm: GeneratedArm,
    temperature: float | None = None,
    config: ReferenceConfig = REFERENCE_SMALL_V0,
) -> tuple[Any, tuple[Any, Any], jax.Array, dict[str, Any]]:
    """One fail-closed Phase-G optimizer update."""

    def objective(value):
        return generated_two_pass_objective(
            value,
            batch,
            operator_root,
            diffusion_root,
            structure_root,
            step,
            arm=arm,
            temperature=temperature,
            apply_score_correction=True,
            config=config,
        )

    (loss, metrics), grads = jax.value_and_grad(objective, has_aux=True)(params)
    update_mask = stochastic_model_update_mask(params)
    gradients_finite = jnp.all(
        jnp.stack([jnp.all(jnp.isfinite(leaf)) for leaf in jax.tree.leaves(grads)])
    )
    numerical_finite = jnp.isfinite(loss) & gradients_finite
    executed = metrics["executed"] & numerical_finite

    params, moments, next_step, raw_norm = fail_closed_adamw_step(
        params, grads, moments, step, executed, update_mask
    )
    return (
        params,
        moments,
        next_step,
        {
            **metrics,
            "executed": executed,
            "error_code_max": jnp.where(
                metrics["executed"] & ~numerical_finite,
                TRAIN_NUMERICAL_INVALID,
                metrics["error_code_max"],
            ),
            "loss": loss,
            "gradients_finite": gradients_finite,
            "grad_raw_norm": raw_norm,
        },
    )


def fail_closed_adamw_step(
    params: Any,
    grads: Any,
    moments: tuple[Any, Any],
    step: jax.Array | int,
    executed: jax.Array,
    update_mask: Any,
) -> tuple[Any, tuple[Any, Any], jax.Array, jax.Array]:
    """Apply the frozen optimizer only when the whole generated batch executed."""

    def update(_):
        updated, updated_moments, raw_norm = adamw_step(
            params,
            grads,
            moments,
            jnp.asarray(step) + 1,
            learning_rate=3.0e-4,
            weight_decay=0.01,
            clip_norm=1.0,
            update_mask=update_mask,
        )
        return updated, updated_moments, jnp.asarray(step) + 1, raw_norm

    dtype = jax.tree.leaves(params)[0].dtype
    return jax.lax.cond(
        executed,
        update,
        lambda _: (params, moments, jnp.asarray(step), jnp.asarray(0.0, dtype)),
        operand=None,
    )
