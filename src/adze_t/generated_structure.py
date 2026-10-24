"""Phase G generated-structure state transitions and diagnostics.

Predictions and committed routing are deliberately separate.  Only an explicit
commit boundary may change packing, activity, or emission semantics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import jax
import jax.numpy as jnp

from .packing import PackMetadata, build_pack_metadata_core
from .state import CommittedStructure, PredictedStructure
from .train.score_bridge import score_corrected_loss


PHASE_G_GENERATED_STRUCTURE_INTERFACE_V0 = "PHASE_G_GENERATED_STRUCTURE_INTERFACE_V0"
CommitPolicy = Literal["prediction_only", "argmax", "sample"]


@dataclass(frozen=True)
class StructureCommit:
    """One controlled transition from predictions to committed routing."""

    previous: CommittedStructure
    committed: CommittedStructure
    boundary_sample: jax.Array
    length_sample: jax.Array
    boundary_log_prob: jax.Array
    length_log_prob: jax.Array
    descendant_log_prob: jax.Array
    changed: jax.Array
    reactivated: jax.Array
    deactivated: jax.Array
    policy: str


@dataclass(frozen=True)
class GeneratedRouting:
    """Fixed-to-generated interface consumed by hard pack and later phases."""

    structure: CommittedStructure
    metadata: PackMetadata


def _selected_log_probability(logits: jax.Array, values: jax.Array) -> jax.Array:
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    return jnp.take_along_axis(log_probs, values[..., None], axis=-1)[..., 0]


def _validate_transition_inputs(
    previous: CommittedStructure,
    predicted: PredictedStructure,
    commit_mask: jax.Array,
) -> None:
    shape = previous.c_b.shape
    if previous.activity.shape != shape or previous.effective_length().shape != shape:
        raise ValueError("previous committed structure fields must share [batch, carrier]")
    if predicted.b_logits.shape != (*shape, 2):
        raise ValueError("boundary logits must have shape [batch, carrier, 2]")
    if predicted.l_logits.shape[:2] != shape or predicted.l_logits.ndim != 3:
        raise ValueError("extent logits must have shape [batch, carrier, classes]")
    if predicted.l_logits.shape[-1] < 2:
        raise ValueError("extent logits must include zero and at least one emitting class")
    if commit_mask.shape != shape:
        raise ValueError("commit_mask must have shape [batch, carrier]")


def commit_predicted_structure(
    previous: CommittedStructure,
    predicted: PredictedStructure,
    *,
    policy: CommitPolicy,
    commit_mask: jax.Array | None = None,
    key: jax.Array | None = None,
) -> StructureCommit:
    """Apply a controlled structure commit without changing carrier identity.

    ``prediction_only`` leaves routing bitwise unchanged. ``argmax`` is a hard,
    stopped deterministic commit. ``sample`` uses explicit JAX keys and returns
    exactly the sampled-occurrence log probability required by the frozen local
    score bridge. The terminal boundary is architectural and never sampled.
    """
    shape = previous.c_b.shape
    selected = (
        jnp.ones(shape, dtype=bool) if commit_mask is None else jnp.asarray(commit_mask, bool)
    )
    _validate_transition_inputs(previous, predicted, selected)
    previous_length = previous.effective_length().astype(jnp.int32)

    if policy == "prediction_only":
        boundary = previous.c_b
        length = previous_length
        boundary_lp = jnp.zeros(shape, dtype=predicted.b_logits.dtype)
        length_lp = jnp.zeros(shape, dtype=predicted.l_logits.dtype)
    elif policy == "argmax":
        boundary = jax.lax.stop_gradient(jnp.argmax(predicted.b_logits, axis=-1)).astype(jnp.int32)
        length = jax.lax.stop_gradient(jnp.argmax(predicted.l_logits, axis=-1)).astype(jnp.int32)
        boundary_lp = jnp.zeros(shape, dtype=predicted.b_logits.dtype)
        length_lp = jnp.zeros(shape, dtype=predicted.l_logits.dtype)
    elif policy == "sample":
        if key is None:
            raise ValueError("sample commit requires an explicit JAX key")
        boundary_key, length_key = jax.random.split(key)
        boundary = jax.random.categorical(boundary_key, predicted.b_logits, axis=-1).astype(
            jnp.int32
        )
        length = jax.random.categorical(length_key, predicted.l_logits, axis=-1).astype(jnp.int32)
        boundary_lp = _selected_log_probability(predicted.b_logits, boundary)
        length_lp = _selected_log_probability(predicted.l_logits, length)
    else:
        raise ValueError(f"unknown structure commit policy: {policy}")

    boundary = jnp.where(selected, boundary, previous.c_b).astype(jnp.int32)
    length = jnp.where(selected, length, previous_length).astype(jnp.int32)
    # The final cut is a sentinel, not a learned/sampled routing occurrence.
    boundary = boundary.at[:, -1].set(1)
    boundary_selected = selected.at[:, -1].set(False)
    if policy != "sample":
        boundary_lp = jnp.zeros_like(boundary_lp)
        length_lp = jnp.zeros_like(length_lp)
    boundary_lp = jnp.where(boundary_selected, boundary_lp, 0)
    length_lp = jnp.where(selected, length_lp, 0)

    activity = length > 0
    committed = CommittedStructure(c_b=boundary, activity=activity, length=length)
    changed = (boundary != previous.c_b) | (length != previous_length)
    previous_active = previous_length > 0
    reactivated = (~previous_active) & activity
    deactivated = previous_active & (~activity)
    descendant_log_prob = jnp.sum(boundary_lp + length_lp, axis=1)
    return StructureCommit(
        previous=previous,
        committed=committed,
        boundary_sample=boundary,
        length_sample=length,
        boundary_log_prob=boundary_lp,
        length_log_prob=length_lp,
        descendant_log_prob=descendant_log_prob,
        changed=changed,
        reactivated=reactivated,
        deactivated=deactivated,
        policy=policy,
    )


def generated_routing(
    structure: CommittedStructure,
    *,
    M_max: int,
    K: int,
) -> GeneratedRouting:
    """Build the common fixed/generated routing interface at static capacity."""
    length = structure.effective_length()
    activity = length > 0
    normalized = CommittedStructure(
        c_b=structure.c_b.astype(jnp.int32),
        activity=activity,
        length=length.astype(jnp.int32),
    )
    metadata = build_pack_metadata_core(normalized.c_b, activity, M_max=M_max, K=K)
    return GeneratedRouting(normalized, metadata)


def score_corrected_structure_loss(
    descendant_loss: jax.Array,
    commit: StructureCommit,
) -> jax.Array:
    """Apply the frozen bridge only to sampled structure's descendant loss."""
    if commit.policy != "sample":
        raise ValueError("score correction is only valid for sampled downstream structure")
    return score_corrected_loss(descendant_loss, commit.descendant_log_prob)


def structure_churn(
    previous: CommittedStructure, current: CommittedStructure
) -> dict[str, jax.Array]:
    """Report topology/activity churn without averaging away event types."""
    old_length = previous.effective_length()
    new_length = current.effective_length()
    old_active = old_length > 0
    new_active = new_length > 0
    nonterminal = jnp.ones_like(previous.c_b, dtype=bool).at[:, -1].set(False)
    return {
        "boundary_change_rate": jnp.sum((previous.c_b != current.c_b) & nonterminal)
        / jnp.maximum(jnp.sum(nonterminal), 1),
        "extent_change_rate": jnp.mean(old_length != new_length),
        "reactivation_rate": jnp.mean((~old_active) & new_active),
        "deactivation_rate": jnp.mean(old_active & (~new_active)),
        "block_count_delta_mean": jnp.mean(
            jnp.sum(current.c_b, axis=1) - jnp.sum(previous.c_b, axis=1)
        ),
    }


def categorical_calibration(
    logits: jax.Array,
    labels: jax.Array,
    mask: jax.Array,
    *,
    bins: int = 10,
) -> dict[str, jax.Array]:
    """Accuracy, NLL, Brier score, and fixed-bin ECE for structural heads."""
    if bins <= 0:
        raise ValueError("bins must be positive")
    probs = jax.nn.softmax(logits, axis=-1)
    predicted = jnp.argmax(probs, axis=-1)
    confidence = jnp.max(probs, axis=-1)
    correct = predicted == labels
    weights = mask.astype(logits.dtype)
    count = jnp.maximum(jnp.sum(weights), 1)
    selected = jnp.take_along_axis(jnp.log(jnp.maximum(probs, 1.0e-30)), labels[..., None], -1)[
        ..., 0
    ]
    one_hot = jax.nn.one_hot(labels, logits.shape[-1], dtype=logits.dtype)
    ece = jnp.asarray(0.0, dtype=logits.dtype)
    for index in range(bins):
        low = index / bins
        high = (index + 1) / bins
        in_bin = (confidence >= low) & (
            confidence < high if index + 1 < bins else confidence <= high
        )
        in_bin = in_bin & mask
        bin_count = jnp.sum(in_bin)
        bin_accuracy = jnp.sum(correct * in_bin) / jnp.maximum(bin_count, 1)
        bin_confidence = jnp.sum(confidence * in_bin) / jnp.maximum(bin_count, 1)
        ece = ece + (bin_count / count) * jnp.abs(bin_accuracy - bin_confidence)
    return {
        "count": jnp.sum(weights),
        "accuracy": jnp.sum(correct * weights) / count,
        "nll": -jnp.sum(selected * weights) / count,
        "brier": jnp.sum(jnp.sum((probs - one_hot) ** 2, axis=-1) * weights) / count,
        "ece": ece,
    }


def structure_calibration(
    predicted: PredictedStructure,
    boundary_labels: jax.Array,
    length_labels: jax.Array,
    *,
    bins: int = 10,
) -> dict[str, dict[str, jax.Array]]:
    """Calibrate boundary decisions (excluding sentinel) and all extents."""
    boundary_mask = jnp.ones_like(boundary_labels, dtype=bool).at[:, -1].set(False)
    length_mask = jnp.ones_like(length_labels, dtype=bool)
    return {
        "boundary": categorical_calibration(
            predicted.b_logits, boundary_labels, boundary_mask, bins=bins
        ),
        "extent": categorical_calibration(
            predicted.l_logits, length_labels, length_mask, bins=bins
        ),
    }
