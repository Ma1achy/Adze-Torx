"""Outer-R draft/select/reset/global-refine orchestration.

This module owns carrier-content orchestration only. Generated structural
commits and topology-safe region closure are explicit protocols supplied by a
future validated Phase-G adapter; no Phase-G policy is duplicated here.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Literal, NamedTuple, Protocol

from jax import Array
import jax.numpy as jnp

from .state import CarrierState, CommittedStructure, ObservedStructure, PredictedStructure
from .structure_unknown import StructureUnknownCodes, structure_unknown_codes


PHASE_H_REFINEMENT_INTERFACE_V1 = "PHASE_H_REFINEMENT_INTERFACE_V1"
RefinementMode = Literal["draft", "refine"]
SelectionBudgetScope = Literal["seed", "expanded"]


@dataclass(frozen=True)
class SelectionPolicy:
    """Deterministic selection with an explicit seed or expanded-region cap."""

    threshold: float
    max_sites: int | None
    budget_scope: SelectionBudgetScope
    halo: int = 0
    allow_inactive_reactivation: bool = False


class RefinementContext(NamedTuple):
    """Explicit roots and identities inherited by every R occurrence."""

    execution_kind: Array
    execution_index: Array
    global_example_ids: Array
    diffusion_root: Array
    operator_root: Array
    structure_root: Array


class RefinementOccurrence(NamedTuple):
    refinement_index: Array
    commit_occurrence: Array


class DenoiseEvidence(NamedTuple):
    """Occurrence evidence returned by the future full-S F adapter.

    Operator-coordinate columns are actual-S, Q, physical layer, and a public
    scope ID. Keys are the exact derived JAX keys.
    """

    actual_s_indices: Array
    denoise_condition_indices: Array
    diffusion_keys: Array
    operator_coordinates: Array
    operator_keys: Array
    model_applications: Array


class DenoiseResult(NamedTuple):
    state: CarrierState
    evidence: DenoiseEvidence


class RegionSelection(NamedTuple):
    scores: Array
    eligible: Array
    seed: Array
    closed: Array
    budget_scope_code: Array
    budget_limit: Array
    seed_count: Array
    closed_count: Array
    realized_error_label: Array
    label_valid: Array


class StructureCommitMasks(NamedTuple):
    """Topology-aware masks produced by the future validated G closure API."""

    boundary_edge: Array
    extent_site: Array


class StructuralCommitResult(Protocol):
    """Structural-only view required from a future validated G adapter."""

    @property
    def committed(self) -> CommittedStructure: ...

    @property
    def routing_valid(self) -> Array: ...

    @property
    def routing_overflow(self) -> Array: ...

    @property
    def changed(self) -> Array: ...

    @property
    def reactivated(self) -> Array: ...

    @property
    def deactivated(self) -> Array: ...

    @property
    def boundary_log_prob(self) -> Array: ...

    @property
    def length_log_prob(self) -> Array: ...

    @property
    def policy_code(self) -> Array: ...


DenoiseFn = Callable[
    [CarrierState, RefinementMode, RefinementContext, RefinementOccurrence], DenoiseResult
]


SelectorFn = Callable[[CarrierState, RefinementContext, RefinementOccurrence], RegionSelection]
RegionClosureFn = Callable[
    [CommittedStructure, RegionSelection, RefinementContext, RefinementOccurrence],
    StructureCommitMasks,
]
StructureKeyFn = Callable[[RefinementContext, RefinementOccurrence], Array]


StructuralCommitFn = Callable[
    [
        CommittedStructure,
        PredictedStructure,
        StructureCommitMasks,
        Array,
        RefinementContext,
        RefinementOccurrence,
    ],
    StructuralCommitResult,
]


class RefinementTrace(NamedTuple):
    refinement_index: Array
    selector_scores: Array
    selector_eligible: Array
    selector_labels: Array
    selector_label_valid: Array
    seed_selected: Array
    selected_closed: Array
    selection_budget_scope: Array
    selection_budget_limit: Array
    seed_count: Array
    closed_count: Array
    erased: Array
    before_h: Array
    reset_h: Array
    regenerated_h: Array
    committed_h: Array
    observed_b_before: Array
    observed_b_reset: Array
    observed_b_regenerated: Array
    observed_b_committed: Array
    observed_l_before: Array
    observed_l_reset: Array
    observed_l_regenerated: Array
    observed_l_committed: Array
    predicted_b_before: Array
    predicted_b_reset: Array
    predicted_b_regenerated: Array
    predicted_b_committed: Array
    predicted_l_before: Array
    predicted_l_reset: Array
    predicted_l_regenerated: Array
    predicted_l_committed: Array
    predicted_valid_before: Array
    predicted_valid_reset: Array
    predicted_valid_regenerated: Array
    predicted_valid_committed: Array
    boundary_before: Array
    boundary_after: Array
    length_before: Array
    length_after: Array
    activity_before: Array
    activity_after: Array
    boundary_commit_mask: Array
    extent_commit_mask: Array
    structure_key: Array
    routing_valid: Array
    routing_overflow: Array
    structure_changed: Array
    reactivated: Array
    deactivated: Array
    boundary_log_prob: Array
    length_log_prob: Array
    commit_policy_code: Array
    actual_s_indices: Array
    denoise_condition_indices: Array
    diffusion_keys: Array
    operator_coordinates: Array
    operator_keys: Array
    model_applications: Array
    selected_change_rms: Array
    selected_change_support: Array
    selected_change_valid: Array
    collateral_change_rms: Array
    collateral_change_support: Array
    collateral_change_valid: Array
    unselected_preserved: Array


class RefinementResult(NamedTuple):
    draft: CarrierState
    final: CarrierState
    context: RefinementContext
    draft_evidence: DenoiseEvidence
    trace: RefinementTrace


def _validate_site_mask(mask: Array, state: CarrierState, name: str = "site mask") -> Array:
    mask = jnp.asarray(mask)
    if mask.dtype != jnp.bool_ or mask.shape != state.h.shape[:2]:
        raise ValueError(f"{name} must be boolean with shape {state.h.shape[:2]}")
    return mask


def _site_where(mask: Array, selected: Array, unselected: Array) -> Array:
    expanded = mask.reshape((*mask.shape, *((1,) * (selected.ndim - mask.ndim))))
    return jnp.where(expanded, selected, unselected)


def _prediction_parts(
    predicted: PredictedStructure | None,
    *,
    batch: int,
    capacity: int,
    max_extent: int,
    dtype: jnp.dtype,
) -> tuple[Array, Array, Array]:
    if predicted is None:
        return (
            jnp.zeros((batch, capacity, 2), dtype=dtype),
            jnp.zeros((batch, capacity, max_extent + 1), dtype=dtype),
            jnp.zeros((batch, capacity), dtype=bool),
        )
    valid = (
        jnp.ones((batch, capacity), dtype=bool)
        if predicted.valid is None
        else jnp.asarray(predicted.valid, dtype=bool)
    )
    return predicted.b_logits, predicted.l_logits, valid


def validate_refinement_state(
    state: CarrierState,
    *,
    max_extent: int,
    unknown_codes: StructureUnknownCodes,
    allow_unknown: bool,
) -> None:
    """Validate every H-owned state boundary with explicit extents."""
    if state.h.ndim != 3 or not bool(jnp.all(jnp.isfinite(state.h))):
        raise ValueError("carrier h must be finite with shape [batch, carrier, width]")
    batch, capacity = state.h.shape[:2]
    if state.committed.length is None:
        raise ValueError("Phase-H boundaries require explicit committed lengths")
    for name, value in (
        ("observed.s_b", state.observed.s_b),
        ("observed.s_l", state.observed.s_l),
        ("committed.c_b", state.committed.c_b),
        ("committed.activity", state.committed.activity),
        ("committed.length", state.committed.length),
    ):
        if value.shape != (batch, capacity):
            raise ValueError(f"{name} must have shape {(batch, capacity)}")
    boundary_valid = (state.observed.s_b == 0) | (state.observed.s_b == 1)
    extent_valid = (state.observed.s_l >= 0) & (state.observed.s_l <= max_extent)
    if allow_unknown:
        boundary_valid |= state.observed.s_b == unknown_codes.boundary
        extent_valid |= state.observed.s_l == unknown_codes.extent
    if not bool(jnp.all(boundary_valid)) or not bool(jnp.all(extent_valid)):
        raise ValueError("observed structure contains an invalid categorical code")
    boundary = state.committed.c_b
    activity = state.committed.activity
    length = state.committed.length
    if not bool(jnp.all((boundary == 0) | (boundary == 1))):
        raise ValueError("committed boundaries must be binary")
    if not bool(jnp.all(boundary[:, -1] == 1)):
        raise ValueError("committed structure requires a terminal boundary")
    if not bool(jnp.all((activity == 0) | (activity == 1))):
        raise ValueError("committed activity must be binary")
    if not bool(jnp.all((length >= 0) & (length <= max_extent))):
        raise ValueError("committed lengths must lie in [0, max_extent]")
    if not bool(jnp.all((length > 0) == (activity > 0))):
        raise ValueError("committed activity must exactly equal length > 0")
    if state.predicted is not None:
        predicted = state.predicted
        if predicted.b_logits.shape != (batch, capacity, 2):
            raise ValueError("boundary predictions must have shape [batch, carrier, 2]")
        if predicted.l_logits.shape != (batch, capacity, max_extent + 1):
            raise ValueError("extent predictions have the wrong class count")
        if not bool(
            jnp.all(jnp.isfinite(predicted.b_logits)) & jnp.all(jnp.isfinite(predicted.l_logits))
        ):
            raise ValueError("structural prediction logits must be finite")
        if predicted.valid is not None and predicted.valid.shape != (batch, capacity):
            raise ValueError("predicted.valid must have shape [batch, carrier]")


def validate_refinement_context(context: RefinementContext, *, batch: int) -> None:
    if context.global_example_ids.shape != (batch,):
        raise ValueError("global_example_ids must have shape [batch]")
    for name in ("diffusion_root", "operator_root", "structure_root"):
        value = getattr(context, name)
        if value.shape != (2,):
            raise ValueError(f"{name} must be one legacy JAX PRNG key with shape [2]")
    for name in ("execution_kind", "execution_index"):
        if jnp.asarray(getattr(context, name)).ndim != 0:
            raise ValueError(f"{name} must be scalar")
    if int(context.execution_kind) not in (0, 1):
        raise ValueError("execution_kind must be 0 (evaluation) or 1 (training)")
    if int(context.execution_index) < 0:
        raise ValueError("execution_index must be non-negative")


def select_regions(
    score: Array,
    activity: Array,
    policy: SelectionPolicy,
    *,
    realized_error_label: Array | None = None,
) -> RegionSelection:
    """Select deterministic seeds, then close with a finite/eligibility-safe halo."""
    score = jnp.asarray(score)
    activity = jnp.asarray(activity, dtype=bool)
    if score.ndim != 2 or activity.shape != score.shape:
        raise ValueError("score and activity must have identical [batch, carrier] shapes")
    if policy.halo < 0:
        raise ValueError("halo must be non-negative")
    if policy.budget_scope not in ("seed", "expanded"):
        raise ValueError("budget_scope must be 'seed' or 'expanded'")
    if policy.max_sites is not None and not 0 <= policy.max_sites <= score.shape[1]:
        raise ValueError("max_sites must lie in [0, carrier capacity]")
    eligible = jnp.isfinite(score)
    if not policy.allow_inactive_reactivation:
        eligible &= activity
    seed = eligible & (score >= policy.threshold)
    if policy.max_sites is not None:
        ranked_score = jnp.where(seed, score, -jnp.inf)
        order = jnp.argsort(-ranked_score, axis=1, stable=True)
        rank = (
            jnp.empty_like(order)
            .at[jnp.arange(score.shape[0])[:, None], order]
            .set(jnp.arange(score.shape[1])[None, :])
        )
        seed &= rank < policy.max_sites
    closed = seed
    for _ in range(policy.halo):
        closed = closed | jnp.pad(closed[:, :-1], ((0, 0), (1, 0)))
        closed = closed | jnp.pad(closed[:, 1:], ((0, 0), (0, 1)))
    closed &= eligible
    if policy.budget_scope == "expanded" and policy.max_sites is not None:
        halo_only = closed & ~seed
        ranked_halo_score = jnp.where(halo_only, score, -jnp.inf)
        halo_order = jnp.argsort(-ranked_halo_score, axis=1, stable=True)
        halo_rank = (
            jnp.empty_like(halo_order)
            .at[jnp.arange(score.shape[0])[:, None], halo_order]
            .set(jnp.arange(score.shape[1])[None, :])
        )
        remaining = jnp.maximum(policy.max_sites - jnp.sum(seed, axis=1), 0)
        closed = seed | (halo_only & (halo_rank < remaining[:, None]))
    scope_code = jnp.asarray(0 if policy.budget_scope == "seed" else 1, dtype=jnp.int32)
    budget_limit = jnp.asarray(
        -1 if policy.max_sites is None else policy.max_sites, dtype=jnp.int32
    )
    if realized_error_label is None:
        labels = jnp.full(score.shape, jnp.nan, dtype=score.dtype)
        label_valid = jnp.zeros(score.shape, dtype=bool)
    else:
        labels = jnp.asarray(realized_error_label, dtype=score.dtype)
        if labels.shape != score.shape:
            raise ValueError("realized_error_label must match score")
        label_valid = jnp.isfinite(labels)
    return RegionSelection(
        score,
        eligible,
        seed,
        closed,
        scope_code,
        budget_limit,
        jnp.sum(seed, axis=1),
        jnp.sum(closed, axis=1),
        labels,
        label_valid,
    )


def reset_selected_carriers(
    state: CarrierState,
    selected: Array,
    reset_h: Array | float,
    *,
    max_extent: int,
    unknown_codes: StructureUnknownCodes | None = None,
) -> CarrierState:
    """Erase selected content/observations without changing committed routing."""
    selected = _validate_site_mask(selected, state)
    codes = unknown_codes or structure_unknown_codes(max_extent)
    reset_value = jnp.asarray(reset_h, dtype=state.h.dtype)
    try:
        reset_value = jnp.broadcast_to(reset_value, state.h.shape)
    except ValueError as exc:
        raise ValueError("reset_h must broadcast to carrier h") from exc
    observed = ObservedStructure(
        s_b=jnp.where(
            selected, jnp.full_like(state.observed.s_b, codes.boundary), state.observed.s_b
        ),
        s_l=jnp.where(
            selected, jnp.full_like(state.observed.s_l, codes.extent), state.observed.s_l
        ),
    )
    predicted = state.predicted
    if predicted is not None:
        valid = jnp.ones(selected.shape, dtype=bool) if predicted.valid is None else predicted.valid
        predicted = PredictedStructure(
            b_logits=_site_where(selected, jnp.zeros_like(predicted.b_logits), predicted.b_logits),
            l_logits=_site_where(selected, jnp.zeros_like(predicted.l_logits), predicted.l_logits),
            valid=jnp.where(selected, False, valid),
        )
    return CarrierState(
        h=_site_where(selected, reset_value, state.h),
        observed=observed,
        committed=state.committed,
        predicted=predicted,
    )


def preserve_unselected_state(
    before: CarrierState,
    candidate: CarrierState,
    selected: Array,
    *,
    max_extent: int,
) -> CarrierState:
    """Pin every unselected H-owned field while preserving prediction emergence."""
    selected = _validate_site_mask(selected, before)
    if candidate.h.shape != before.h.shape:
        raise ValueError("candidate carrier shape must match the pre-refinement state")
    batch, capacity = selected.shape
    before_b, before_l, before_valid = _prediction_parts(
        before.predicted,
        batch=batch,
        capacity=capacity,
        max_extent=max_extent,
        dtype=before.h.dtype,
    )
    candidate_b, candidate_l, candidate_valid = _prediction_parts(
        candidate.predicted,
        batch=batch,
        capacity=capacity,
        max_extent=max_extent,
        dtype=before.h.dtype,
    )
    merged_valid = jnp.where(selected, candidate_valid, before_valid)
    predicted = (
        PredictedStructure(
            b_logits=_site_where(selected, candidate_b, before_b),
            l_logits=_site_where(selected, candidate_l, before_l),
            valid=merged_valid,
        )
        if bool(jnp.any(merged_valid))
        else None
    )
    return CarrierState(
        h=_site_where(selected, candidate.h, before.h),
        observed=ObservedStructure(
            s_b=jnp.where(selected, candidate.observed.s_b, before.observed.s_b),
            s_l=jnp.where(selected, candidate.observed.s_l, before.observed.s_l),
        ),
        committed=before.committed,
        predicted=predicted,
    )


def _masked_rms(delta: Array, mask: Array) -> tuple[Array, Array, Array]:
    per_site = jnp.mean(jnp.square(delta), axis=-1)
    support = jnp.sum(mask, axis=1)
    valid = support > 0
    value = jnp.sum(jnp.where(mask, per_site, 0.0), axis=1) / jnp.maximum(support, 1)
    return jnp.where(valid, jnp.sqrt(value), jnp.nan), support, valid


def _validate_selection(selection: RegionSelection, state: CarrierState) -> RegionSelection:
    for name in ("eligible", "seed", "closed", "label_valid"):
        _validate_site_mask(getattr(selection, name), state, name)
    if (
        selection.scores.shape != state.h.shape[:2]
        or selection.realized_error_label.shape != state.h.shape[:2]
    ):
        raise ValueError("selector scores and labels must match [batch, carrier]")
    if not bool(jnp.all(selection.seed <= selection.closed)):
        raise ValueError("selection closure must contain every seed")
    if not bool(jnp.all(selection.closed <= selection.eligible)):
        raise ValueError("selection closure may contain only eligible sites")
    if jnp.asarray(selection.budget_scope_code).ndim != 0 or int(
        selection.budget_scope_code
    ) not in (0, 1):
        raise ValueError("selection budget scope code must be scalar seed=0 or expanded=1")
    if jnp.asarray(selection.budget_limit).ndim != 0 or int(selection.budget_limit) < -1:
        raise ValueError("selection budget limit must be scalar and use -1 for unlimited")
    expected_seed_count = jnp.sum(selection.seed, axis=1)
    expected_closed_count = jnp.sum(selection.closed, axis=1)
    if selection.seed_count.shape != expected_seed_count.shape or not bool(
        jnp.array_equal(selection.seed_count, expected_seed_count)
    ):
        raise ValueError("selection seed count disagrees with seed mask")
    if selection.closed_count.shape != expected_closed_count.shape or not bool(
        jnp.array_equal(selection.closed_count, expected_closed_count)
    ):
        raise ValueError("selection closed count disagrees with closed mask")
    limit = int(selection.budget_limit)
    if limit >= 0:
        measured = (
            selection.seed_count
            if int(selection.budget_scope_code) == 0
            else selection.closed_count
        )
        if not bool(jnp.all(measured <= limit)):
            raise ValueError("selection exceeds its declared budget scope")
    return selection


def _validate_evidence(evidence: DenoiseEvidence) -> None:
    if evidence.actual_s_indices.ndim != 1:
        raise ValueError("actual_s_indices must have shape [S]")
    steps = evidence.actual_s_indices.shape[0]
    if steps < 1 or not bool(
        jnp.array_equal(evidence.actual_s_indices, jnp.arange(steps, dtype=jnp.int32))
    ):
        raise ValueError("actual_s_indices must be the complete zero-based S trajectory")
    if evidence.denoise_condition_indices.shape != evidence.actual_s_indices.shape:
        raise ValueError("denoise conditioning coordinates must match actual-S shape")
    if evidence.diffusion_keys.ndim != 2 or evidence.diffusion_keys.shape[-1] != 2:
        raise ValueError("diffusion_keys must have shape [occurrences, 2]")
    if evidence.diffusion_keys.shape[0] != max(steps - 1, 0):
        raise ValueError("diffusion_keys must contain one key per S transition")
    if evidence.operator_coordinates.ndim != 2 or evidence.operator_coordinates.shape[-1] != 4:
        raise ValueError("operator_coordinates must have shape [occurrences, 4]")
    if evidence.operator_keys.shape != (evidence.operator_coordinates.shape[0], 2):
        raise ValueError("operator_keys must correspond one-to-one with coordinates")
    if jnp.asarray(evidence.model_applications).ndim != 0:
        raise ValueError("model_applications must be scalar")
    if int(evidence.model_applications) != steps:
        raise ValueError("model_applications must equal the executed S depth")
    if evidence.operator_coordinates.shape[0] and not bool(
        jnp.all(
            (evidence.operator_coordinates[:, 0] >= 0)
            & (evidence.operator_coordinates[:, 0] < steps)
        )
    ):
        raise ValueError("operator actual-S coordinates must belong to the executed trajectory")


def _prediction_snapshot(state: CarrierState, max_extent: int) -> tuple[Array, Array, Array]:
    batch, capacity = state.h.shape[:2]
    return _prediction_parts(
        state.predicted, batch=batch, capacity=capacity, max_extent=max_extent, dtype=state.h.dtype
    )


def _unselected_preserved(
    before: CarrierState, after: CarrierState, selected: Array, max_extent: int
) -> Array:
    before_b, before_l, before_valid = _prediction_snapshot(before, max_extent)
    after_b, after_l, after_valid = _prediction_snapshot(after, max_extent)
    equal = (
        jnp.all(after.h == before.h, axis=-1)
        & (after.observed.s_b == before.observed.s_b)
        & (after.observed.s_l == before.observed.s_l)
        & (after.committed.c_b == before.committed.c_b)
        & (after.committed.activity == before.committed.activity)
        & (after.committed.length == before.committed.length)
        & jnp.all(after_b == before_b, axis=-1)
        & jnp.all(after_l == before_l, axis=-1)
        & (after_valid == before_valid)
    )
    return jnp.all(jnp.where(selected, True, equal), axis=1)


def _validate_commit_scope(
    previous: CommittedStructure,
    current: CommittedStructure,
    masks: StructureCommitMasks,
    state: CarrierState,
) -> None:
    boundary_mask = _validate_site_mask(masks.boundary_edge, state, "boundary_edge")
    extent_mask = _validate_site_mask(masks.extent_site, state, "extent_site")
    if not bool(jnp.all(jnp.where(boundary_mask, True, current.c_b == previous.c_b))):
        raise ValueError("structural commit changed a boundary outside its G-owned mask")
    if not bool(jnp.all(jnp.where(extent_mask, True, current.length == previous.length))):
        raise ValueError("structural commit changed an extent outside its G-owned mask")


def _empty_trace(
    state: CarrierState, evidence: DenoiseEvidence, max_extent: int
) -> RefinementTrace:
    batch, capacity, width = state.h.shape
    p_b, p_l, _ = _prediction_snapshot(state, max_extent)
    length = state.committed.length
    if length is None:
        raise ValueError("Phase-H traces require explicit committed lengths")

    def empty(value: Array) -> Array:
        return jnp.zeros((0, *value.shape), dtype=value.dtype)

    site_bool = jnp.zeros((0, batch, capacity), dtype=bool)
    site_float = jnp.zeros((0, batch, capacity), dtype=state.h.dtype)
    batch_bool = jnp.zeros((0, batch), dtype=bool)
    batch_int = jnp.zeros((0, batch), dtype=jnp.int32)
    return RefinementTrace(
        jnp.zeros((0,), dtype=jnp.int32),
        site_float,
        site_bool,
        site_float,
        site_bool,
        site_bool,
        site_bool,
        jnp.zeros((0,), dtype=jnp.int32),
        jnp.zeros((0,), dtype=jnp.int32),
        batch_int,
        batch_int,
        site_bool,
        jnp.zeros((0, batch, capacity, width), dtype=state.h.dtype),
        jnp.zeros((0, batch, capacity, width), dtype=state.h.dtype),
        jnp.zeros((0, batch, capacity, width), dtype=state.h.dtype),
        jnp.zeros((0, batch, capacity, width), dtype=state.h.dtype),
        empty(state.observed.s_b),
        empty(state.observed.s_b),
        empty(state.observed.s_b),
        empty(state.observed.s_b),
        empty(state.observed.s_l),
        empty(state.observed.s_l),
        empty(state.observed.s_l),
        empty(state.observed.s_l),
        empty(p_b),
        empty(p_b),
        empty(p_b),
        empty(p_b),
        empty(p_l),
        empty(p_l),
        empty(p_l),
        empty(p_l),
        site_bool,
        site_bool,
        site_bool,
        site_bool,
        empty(state.committed.c_b),
        empty(state.committed.c_b),
        empty(length),
        empty(length),
        empty(state.committed.activity),
        empty(state.committed.activity),
        site_bool,
        site_bool,
        jnp.zeros((0, 2), dtype=jnp.uint32),
        batch_bool,
        batch_bool,
        site_bool,
        site_bool,
        site_bool,
        site_float,
        site_float,
        jnp.zeros((0,), dtype=jnp.int32),
        empty(evidence.actual_s_indices),
        empty(evidence.denoise_condition_indices),
        empty(evidence.diffusion_keys),
        empty(evidence.operator_coordinates),
        empty(evidence.operator_keys),
        jnp.zeros((0,), dtype=jnp.int32),
        jnp.zeros((0, batch), dtype=state.h.dtype),
        batch_int,
        batch_bool,
        jnp.zeros((0, batch), dtype=state.h.dtype),
        batch_int,
        batch_bool,
        batch_bool,
    )


def run_draft_refinement(
    initial: CarrierState,
    denoise: DenoiseFn,
    selector: SelectorFn,
    reset_h: Array | float,
    context: RefinementContext,
    *,
    refinement_steps: int,
    max_extent: int,
    unknown_codes: StructureUnknownCodes | None = None,
    commit: StructuralCommitFn | None = None,
    close_region: RegionClosureFn | None = None,
    structure_key: StructureKeyFn | None = None,
) -> RefinementResult:
    """Run draft then select/reset/global-refine/structural-commit x R."""
    if refinement_steps < 0:
        raise ValueError("refinement_steps must be non-negative")
    codes = unknown_codes or structure_unknown_codes(max_extent)
    validate_refinement_context(context, batch=initial.h.shape[0])
    validate_refinement_state(
        initial, max_extent=max_extent, unknown_codes=codes, allow_unknown=False
    )
    draft_occurrence = RefinementOccurrence(jnp.asarray(0), jnp.asarray(-1))
    draft_result = denoise(initial, "draft", context, draft_occurrence)
    _validate_evidence(draft_result.evidence)
    draft = replace(draft_result.state, committed=initial.committed)
    validate_refinement_state(draft, max_extent=max_extent, unknown_codes=codes, allow_unknown=True)
    if refinement_steps == 0:
        return RefinementResult(
            draft,
            draft,
            context,
            draft_result.evidence,
            _empty_trace(draft, draft_result.evidence, max_extent),
        )
    supplied = (commit is not None, close_region is not None, structure_key is not None)
    if any(supplied) and not all(supplied):
        raise ValueError("commit, close_region, and structure_key must be supplied together")

    rows: dict[str, list[Array]] = {name: [] for name in RefinementTrace._fields}
    current = draft
    for r_index in range(1, refinement_steps + 1):
        occurrence = RefinementOccurrence(jnp.asarray(r_index), jnp.asarray(r_index - 1))
        selection = _validate_selection(selector(current, context, occurrence), current)
        reset = reset_selected_carriers(
            current, selection.closed, reset_h, max_extent=max_extent, unknown_codes=codes
        )
        validate_refinement_state(
            reset, max_extent=max_extent, unknown_codes=codes, allow_unknown=True
        )
        refine_result = denoise(reset, "refine", context, occurrence)
        _validate_evidence(refine_result.evidence)
        candidate = replace(refine_result.state, committed=current.committed)
        validate_refinement_state(
            candidate, max_extent=max_extent, unknown_codes=codes, allow_unknown=True
        )
        regenerated = preserve_unselected_state(
            current, candidate, selection.closed, max_extent=max_extent
        )
        validate_refinement_state(
            regenerated, max_extent=max_extent, unknown_codes=codes, allow_unknown=True
        )

        zero_site = jnp.zeros(selection.closed.shape, dtype=bool)
        zero_float = jnp.zeros(selection.closed.shape, dtype=current.h.dtype)
        key = jnp.zeros((2,), dtype=jnp.uint32)
        routing_valid = jnp.ones((current.h.shape[0],), dtype=bool)
        routing_overflow = jnp.zeros_like(routing_valid)
        masks = StructureCommitMasks(zero_site, zero_site)
        changed = reactivated = deactivated = zero_site
        boundary_log_prob = length_log_prob = zero_float
        policy_code = jnp.asarray(-1, dtype=jnp.int32)
        committed = regenerated
        if commit is not None and close_region is not None and structure_key is not None:
            masks = close_region(current.committed, selection, context, occurrence)
            _validate_site_mask(masks.boundary_edge, current, "boundary_edge")
            _validate_site_mask(masks.extent_site, current, "extent_site")
            if not bool(jnp.all(masks.boundary_edge <= selection.closed)) or not bool(
                jnp.all(masks.extent_site <= selection.closed)
            ):
                raise ValueError("G commit masks must remain within the selected closed region")
            if regenerated.predicted is None:
                raise ValueError("structural commit requires selected predictions")
            _, _, prediction_valid = _prediction_snapshot(regenerated, max_extent)
            required = masks.boundary_edge | masks.extent_site
            if not bool(jnp.all(jnp.where(required, prediction_valid, True))):
                raise ValueError("structural commit mask includes a site without a prediction")
            key = jnp.asarray(structure_key(context, occurrence), dtype=jnp.uint32)
            if key.shape != (2,):
                raise ValueError("structure key must have shape [2]")
            outcome = commit(
                current.committed, regenerated.predicted, masks, key, context, occurrence
            )
            routing_valid = jnp.asarray(outcome.routing_valid, dtype=bool)
            routing_overflow = jnp.asarray(outcome.routing_overflow, dtype=bool)
            if (
                routing_valid.shape != (current.h.shape[0],)
                or routing_overflow.shape != routing_valid.shape
            ):
                raise ValueError("G routing validity must have shape [batch]")
            if not bool(jnp.all(routing_valid)) or bool(jnp.any(routing_overflow)):
                raise ValueError("G structural commit did not produce consumable routing")
            committed = replace(regenerated, committed=outcome.committed)
            validate_refinement_state(
                committed, max_extent=max_extent, unknown_codes=codes, allow_unknown=True
            )
            _validate_commit_scope(current.committed, committed.committed, masks, current)
            changed = _validate_site_mask(outcome.changed, current, "changed")
            reactivated = _validate_site_mask(outcome.reactivated, current, "reactivated")
            deactivated = _validate_site_mask(outcome.deactivated, current, "deactivated")
            previous_length = current.committed.length
            committed_length = committed.committed.length
            if previous_length is None or committed_length is None:
                raise ValueError("Phase-H structural commits require explicit lengths")
            expected_changed = (committed.committed.c_b != current.committed.c_b) | (
                committed_length != previous_length
            )
            expected_reactivated = (previous_length == 0) & (committed_length > 0)
            expected_deactivated = (previous_length > 0) & (committed_length == 0)
            if not bool(jnp.array_equal(changed, expected_changed)):
                raise ValueError("G changed diagnostics disagree with committed structure")
            if not bool(jnp.array_equal(reactivated, expected_reactivated)) or not bool(
                jnp.array_equal(deactivated, expected_deactivated)
            ):
                raise ValueError("G activation diagnostics disagree with committed extents")
            boundary_log_prob = jnp.asarray(outcome.boundary_log_prob)
            length_log_prob = jnp.asarray(outcome.length_log_prob)
            if (
                boundary_log_prob.shape != selection.closed.shape
                or length_log_prob.shape != selection.closed.shape
            ):
                raise ValueError("G log-probability diagnostics must have shape [batch, carrier]")
            if not bool(
                jnp.all(jnp.isfinite(boundary_log_prob)) & jnp.all(jnp.isfinite(length_log_prob))
            ):
                raise ValueError("G log-probability diagnostics must be finite")
            if not bool(
                jnp.all(jnp.where(masks.boundary_edge, True, boundary_log_prob == 0))
                & jnp.all(jnp.where(masks.extent_site, True, length_log_prob == 0))
            ):
                raise ValueError("G log probabilities must be zero outside committed occurrences")
            policy_code = jnp.asarray(outcome.policy_code, dtype=jnp.int32)
            if policy_code.ndim != 0:
                raise ValueError("G policy_code must be scalar")

        selected_rms, selected_support, selected_valid = _masked_rms(
            committed.h - current.h, selection.closed
        )
        collateral_rms, collateral_support, collateral_valid = _masked_rms(
            committed.h - current.h, ~selection.closed
        )
        before_pb, before_pl, before_pv = _prediction_snapshot(current, max_extent)
        reset_pb, reset_pl, reset_pv = _prediction_snapshot(reset, max_extent)
        regenerated_pb, regenerated_pl, regenerated_pv = _prediction_snapshot(
            regenerated, max_extent
        )
        committed_pb, committed_pl, committed_pv = _prediction_snapshot(committed, max_extent)
        values = {
            "refinement_index": occurrence.refinement_index,
            "selector_scores": selection.scores,
            "selector_eligible": selection.eligible,
            "selector_labels": selection.realized_error_label,
            "selector_label_valid": selection.label_valid,
            "seed_selected": selection.seed,
            "selected_closed": selection.closed,
            "selection_budget_scope": selection.budget_scope_code,
            "selection_budget_limit": selection.budget_limit,
            "seed_count": selection.seed_count,
            "closed_count": selection.closed_count,
            "erased": selection.closed,
            "before_h": current.h,
            "reset_h": reset.h,
            "regenerated_h": regenerated.h,
            "committed_h": committed.h,
            "observed_b_before": current.observed.s_b,
            "observed_b_reset": reset.observed.s_b,
            "observed_b_regenerated": regenerated.observed.s_b,
            "observed_b_committed": committed.observed.s_b,
            "observed_l_before": current.observed.s_l,
            "observed_l_reset": reset.observed.s_l,
            "observed_l_regenerated": regenerated.observed.s_l,
            "observed_l_committed": committed.observed.s_l,
            "predicted_b_before": before_pb,
            "predicted_b_reset": reset_pb,
            "predicted_b_regenerated": regenerated_pb,
            "predicted_b_committed": committed_pb,
            "predicted_l_before": before_pl,
            "predicted_l_reset": reset_pl,
            "predicted_l_regenerated": regenerated_pl,
            "predicted_l_committed": committed_pl,
            "predicted_valid_before": before_pv,
            "predicted_valid_reset": reset_pv,
            "predicted_valid_regenerated": regenerated_pv,
            "predicted_valid_committed": committed_pv,
            "boundary_before": current.committed.c_b,
            "boundary_after": committed.committed.c_b,
            "length_before": current.committed.length,
            "length_after": committed.committed.length,
            "activity_before": current.committed.activity,
            "activity_after": committed.committed.activity,
            "boundary_commit_mask": masks.boundary_edge,
            "extent_commit_mask": masks.extent_site,
            "structure_key": key,
            "routing_valid": routing_valid,
            "routing_overflow": routing_overflow,
            "structure_changed": changed,
            "reactivated": reactivated,
            "deactivated": deactivated,
            "boundary_log_prob": boundary_log_prob,
            "length_log_prob": length_log_prob,
            "commit_policy_code": policy_code,
            "actual_s_indices": refine_result.evidence.actual_s_indices,
            "denoise_condition_indices": refine_result.evidence.denoise_condition_indices,
            "diffusion_keys": refine_result.evidence.diffusion_keys,
            "operator_coordinates": refine_result.evidence.operator_coordinates,
            "operator_keys": refine_result.evidence.operator_keys,
            "model_applications": refine_result.evidence.model_applications,
            "selected_change_rms": selected_rms,
            "selected_change_support": selected_support,
            "selected_change_valid": selected_valid,
            "collateral_change_rms": collateral_rms,
            "collateral_change_support": collateral_support,
            "collateral_change_valid": collateral_valid,
            "unselected_preserved": _unselected_preserved(
                current, committed, selection.closed, max_extent
            ),
        }
        for name in RefinementTrace._fields:
            rows[name].append(values[name])
        current = committed
    trace = RefinementTrace(*(jnp.stack(rows[name]) for name in RefinementTrace._fields))
    return RefinementResult(draft, current, context, draft_result.evidence, trace)


def refinement_error_diagnostics(trace: RefinementTrace, clean_h: Array) -> dict[str, Array]:
    """Return selected gain and collateral change with explicit support validity."""
    clean_h = jnp.asarray(clean_h)
    if trace.before_h.shape[1:] != clean_h.shape:
        raise ValueError("clean_h must match one traced carrier state")
    before_error = jnp.mean(jnp.square(trace.before_h - clean_h[None, ...]), axis=-1)
    after_error = jnp.mean(jnp.square(trace.committed_h - clean_h[None, ...]), axis=-1)

    def masked_mean(value: Array, mask: Array) -> tuple[Array, Array, Array]:
        support = jnp.sum(mask, axis=2)
        valid = support > 0
        mean = jnp.sum(jnp.where(mask, value, 0.0), axis=2) / jnp.maximum(support, 1)
        return jnp.where(valid, mean, jnp.nan), support, valid

    baseline, selected_support, selected_valid = masked_mean(before_error, trace.selected_closed)
    after, _, _ = masked_mean(after_error, trace.selected_closed)
    collateral, collateral_support, collateral_valid = masked_mean(
        after_error - before_error, ~trace.selected_closed
    )
    return {
        "selected_region_baseline_mse": baseline,
        "selected_region_after_mse": after,
        "selected_region_gain": baseline - after,
        "selected_support": selected_support,
        "selected_valid": selected_valid,
        "collateral_mse_change": collateral,
        "collateral_support": collateral_support,
        "collateral_valid": collateral_valid,
        "unselected_preserved": trace.unselected_preserved,
    }


def selector_diagnostics(selected: Array, realized_error: Array) -> dict[str, Array]:
    """Report coverage/error summaries with honest empty-support handling."""
    selected = jnp.asarray(selected, dtype=bool)
    realized_error = jnp.asarray(realized_error)
    if selected.shape != realized_error.shape or selected.ndim != 2:
        raise ValueError("selected and realized_error must have identical [batch, carrier] shapes")

    def masked_mean(mask: Array) -> tuple[Array, Array, Array]:
        support = jnp.sum(mask, axis=1)
        valid = support > 0
        mean = jnp.sum(jnp.where(mask, realized_error, 0.0), axis=1) / jnp.maximum(support, 1)
        return jnp.where(valid, mean, jnp.nan), support, valid

    selected_mean, selected_support, selected_valid = masked_mean(selected)
    unselected_mean, unselected_support, unselected_valid = masked_mean(~selected)
    return {
        "selected_fraction": jnp.mean(selected, axis=1),
        "selected_error_mean": selected_mean,
        "selected_support": selected_support,
        "selected_valid": selected_valid,
        "unselected_error_mean": unselected_mean,
        "unselected_support": unselected_support,
        "unselected_valid": unselected_valid,
    }


def selector_calibration(
    error_probability: Array,
    error_label: Array,
    eligible: Array,
    *,
    thresholds: Array,
    bins: int = 10,
) -> dict[str, Array]:
    """Binary selector calibration with supports and coverage/capture curves."""
    probability = jnp.asarray(error_probability)
    label = jnp.asarray(error_label, dtype=bool)
    eligible = jnp.asarray(eligible, dtype=bool)
    thresholds = jnp.asarray(thresholds, dtype=probability.dtype)
    if (
        probability.ndim != 2
        or label.shape != probability.shape
        or eligible.shape != probability.shape
    ):
        raise ValueError("selector calibration inputs must share [batch, carrier] shape")
    if bins <= 0 or thresholds.ndim != 1:
        raise ValueError("bins must be positive and thresholds must be one-dimensional")
    if not bool(jnp.all(jnp.isfinite(probability))) or not bool(
        jnp.all((probability >= 0) & (probability <= 1))
    ):
        raise ValueError("error_probability must be finite and lie in [0, 1]")
    support = jnp.sum(eligible)
    valid = support > 0
    weights = eligible.astype(probability.dtype)
    brier = jnp.sum(
        jnp.square(probability - label.astype(probability.dtype)) * weights
    ) / jnp.maximum(support, 1)
    ece = jnp.asarray(0.0, dtype=probability.dtype)
    bin_support = []
    for index in range(bins):
        low, high = index / bins, (index + 1) / bins
        in_bin = (
            eligible
            & (probability >= low)
            & (probability < high if index + 1 < bins else probability <= high)
        )
        count = jnp.sum(in_bin)
        bin_support.append(count)
        accuracy = jnp.sum(label & in_bin) / jnp.maximum(count, 1)
        confidence = jnp.sum(jnp.where(in_bin, probability, 0.0)) / jnp.maximum(count, 1)
        ece += (count / jnp.maximum(support, 1)) * jnp.abs(accuracy - confidence)
    selected = (probability[None, ...] >= thresholds[:, None, None]) & eligible[None, ...]
    selected_support = jnp.sum(selected, axis=(1, 2))
    error_support = jnp.sum(label & eligible)
    captured = jnp.sum(selected & label[None, ...], axis=(1, 2))
    return {
        "support": support,
        "valid": valid,
        "brier": jnp.where(valid, brier, jnp.nan),
        "ece": jnp.where(valid, ece, jnp.nan),
        "bin_support": jnp.stack(bin_support),
        "thresholds": thresholds,
        "selected_support_by_threshold": selected_support,
        "coverage_by_threshold": selected_support / jnp.maximum(support, 1),
        "error_captured_by_threshold": captured / jnp.maximum(error_support, 1),
    }
