"""Outer-R draft/select/reset/global-refine orchestration.

The controller depends only on :class:`CarrierState` and callbacks, so Phase H
can validate control semantics against fixed/mock structure before Phase G's
generated-structure commit interface is frozen.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Literal, NamedTuple

from jax import Array
import jax.numpy as jnp

from .state import CarrierState, ObservedStructure, PredictedStructure


STRUCTURE_UNKNOWN = -1
RefinementMode = Literal["draft", "refine"]
DenoiseFn = Callable[[CarrierState, RefinementMode, int], CarrierState]
SelectorFn = Callable[[CarrierState, int], Array]
CommitFn = Callable[[CarrierState, CarrierState, Array, int], CarrierState]


@dataclass(frozen=True)
class SelectionPolicy:
    """Deterministic threshold/top-k selector with contiguous halo."""

    threshold: float
    max_sites: int | None = None
    halo: int = 0
    include_inactive: bool = False


class RefinementTrace(NamedTuple):
    """Step-major state trace for the R refinement iterations."""

    selected: Array
    before_h: Array
    reset_h: Array
    regenerated_h: Array
    committed_h: Array
    boundary_before: Array
    boundary_after: Array
    activity_before: Array
    activity_after: Array
    selected_change_rms: Array
    collateral_change_rms: Array
    unselected_preserved: Array


class RefinementResult(NamedTuple):
    draft: CarrierState
    final: CarrierState
    trace: RefinementTrace


def _validate_site_mask(mask: Array, state: CarrierState) -> Array:
    mask = jnp.asarray(mask, dtype=bool)
    if mask.shape != state.h.shape[:2]:
        raise ValueError(f"site mask must have shape {state.h.shape[:2]}, got {mask.shape}")
    return mask


def _site_where(mask: Array, selected: Array, unselected: Array) -> Array:
    expanded = mask.reshape((*mask.shape, *((1,) * (selected.ndim - mask.ndim))))
    return jnp.where(expanded, selected, unselected)


def select_regions(score: Array, activity: Array, policy: SelectionPolicy) -> Array:
    """Select carrier regions from per-site error scores."""
    score = jnp.asarray(score)
    activity = jnp.asarray(activity, dtype=bool)
    if score.ndim != 2 or activity.shape != score.shape:
        raise ValueError("score and activity must have identical [batch, carrier] shapes")
    if policy.halo < 0:
        raise ValueError("halo must be non-negative")
    if policy.max_sites is not None and not 1 <= policy.max_sites <= score.shape[1]:
        raise ValueError("max_sites must lie in [1, carrier capacity]")

    eligible = jnp.isfinite(score)
    if not policy.include_inactive:
        eligible &= activity
    selected = eligible & (score >= policy.threshold)
    if policy.max_sites is not None:
        ranked_score = jnp.where(selected, score, -jnp.inf)
        order = jnp.argsort(ranked_score, axis=1, descending=True)
        rank = (
            jnp.empty_like(order)
            .at[jnp.arange(score.shape[0])[:, None], order]
            .set(jnp.arange(score.shape[1])[None, :])
        )
        selected &= rank < policy.max_sites
    for _ in range(policy.halo):
        selected = selected | jnp.pad(selected[:, :-1], ((0, 0), (1, 0)))
        selected = selected | jnp.pad(selected[:, 1:], ((0, 0), (0, 1)))
    return selected


def reset_selected_carriers(
    state: CarrierState,
    selected: Array,
    reset_h: Array | float,
    *,
    structure_unknown: int = STRUCTURE_UNKNOWN,
) -> CarrierState:
    """Erase selected state without changing committed routing."""
    selected = _validate_site_mask(selected, state)
    reset_h = jnp.asarray(reset_h, dtype=state.h.dtype)
    try:
        reset_h = jnp.broadcast_to(reset_h, state.h.shape)
    except ValueError as exc:
        raise ValueError("reset_h must broadcast to carrier h") from exc
    observed = ObservedStructure(
        s_b=jnp.where(
            selected, jnp.full_like(state.observed.s_b, structure_unknown), state.observed.s_b
        ),
        s_l=jnp.where(
            selected, jnp.full_like(state.observed.s_l, structure_unknown), state.observed.s_l
        ),
    )
    predicted = state.predicted
    if predicted is not None:
        predicted = PredictedStructure(
            b_logits=_site_where(selected, jnp.zeros_like(predicted.b_logits), predicted.b_logits),
            l_logits=_site_where(selected, jnp.zeros_like(predicted.l_logits), predicted.l_logits),
        )
    return CarrierState(
        h=_site_where(selected, reset_h, state.h),
        observed=observed,
        committed=state.committed,
        predicted=predicted,
    )


def preserve_unselected_state(
    before: CarrierState,
    candidate: CarrierState,
    selected: Array,
    *,
    preserve_committed_everywhere: bool = False,
) -> CarrierState:
    """Pin unselected carrier identity/state after global computation."""
    selected = _validate_site_mask(selected, before)
    if candidate.h.shape != before.h.shape:
        raise ValueError("candidate carrier shape must match the pre-refinement state")
    predicted = None
    if before.predicted is not None:
        candidate_b = (
            jnp.zeros_like(before.predicted.b_logits)
            if candidate.predicted is None
            else candidate.predicted.b_logits
        )
        candidate_l = (
            jnp.zeros_like(before.predicted.l_logits)
            if candidate.predicted is None
            else candidate.predicted.l_logits
        )
        predicted = PredictedStructure(
            b_logits=_site_where(selected, candidate_b, before.predicted.b_logits),
            l_logits=_site_where(selected, candidate_l, before.predicted.l_logits),
        )
    observed = ObservedStructure(
        s_b=jnp.where(selected, candidate.observed.s_b, before.observed.s_b),
        s_l=jnp.where(selected, candidate.observed.s_l, before.observed.s_l),
    )
    if preserve_committed_everywhere:
        committed = before.committed
    else:
        committed = replace(
            before.committed,
            c_b=jnp.where(selected, candidate.committed.c_b, before.committed.c_b),
            activity=jnp.where(selected, candidate.committed.activity, before.committed.activity),
            length=jnp.where(
                selected,
                candidate.committed.effective_length(),
                before.committed.effective_length(),
            ),
        )
    return CarrierState(
        h=_site_where(selected, candidate.h, before.h),
        observed=observed,
        committed=committed,
        predicted=predicted,
    )


def _masked_rms(delta: Array, mask: Array) -> Array:
    per_site = jnp.mean(jnp.square(delta), axis=-1)
    count = jnp.maximum(jnp.sum(mask, axis=1), 1)
    return jnp.sqrt(jnp.sum(jnp.where(mask, per_site, 0.0), axis=1) / count)


def _empty_trace(state: CarrierState) -> RefinementTrace:
    batch, capacity, width = state.h.shape
    empty_h = jnp.zeros((0, batch, capacity, width), dtype=state.h.dtype)
    empty_s = jnp.zeros((0, batch, capacity), dtype=bool)
    empty_i = jnp.zeros((0, batch, capacity), dtype=state.committed.c_b.dtype)
    empty_b = jnp.zeros((0, batch), dtype=state.h.dtype)
    return RefinementTrace(
        empty_s,
        empty_h,
        empty_h,
        empty_h,
        empty_h,
        empty_i,
        empty_i,
        empty_i,
        empty_i,
        empty_b,
        empty_b,
        jnp.zeros((0, batch), dtype=bool),
    )


def run_draft_refinement(
    initial: CarrierState,
    denoise: DenoiseFn,
    selector: SelectorFn,
    reset_h: Array | float,
    *,
    refinement_steps: int,
    commit: CommitFn | None = None,
) -> RefinementResult:
    """Run one draft then select/reset/global-refine/commit x R.

    One callback serves both modes, making shared weights the API default.
    Structure is frozen inside denoising and may change only in ``commit``.
    """
    if refinement_steps < 0:
        raise ValueError("refinement_steps must be non-negative")
    draft_candidate = denoise(initial, "draft", 0)
    draft = replace(draft_candidate, committed=initial.committed)
    if refinement_steps == 0:
        return RefinementResult(draft, draft, _empty_trace(draft))

    rows: dict[str, list[Array]] = {name: [] for name in RefinementTrace._fields}
    current = draft
    for r_index in range(1, refinement_steps + 1):
        selected = _validate_site_mask(selector(current, r_index), current)
        reset = reset_selected_carriers(current, selected, reset_h)
        candidate = denoise(reset, "refine", r_index)
        candidate = replace(candidate, committed=current.committed)
        regenerated = preserve_unselected_state(
            current, candidate, selected, preserve_committed_everywhere=True
        )
        committed = (
            regenerated if commit is None else commit(current, regenerated, selected, r_index)
        )
        committed = preserve_unselected_state(current, committed, selected)
        preserved = jnp.all(
            jnp.where(
                selected,
                True,
                jnp.all(committed.h == current.h, axis=-1)
                & (committed.observed.s_b == current.observed.s_b)
                & (committed.observed.s_l == current.observed.s_l)
                & (committed.committed.c_b == current.committed.c_b)
                & (committed.committed.activity == current.committed.activity),
            ),
            axis=1,
        )
        rows["selected"].append(selected)
        rows["before_h"].append(current.h)
        rows["reset_h"].append(reset.h)
        rows["regenerated_h"].append(regenerated.h)
        rows["committed_h"].append(committed.h)
        rows["boundary_before"].append(current.committed.c_b)
        rows["boundary_after"].append(committed.committed.c_b)
        rows["activity_before"].append(current.committed.activity)
        rows["activity_after"].append(committed.committed.activity)
        rows["selected_change_rms"].append(_masked_rms(committed.h - current.h, selected))
        rows["collateral_change_rms"].append(_masked_rms(committed.h - current.h, ~selected))
        rows["unselected_preserved"].append(preserved)
        current = committed
    trace = RefinementTrace(*(jnp.stack(rows[name]) for name in RefinementTrace._fields))
    return RefinementResult(draft, current, trace)


def refinement_error_diagnostics(trace: RefinementTrace, clean_h: Array) -> dict[str, Array]:
    """Return selected-region gain and collateral error by R iteration."""
    clean_h = jnp.asarray(clean_h)
    if trace.before_h.shape[1:] != clean_h.shape:
        raise ValueError("clean_h must match one traced carrier state")
    before_error = jnp.mean(jnp.square(trace.before_h - clean_h[None, ...]), axis=-1)
    after_error = jnp.mean(jnp.square(trace.committed_h - clean_h[None, ...]), axis=-1)

    def masked_mean(value: Array, mask: Array) -> Array:
        count = jnp.maximum(jnp.sum(mask, axis=2), 1)
        return jnp.sum(jnp.where(mask, value, 0.0), axis=2) / count

    baseline = masked_mean(before_error, trace.selected)
    after = masked_mean(after_error, trace.selected)
    return {
        "selected_region_baseline_mse": baseline,
        "selected_region_after_mse": after,
        "selected_region_gain": baseline - after,
        "collateral_mse_change": masked_mean(after_error - before_error, ~trace.selected),
        "unselected_preserved": trace.unselected_preserved,
    }


def selector_diagnostics(selected: Array, realized_error: Array) -> dict[str, Array]:
    """Report selector coverage and selected/unselected realized error."""
    selected = jnp.asarray(selected, dtype=bool)
    realized_error = jnp.asarray(realized_error)
    if selected.shape != realized_error.shape or selected.ndim != 2:
        raise ValueError("selected and realized_error must have identical [batch, carrier] shapes")

    def masked_mean(mask: Array) -> Array:
        count = jnp.maximum(jnp.sum(mask, axis=1), 1)
        return jnp.sum(jnp.where(mask, realized_error, 0.0), axis=1) / count

    return {
        "selected_fraction": jnp.mean(selected, axis=1),
        "selected_error_mean": masked_mean(selected),
        "unselected_error_mean": masked_mean(~selected),
    }
