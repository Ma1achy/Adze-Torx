from dataclasses import replace
from typing import NamedTuple

import jax
import jax.numpy as jnp
import pytest

from adze_t.refinement import (
    DenoiseEvidence,
    DenoiseResult,
    RefinementContext,
    RegionSelection,
    SelectionPolicy,
    StructureCommitMasks,
    refinement_error_diagnostics,
    reset_selected_carriers,
    run_draft_refinement,
    select_regions,
    selector_calibration,
    selector_diagnostics,
    validate_refinement_state,
)
from adze_t.state import CarrierState, CommittedStructure, ObservedStructure, PredictedStructure
from adze_t.structure_unknown import (
    BOUNDARY_LEFT_BOS,
    BOUNDARY_LEFT_UNKNOWN,
    BOUNDARY_UNKNOWN,
    migrate_dit_structure_unknown_embeddings,
    resolve_structural_embedding_indices,
    structure_unknown_codes,
)


MAX_EXTENT = 4


def _state(*, predicted: bool = True, batch: int = 1) -> CarrierState:
    h = jnp.arange(batch * 8, dtype=jnp.float32).reshape(batch, 4, 2)
    c_b = jnp.broadcast_to(jnp.array([[0, 0, 0, 1]], dtype=jnp.int32), (batch, 4))
    length = jnp.ones((batch, 4), dtype=jnp.int32)
    prediction = None
    if predicted:
        prediction = PredictedStructure(
            b_logits=jnp.ones((batch, 4, 2)),
            l_logits=jnp.ones((batch, 4, MAX_EXTENT + 1)),
        )
    return CarrierState(
        h=h,
        observed=ObservedStructure(s_b=c_b, s_l=length),
        committed=CommittedStructure(c_b=c_b, activity=jnp.ones_like(length), length=length),
        predicted=prediction,
    )


def _context(batch: int = 1) -> RefinementContext:
    return RefinementContext(
        execution_kind=jnp.asarray(0),
        execution_index=jnp.asarray(17),
        global_example_ids=jnp.arange(batch, dtype=jnp.int32) + 100,
        diffusion_root=jax.random.PRNGKey(1),
        operator_root=jax.random.PRNGKey(2),
        structure_root=jax.random.PRNGKey(3),
    )


def _evidence(offset: int = 0) -> DenoiseEvidence:
    return DenoiseEvidence(
        actual_s_indices=jnp.array([0, 1], dtype=jnp.int32),
        denoise_condition_indices=jnp.array([0, 1], dtype=jnp.int32),
        diffusion_keys=jnp.array([[offset + 1, offset + 2]], dtype=jnp.uint32),
        operator_coordinates=jnp.array([[0, 0, 0, 7], [1, 0, 0, 7]], dtype=jnp.int32),
        operator_keys=jnp.array(
            [[offset + 3, offset + 4], [offset + 5, offset + 6]], dtype=jnp.uint32
        ),
        model_applications=jnp.asarray(2, dtype=jnp.int32),
    )


def _selection(state: CarrierState, mask: jax.Array) -> RegionSelection:
    scores = jnp.where(mask, 0.9, 0.1)
    return RegionSelection(
        scores=scores,
        eligible=jnp.ones_like(mask),
        seed=mask,
        closed=mask,
        budget_scope_code=jnp.asarray(0),
        budget_limit=jnp.asarray(-1),
        seed_count=jnp.sum(mask, axis=1),
        closed_count=jnp.sum(mask, axis=1),
        realized_error_label=scores,
        label_valid=jnp.ones_like(mask),
    )


class _CommitOutcome(NamedTuple):
    committed: CommittedStructure
    routing_valid: jax.Array
    routing_overflow: jax.Array
    changed: jax.Array
    reactivated: jax.Array
    deactivated: jax.Array
    boundary_log_prob: jax.Array
    length_log_prob: jax.Array
    policy_code: jax.Array


def _outcome(previous: CommittedStructure, masks: StructureCommitMasks) -> _CommitOutcome:
    assert previous.length is not None
    boundary = jnp.where(masks.boundary_edge, 1, previous.c_b)
    length = jnp.where(masks.extent_site, 2, previous.length)
    committed = CommittedStructure(boundary, length > 0, length)
    changed = (boundary != previous.c_b) | (length != previous.length)
    zeros = jnp.zeros_like(changed)
    return _CommitOutcome(
        committed,
        jnp.ones((boundary.shape[0],), dtype=bool),
        jnp.zeros((boundary.shape[0],), dtype=bool),
        changed,
        zeros,
        zeros,
        jnp.zeros(boundary.shape),
        jnp.zeros(boundary.shape),
        jnp.asarray(2),
    )


def test_selector_seed_budget_halo_eligibility_ties_and_inactive_reactivation():
    score = jnp.array([[0.9, 0.9, jnp.inf, 0.8]])
    activity = jnp.array([[1, 1, 0, 0]], dtype=bool)
    result = select_regions(score, activity, SelectionPolicy(0.5, 1, "seed", halo=1))
    assert result.seed.tolist() == [[True, False, False, False]]
    assert result.closed.tolist() == [[True, True, False, False]]
    assert result.eligible.tolist() == [[True, True, False, False]]

    reactivation = select_regions(
        score,
        activity,
        SelectionPolicy(0.5, 2, "seed", halo=1, allow_inactive_reactivation=True),
    )
    assert reactivation.seed.tolist() == [[True, True, False, False]]
    assert reactivation.closed.tolist() == [[True, True, False, False]]
    assert not bool(reactivation.eligible[0, 2])  # nonfinite is never promoted


def test_selector_expanded_budget_caps_final_region_and_retains_seed_diagnostics():
    score = jnp.array([[0.1, 0.8, 0.9, 0.7, 0.6]])
    activity = jnp.ones_like(score, dtype=bool)
    seed_scoped = select_regions(score, activity, SelectionPolicy(0.75, 2, "seed", halo=1))
    expanded_scoped = select_regions(score, activity, SelectionPolicy(0.75, 3, "expanded", halo=1))
    assert seed_scoped.seed.tolist() == [[False, True, True, False, False]]
    assert seed_scoped.closed.tolist() == [[True, True, True, True, False]]
    assert seed_scoped.seed_count.tolist() == [2]
    assert seed_scoped.closed_count.tolist() == [4]
    assert expanded_scoped.seed.tolist() == seed_scoped.seed.tolist()
    assert expanded_scoped.closed.tolist() == [[False, True, True, True, False]]
    assert expanded_scoped.closed_count.tolist() == [3]
    assert int(expanded_scoped.budget_scope_code) == 1


def test_selector_budget_scope_is_required_and_invalid_scope_rejected():
    score = jnp.ones((1, 2))
    activity = jnp.ones((1, 2), dtype=bool)
    with pytest.raises(ValueError, match="budget_scope"):
        select_regions(score, activity, SelectionPolicy(0.5, 1, "invalid"))  # type: ignore[arg-type]


def test_reset_uses_valid_distinct_unknown_codes_and_invalidates_selected_predictions():
    state = _state()
    selected = jnp.array([[False, True, False, False]])
    reset = reset_selected_carriers(state, selected, -4.0, max_extent=MAX_EXTENT)
    assert reset.h[0, 1].tolist() == [-4.0, -4.0]
    assert int(reset.observed.s_b[0, 1]) == BOUNDARY_UNKNOWN
    assert int(reset.observed.s_l[0, 1]) == MAX_EXTENT + 1
    assert reset.predicted is not None and reset.predicted.valid is not None
    assert not bool(reset.predicted.valid[0, 1])
    assert reset.committed is state.committed


def test_unknown_embedding_migration_preserves_every_existing_row_and_rejects_repeat():
    params = {
        "length_embed": jnp.arange(10, dtype=jnp.float32).reshape(5, 2),
        "boundary_left_embed": jnp.arange(6, dtype=jnp.float32).reshape(3, 2),
        "boundary_right_embed": jnp.arange(4, dtype=jnp.float32).reshape(2, 2),
        "other": jnp.array([7.0]),
    }
    migrated = migrate_dit_structure_unknown_embeddings(params, max_extent=MAX_EXTENT)
    for name in ("length_embed", "boundary_left_embed", "boundary_right_embed"):
        assert jnp.array_equal(migrated[name][:-1], params[name])
        assert jnp.array_equal(migrated[name][-1], jnp.zeros_like(migrated[name][-1]))
    assert migrated["other"] is params["other"]
    with pytest.raises(ValueError, match="length_embed"):
        migrate_dit_structure_unknown_embeddings(migrated, max_extent=MAX_EXTENT)


def test_unknown_embedding_indices_keep_historical_bos_distinct():
    observed_b = jnp.array([[BOUNDARY_UNKNOWN, 0, 1, BOUNDARY_UNKNOWN]])
    observed_l = jnp.array([[MAX_EXTENT + 1, 1, 2, 3]])
    carriers = jnp.array([[0, 1, 2, 3]])
    right, left, extent = resolve_structural_embedding_indices(
        observed_b, observed_l, carriers, batch=1
    )
    assert right.tolist() == [[BOUNDARY_UNKNOWN, 0, 1, BOUNDARY_UNKNOWN]]
    assert left.tolist() == [[BOUNDARY_LEFT_BOS, BOUNDARY_LEFT_UNKNOWN, 0, 1]]
    assert extent.tolist() == observed_l.tolist()


def test_r_zero_records_context_and_strict_dtype_consistent_empty_trace():
    calls = []

    def denoise(state, mode, context, occurrence):
        calls.append((mode, int(occurrence.refinement_index)))
        return DenoiseResult(replace(state, h=state.h + 1), _evidence())

    result = run_draft_refinement(
        _state(),
        denoise,
        lambda state, context, occurrence: _selection(
            state, jnp.zeros(state.h.shape[:2], dtype=bool)
        ),
        0.0,
        _context(),
        refinement_steps=0,
        max_extent=MAX_EXTENT,
    )
    assert calls == [("draft", 0)]
    assert all(
        bool(jnp.array_equal(actual, expected))
        for actual, expected in zip(result.context, _context(), strict=True)
    )
    assert result.trace.activity_before.shape == (0, 1, 4)
    assert result.trace.activity_before.dtype == result.final.committed.activity.dtype
    assert result.trace.operator_keys.shape == (0, 2, 2)


def test_prediction_emerges_on_selected_site_and_every_unselected_field_is_preserved():
    initial = _state(predicted=False)
    selected = jnp.array([[False, True, False, False]])

    def denoise(state, mode, context, occurrence):
        if mode == "draft":
            return DenoiseResult(state, _evidence())
        prediction = PredictedStructure(
            jnp.full((1, 4, 2), 3.0),
            jnp.full((1, 4, MAX_EXTENT + 1), 4.0),
        )
        changed = replace(
            state,
            h=state.h + 9,
            observed=ObservedStructure(
                jnp.zeros_like(state.observed.s_b), jnp.zeros_like(state.observed.s_l)
            ),
            predicted=prediction,
        )
        return DenoiseResult(changed, _evidence(10))

    result = run_draft_refinement(
        initial,
        denoise,
        lambda state, context, occurrence: _selection(state, selected),
        -2.0,
        _context(),
        refinement_steps=1,
        max_extent=MAX_EXTENT,
    )
    assert result.final.predicted is not None and result.final.predicted.valid is not None
    assert result.final.predicted.valid.tolist() == [[False, True, False, False]]
    assert result.final.predicted.b_logits[0, 1].tolist() == [3.0, 3.0]
    assert jnp.array_equal(result.final.h[:, [0, 2, 3]], initial.h[:, [0, 2, 3]])
    assert bool(result.trace.unselected_preserved[0, 0])


def test_structural_only_commit_is_keyed_traced_and_visible_to_next_r_iteration():
    selected = jnp.array([[False, True, False, False]])
    seen = []
    commit_calls = []

    def denoise(state, mode, context, occurrence):
        seen.append((mode, int(occurrence.refinement_index), state.committed.c_b.copy()))
        prediction = PredictedStructure(jnp.ones((1, 4, 2)), jnp.ones((1, 4, MAX_EXTENT + 1)))
        return DenoiseResult(
            replace(state, predicted=prediction), _evidence(int(occurrence.refinement_index) * 10)
        )

    def close(previous, selection, context, occurrence):
        return StructureCommitMasks(selection.closed, selection.closed)

    def key(context, occurrence):
        return jnp.array([99, occurrence.commit_occurrence], dtype=jnp.uint32)

    def commit(previous, predicted, masks, key_value, context, occurrence):
        commit_calls.append((key_value.copy(), int(occurrence.refinement_index)))
        return _outcome(previous, masks)

    result = run_draft_refinement(
        _state(),
        denoise,
        lambda state, context, occurrence: _selection(state, selected),
        0.0,
        _context(),
        refinement_steps=2,
        max_extent=MAX_EXTENT,
        commit=commit,
        close_region=close,
        structure_key=key,
    )
    assert len(commit_calls) == 2
    assert result.trace.structure_key.tolist() == [[99, 0], [99, 1]]
    assert seen[2][2].tolist() == [[0, 1, 0, 1]]  # second refine sees first committed topology
    assert result.trace.actual_s_indices.tolist() == [[0, 1], [0, 1]]
    assert result.trace.operator_coordinates.shape == (2, 2, 4)
    assert result.trace.model_applications.tolist() == [2, 2]


@pytest.mark.parametrize("failure", ["terminal", "overflow", "outside"])
def test_invalid_g_commit_fails_closed(failure):
    selected = jnp.array([[False, True, False, False]])

    def denoise(state, mode, context, occurrence):
        prediction = PredictedStructure(jnp.ones((1, 4, 2)), jnp.ones((1, 4, MAX_EXTENT + 1)))
        return DenoiseResult(replace(state, predicted=prediction), _evidence())

    def close(previous, selection, context, occurrence):
        return StructureCommitMasks(selection.closed, selection.closed)

    def commit(previous, predicted, masks, key, context, occurrence):
        outcome = _outcome(previous, masks)
        if failure == "terminal":
            return outcome._replace(
                committed=replace(outcome.committed, c_b=outcome.committed.c_b.at[:, -1].set(0))
            )
        if failure == "overflow":
            return outcome._replace(routing_overflow=jnp.ones((1,), dtype=bool))
        return outcome._replace(
            committed=replace(outcome.committed, c_b=outcome.committed.c_b.at[:, 0].set(1))
        )

    message = {
        "terminal": "terminal boundary",
        "overflow": "consumable routing",
        "outside": "outside its G-owned mask",
    }[failure]
    with pytest.raises(ValueError, match=message):
        run_draft_refinement(
            _state(),
            denoise,
            lambda state, context, occurrence: _selection(state, selected),
            0.0,
            _context(),
            refinement_steps=1,
            max_extent=MAX_EXTENT,
            commit=commit,
            close_region=close,
            structure_key=lambda context, occurrence: jax.random.PRNGKey(8),
        )


def test_commit_contract_is_all_or_nothing():
    def denoise(state, mode, context, occurrence):
        return DenoiseResult(state, _evidence())

    with pytest.raises(ValueError, match="supplied together"):
        run_draft_refinement(
            _state(),
            denoise,
            lambda state, context, occurrence: _selection(
                state, jnp.zeros(state.h.shape[:2], dtype=bool)
            ),
            0.0,
            _context(),
            refinement_steps=1,
            max_extent=MAX_EXTENT,
            structure_key=lambda context, occurrence: jax.random.PRNGKey(1),
        )


def test_validation_requires_explicit_valid_structure_and_finite_predictions():
    codes = structure_unknown_codes(MAX_EXTENT)
    with pytest.raises(ValueError, match="explicit committed lengths"):
        validate_refinement_state(
            replace(_state(), committed=replace(_state().committed, length=None)),
            max_extent=MAX_EXTENT,
            unknown_codes=codes,
            allow_unknown=False,
        )
    invalid = replace(
        _state(),
        predicted=PredictedStructure(
            jnp.full((1, 4, 2), jnp.nan), jnp.ones((1, 4, MAX_EXTENT + 1))
        ),
    )
    with pytest.raises(ValueError, match="finite"):
        validate_refinement_state(
            invalid, max_extent=MAX_EXTENT, unknown_codes=codes, allow_unknown=False
        )


def test_empty_support_is_nan_with_explicit_counts_and_validity():
    diagnostics = selector_diagnostics(jnp.array([[False, False]]), jnp.array([[1.0, 2.0]]))
    assert int(diagnostics["selected_support"][0]) == 0
    assert not bool(diagnostics["selected_valid"][0])
    assert bool(jnp.isnan(diagnostics["selected_error_mean"][0]))

    def denoise(state, mode, context, occurrence):
        return DenoiseResult(state, _evidence())

    result = run_draft_refinement(
        _state(),
        denoise,
        lambda state, context, occurrence: _selection(
            state, jnp.zeros(state.h.shape[:2], dtype=bool)
        ),
        0.0,
        _context(),
        refinement_steps=1,
        max_extent=MAX_EXTENT,
    )
    errors = refinement_error_diagnostics(result.trace, jnp.zeros_like(result.final.h))
    assert not bool(errors["selected_valid"][0, 0])
    assert bool(jnp.isnan(errors["selected_region_gain"][0, 0]))


def test_selector_calibration_reports_support_bins_and_threshold_curves():
    result = selector_calibration(
        jnp.array([[0.1, 0.9, 0.8, 0.2]]),
        jnp.array([[0, 1, 1, 0]], dtype=bool),
        jnp.ones((1, 4), dtype=bool),
        thresholds=jnp.array([0.0, 0.5, 1.0]),
        bins=2,
    )
    assert int(result["support"]) == 4
    assert result["bin_support"].tolist() == [2, 2]
    assert result["selected_support_by_threshold"].tolist() == [4, 2, 0]
    assert result["error_captured_by_threshold"].tolist() == [1.0, 1.0, 0.0]
    assert float(result["brier"]) == pytest.approx(0.025)


def test_multibatch_trace_preserves_schema_and_selected_collateral_supports():
    state = _state(batch=2)
    masks = jnp.array([[True, False, False, False], [False, False, False, False]])

    def denoise(value, mode, context, occurrence):
        return DenoiseResult(
            value if mode == "draft" else replace(value, h=jnp.zeros_like(value.h)), _evidence()
        )

    result = run_draft_refinement(
        state,
        denoise,
        lambda value, context, occurrence: _selection(value, masks),
        0.0,
        _context(batch=2),
        refinement_steps=1,
        max_extent=MAX_EXTENT,
    )
    assert result.trace.selected_change_support.tolist() == [[1, 0]]
    assert result.trace.selected_change_valid.tolist() == [[True, False]]
    assert bool(jnp.isnan(result.trace.selected_change_rms[0, 1]))
    assert result.trace.collateral_change_support.tolist() == [[3, 4]]
