from dataclasses import replace

import jax.numpy as jnp

from adze_t.refinement import (
    SelectionPolicy,
    refinement_error_diagnostics,
    reset_selected_carriers,
    run_draft_refinement,
    select_regions,
    selector_diagnostics,
)
from adze_t.state import CarrierState, CommittedStructure, ObservedStructure, PredictedStructure


def _state() -> CarrierState:
    h = jnp.arange(8, dtype=jnp.float32).reshape(1, 4, 2)
    return CarrierState(
        h=h,
        observed=ObservedStructure(
            s_b=jnp.array([[0, 0, 0, 1]], dtype=jnp.int32),
            s_l=jnp.ones((1, 4), dtype=jnp.int32),
        ),
        committed=CommittedStructure(
            c_b=jnp.array([[0, 0, 0, 1]], dtype=jnp.int32),
            activity=jnp.ones((1, 4), dtype=jnp.int32),
            length=jnp.ones((1, 4), dtype=jnp.int32),
        ),
        predicted=PredictedStructure(b_logits=jnp.ones((1, 4, 2)), l_logits=jnp.ones((1, 4, 5))),
    )


def test_selector_threshold_top_k_halo_and_inactive_policy():
    score = jnp.array([[0.1, 0.9, 0.8, 0.7]])
    activity = jnp.array([[1, 1, 0, 1]])
    selected = select_regions(score, activity, SelectionPolicy(0.5, max_sites=1, halo=1))
    assert selected.tolist() == [[True, True, True, False]]
    selected_with_inactive = select_regions(
        score, activity, SelectionPolicy(0.5, max_sites=1, include_inactive=True)
    )
    assert selected_with_inactive.tolist() == [[False, True, False, False]]


def test_reset_erases_only_selected_and_never_commits_structure():
    state = _state()
    selected = jnp.array([[False, True, False, False]])
    reset = reset_selected_carriers(state, selected, jnp.full_like(state.h, -4.0))
    assert reset.h[0, 1].tolist() == [-4.0, -4.0]
    assert reset.h[0, 0].tolist() == state.h[0, 0].tolist()
    assert int(reset.observed.s_b[0, 1]) == -1
    assert int(reset.observed.s_l[0, 1]) == -1
    assert reset.predicted is not None
    assert reset.predicted.b_logits[0, 1].tolist() == [0.0, 0.0]
    assert reset.committed is state.committed


def test_r_zero_runs_only_draft_and_has_strict_empty_trace():
    calls = []

    def denoise(state, mode, r_index):
        calls.append((mode, r_index))
        return replace(state, h=state.h + 1)

    result = run_draft_refinement(
        _state(),
        denoise,
        lambda state, r: jnp.zeros(state.h.shape[:2], bool),
        0.0,
        refinement_steps=0,
    )
    assert calls == [("draft", 0)]
    assert result.trace.selected.shape == (0, 1, 4)
    assert result.trace.before_h.shape == (0, 1, 4, 2)
    assert jnp.array_equal(result.final.h, _state().h + 1)


def test_shared_callback_global_mode_and_unselected_preservation():
    calls = []
    selected = jnp.array([[False, True, False, False]])

    def shared_denoise(state, mode, r_index):
        calls.append((id(shared_denoise), mode, r_index, state.committed.c_b.copy()))
        return replace(state, h=state.h + (1 if mode == "draft" else 10))

    result = run_draft_refinement(
        _state(), shared_denoise, lambda state, r: selected, -2.0, refinement_steps=2
    )
    assert [call[1:3] for call in calls] == [("draft", 0), ("refine", 1), ("refine", 2)]
    assert len({call[0] for call in calls}) == 1
    assert all(jnp.array_equal(call[3], _state().committed.c_b) for call in calls)
    assert jnp.array_equal(result.final.h[:, [0, 2, 3]], result.draft.h[:, [0, 2, 3]])
    assert jnp.all(result.trace.unselected_preserved)
    assert jnp.all(result.trace.collateral_change_rms == 0)


def test_structural_commit_occurs_only_after_refine_and_only_selected_sites_change():
    state = _state()
    selected = jnp.array([[False, True, False, False]])
    seen_in_denoise = []
    commits = []

    def denoise(value, mode, r_index):
        seen_in_denoise.append(value.committed.c_b.copy())
        return replace(value, h=value.h + 1)

    def commit(before, candidate, mask, r_index):
        commits.append(r_index)
        structure = replace(before.committed, c_b=before.committed.c_b.at[0, 1].set(1))
        return replace(candidate, committed=structure)

    result = run_draft_refinement(
        state, denoise, lambda value, r: selected, 0.0, refinement_steps=1, commit=commit
    )
    assert commits == [1]
    assert len(seen_in_denoise) == 2
    assert jnp.array_equal(seen_in_denoise[1], state.committed.c_b)
    assert result.final.committed.c_b.tolist() == [[0, 1, 0, 1]]
    assert result.trace.boundary_before[0].tolist() == [[0, 0, 0, 1]]
    assert result.trace.boundary_after[0].tolist() == [[0, 1, 0, 1]]


def test_missing_candidate_predictions_cannot_erase_unselected_predictions():
    state = _state()
    selected = jnp.array([[False, True, False, False]])

    def denoise(value, mode, r_index):
        if mode == "draft":
            return value
        return replace(value, predicted=None)

    result = run_draft_refinement(
        state, denoise, lambda value, r: selected, 0.0, refinement_steps=1
    )
    assert result.final.predicted is not None
    assert result.final.predicted.b_logits[0, 1].tolist() == [0.0, 0.0]
    assert result.final.predicted.b_logits[0, 0].tolist() == [1.0, 1.0]


def test_trace_and_error_diagnostics_measure_gain_and_collateral_change():
    state = replace(_state(), h=jnp.ones((1, 4, 2)))
    selected = jnp.array([[True, False, False, False]])

    def denoise(value, mode, r_index):
        return value if mode == "draft" else replace(value, h=jnp.zeros_like(value.h))

    result = run_draft_refinement(
        state, denoise, lambda value, r: selected, 0.0, refinement_steps=1
    )
    diagnostics = refinement_error_diagnostics(result.trace, jnp.zeros_like(state.h))
    assert diagnostics["selected_region_baseline_mse"].tolist() == [[1.0]]
    assert diagnostics["selected_region_after_mse"].tolist() == [[0.0]]
    assert diagnostics["selected_region_gain"].tolist() == [[1.0]]
    assert diagnostics["collateral_mse_change"].tolist() == [[0.0]]
    selector = selector_diagnostics(selected, jnp.array([[3.0, 1.0, 1.0, 1.0]]))
    assert selector["selected_fraction"].tolist() == [0.25]
    assert selector["selected_error_mean"].tolist() == [3.0]
    assert selector["unselected_error_mean"].tolist() == [1.0]
