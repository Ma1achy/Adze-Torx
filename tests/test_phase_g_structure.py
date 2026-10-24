from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from adze_t.generated_structure import (
    PHASE_G_GENERATED_STRUCTURE_INTERFACE_V0,
    categorical_calibration,
    commit_predicted_structure,
    generated_routing,
    score_corrected_structure_loss,
    structure_calibration,
    structure_churn,
)
from adze_t.state import CarrierState, CommittedStructure, ObservedStructure, PredictedStructure


def _committed():
    length = jnp.array([[2, 0, 1, 1]], dtype=jnp.int32)
    return CommittedStructure(
        c_b=jnp.array([[0, 1, 0, 1]], dtype=jnp.int32),
        activity=length > 0,
        length=length,
    )


def _logits(boundaries, lengths, *, extent_classes=4):
    b = jax.nn.one_hot(jnp.asarray(boundaries), 2) * 12.0
    ell = jax.nn.one_hot(jnp.asarray(lengths), extent_classes) * 12.0
    return PredictedStructure(b, ell)


def test_prediction_only_never_rewrites_committed_routing():
    previous = _committed()
    predicted = _logits([[1, 0, 1, 0]], [[0, 3, 0, 2]])
    transition = commit_predicted_structure(previous, predicted, policy="prediction_only")
    assert PHASE_G_GENERATED_STRUCTURE_INTERFACE_V0.endswith("_V0")
    assert jnp.array_equal(transition.committed.c_b, previous.c_b)
    assert jnp.array_equal(transition.committed.effective_length(), previous.effective_length())
    assert not bool(jnp.any(transition.changed))
    assert jnp.array_equal(transition.descendant_log_prob, jnp.zeros((1,)))


def test_argmax_commit_recovers_boundary_extent_and_forces_terminal_cut():
    previous = _committed()
    predicted = _logits([[1, 0, 1, 0]], [[0, 3, 0, 2]])
    transition = commit_predicted_structure(previous, predicted, policy="argmax")
    assert transition.committed.c_b.tolist() == [[1, 0, 1, 1]]
    assert transition.committed.effective_length().tolist() == [[0, 3, 0, 2]]
    assert transition.committed.activity.tolist() == [[False, True, False, True]]
    assert transition.reactivated.tolist() == [[False, True, False, False]]
    assert transition.deactivated.tolist() == [[True, False, True, False]]


def test_partial_commit_preserves_unselected_sites_and_carrier_identity():
    previous = _committed()
    predicted = _logits([[1, 0, 1, 0]], [[0, 3, 0, 2]])
    mask = jnp.array([[False, True, False, True]])
    transition = commit_predicted_structure(previous, predicted, policy="argmax", commit_mask=mask)
    assert transition.committed.c_b.tolist() == [[0, 0, 0, 1]]
    assert transition.committed.effective_length().tolist() == [[2, 3, 1, 2]]


def test_ell_zero_is_non_emitting_but_query_active_and_reactivatable():
    previous = _committed()
    before = generated_routing(previous, M_max=4, K=4)
    m = int(before.metadata.carrier_to_m[0, 1])
    k = int(before.metadata.carrier_to_k[0, 1])
    assert bool(before.metadata.query_mask[0, m, k])
    assert not bool(before.metadata.kv_mask[0, m, k])
    assert not bool(before.metadata.pool_mask[0, m, k])
    assert not bool(before.metadata.emit_mask[0, m, k])

    predicted = _logits([[0, 1, 0, 1]], [[2, 3, 1, 1]])
    after_commit = commit_predicted_structure(previous, predicted, policy="argmax")
    after = generated_routing(after_commit.committed, M_max=4, K=4)
    new_m = int(after.metadata.carrier_to_m[0, 1])
    new_k = int(after.metadata.carrier_to_k[0, 1])
    assert bool(after.metadata.kv_mask[0, new_m, new_k])
    assert bool(after.metadata.emit_mask[0, new_m, new_k])


def test_changed_boundaries_repack_without_reindexing_carriers():
    previous = _committed()
    old = generated_routing(previous, M_max=4, K=4)
    predicted = _logits([[1, 0, 0, 1]], [[2, 0, 1, 1]])
    transition = commit_predicted_structure(previous, predicted, policy="argmax")
    new = generated_routing(transition.committed, M_max=4, K=4)
    assert old.metadata.carrier_to_m.tolist() == [[0, 0, 1, 1]]
    assert new.metadata.carrier_to_m.tolist() == [[0, 1, 1, 1]]
    assert sorted(new.metadata.packed_to_carrier[0][new.metadata.slot_valid[0]].tolist()) == [
        0,
        1,
        2,
        3,
    ]


def test_fixed_and_generated_structure_share_identical_routing_interface():
    fixed = _committed()
    predicted = _logits(fixed.c_b, fixed.length)
    generated = commit_predicted_structure(fixed, predicted, policy="argmax").committed
    fixed_route = generated_routing(fixed, M_max=4, K=4)
    generated_route = generated_routing(generated, M_max=4, K=4)
    assert jax.tree_util.tree_all(
        jax.tree_util.tree_map(jnp.array_equal, fixed_route.metadata, generated_route.metadata)
    )


def test_unknown_observations_have_no_route_to_committed_packing():
    committed = _committed()
    first = CarrierState(
        h=jnp.zeros((1, 4, 2)),
        observed=ObservedStructure(
            s_b=jnp.full((1, 4), 2, dtype=jnp.int32),
            s_l=jnp.full((1, 4), 5, dtype=jnp.int32),
        ),
        committed=committed,
    )
    second = CarrierState(
        h=first.h,
        observed=ObservedStructure(
            s_b=jnp.full((1, 4), 91, dtype=jnp.int32),
            s_l=jnp.full((1, 4), -7, dtype=jnp.int32),
        ),
        committed=committed,
    )
    first_route = generated_routing(first.committed, M_max=4, K=4)
    second_route = generated_routing(second.committed, M_max=4, K=4)
    assert jax.tree_util.tree_all(
        jax.tree_util.tree_map(jnp.array_equal, first_route.metadata, second_route.metadata)
    )


def test_sample_commit_is_keyed_and_reports_only_actual_occurrence_log_probs():
    previous = _committed()
    predicted = PredictedStructure(
        b_logits=jnp.zeros((1, 4, 2)),
        l_logits=jnp.zeros((1, 4, 4)),
    )
    mask = jnp.array([[True, False, True, True]])
    first = commit_predicted_structure(
        previous, predicted, policy="sample", commit_mask=mask, key=jax.random.key(7)
    )
    second = commit_predicted_structure(
        previous, predicted, policy="sample", commit_mask=mask, key=jax.random.key(7)
    )
    assert jnp.array_equal(first.boundary_sample, second.boundary_sample)
    assert jnp.array_equal(first.length_sample, second.length_sample)
    assert first.boundary_log_prob[0, -1] == 0
    assert first.boundary_log_prob[0, 1] == 0
    assert first.length_log_prob[0, 1] == 0


def test_sampled_downstream_effect_uses_frozen_score_bridge_and_has_gradient():
    previous = CommittedStructure(
        c_b=jnp.ones((1, 1), dtype=jnp.int32),
        activity=jnp.ones((1, 1), dtype=bool),
        length=jnp.ones((1, 1), dtype=jnp.int32),
    )
    boundary_logits = jnp.zeros((1, 1, 2))
    key = jax.random.key(11)

    def corrected(theta):
        logits = jnp.stack((jnp.zeros_like(theta), theta)).reshape(1, 1, 2)
        transition = commit_predicted_structure(
            previous,
            PredictedStructure(boundary_logits, logits),
            policy="sample",
            key=key,
        )
        descendant = 1.0 + transition.length_sample[:, 0].astype(jnp.float32)
        return jnp.mean(score_corrected_structure_loss(descendant, transition))

    value, gradient = jax.value_and_grad(corrected)(jnp.asarray(0.2))
    assert value in (1.0, 2.0)
    assert jnp.isfinite(gradient)
    assert jnp.abs(gradient) > 0


def test_score_bridge_rejects_argmax_to_prevent_estimator_double_counting():
    transition = commit_predicted_structure(
        _committed(), _logits([[0, 1, 0, 1]], [[2, 0, 1, 1]]), policy="argmax"
    )
    with pytest.raises(ValueError, match="sampled downstream"):
        score_corrected_structure_loss(jnp.ones((1,)), transition)


def test_churn_separates_boundary_extent_reactivation_and_deactivation():
    previous = _committed()
    current = commit_predicted_structure(
        previous, _logits([[1, 1, 0, 1]], [[0, 2, 1, 0]]), policy="argmax"
    ).committed
    metrics = structure_churn(previous, current)
    assert metrics["boundary_change_rate"] == pytest.approx(1 / 3)
    assert metrics["extent_change_rate"] == pytest.approx(0.75)
    assert metrics["reactivation_rate"] == pytest.approx(0.25)
    assert metrics["deactivation_rate"] == pytest.approx(0.5)


def test_boundary_and_extent_calibration_are_correct_for_confident_recovery():
    labels_b = jnp.array([[0, 1, 0, 1]])
    labels_l = jnp.array([[2, 0, 1, 3]])
    predicted = _logits(labels_b, labels_l)
    metrics = structure_calibration(predicted, labels_b, labels_l, bins=5)
    assert metrics["boundary"]["count"] == 3
    assert metrics["boundary"]["accuracy"] == 1
    assert metrics["extent"]["accuracy"] == 1
    assert metrics["boundary"]["nll"] < 1.0e-4
    assert metrics["extent"]["brier"] < 1.0e-6


def test_calibration_rejects_invalid_bin_count():
    with pytest.raises(ValueError, match="positive"):
        categorical_calibration(
            jnp.zeros((1, 1, 2)), jnp.zeros((1, 1), int), jnp.ones((1, 1), bool), bins=0
        )


def test_generated_routing_exposes_overflow_instead_of_truncating():
    structure = CommittedStructure(
        c_b=jnp.array([[1, 1, 1, 1]], dtype=jnp.int32),
        activity=jnp.ones((1, 4), dtype=bool),
        length=jnp.ones((1, 4), dtype=jnp.int32),
    )
    routing = generated_routing(structure, M_max=2, K=2)
    assert bool(routing.metadata.block_count_overflow[0])


def test_commit_shape_and_key_contracts_fail_closed():
    previous = _committed()
    with pytest.raises(ValueError, match="boundary logits"):
        commit_predicted_structure(
            previous,
            PredictedStructure(jnp.zeros((1, 4, 3)), jnp.zeros((1, 4, 4))),
            policy="argmax",
        )
    with pytest.raises(ValueError, match="explicit JAX key"):
        commit_predicted_structure(
            previous,
            PredictedStructure(jnp.zeros((1, 4, 2)), jnp.zeros((1, 4, 4))),
            policy="sample",
        )
