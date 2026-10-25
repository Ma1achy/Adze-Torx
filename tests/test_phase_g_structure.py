from __future__ import annotations

from dataclasses import replace
from typing import Any

import jax
import jax.numpy as jnp
import pytest

from adze_t.generated_structure import (
    COMMIT_REACTIVATION_FORBIDDEN,
    COMMIT_LOGITS_NONFINITE,
    COMMIT_TEMPERATURE_INVALID,
    ROUTING_BLOCK_COUNT_OVERFLOW,
    ROUTING_BLOCK_LENGTH_OVERFLOW,
    ROUTING_DTYPE,
    CategoricalCommitFactor,
    CommitPolicy,
    HysteresisConfig,
    StructureEstimatorOwner,
    StructureHysteresisState,
    StructureKeyContext,
    categorical_calibration,
    closed_commit_masks,
    commit_predicted_structure,
    commit_predicted_structure_core,
    full_commit_masks,
    generated_routing,
    generated_routing_core,
    score_corrected_structure_loss,
    structure_churn,
)
from adze_t.phase_g import (
    generated_byte_metrics,
    neutral_structure,
    task_structure_commit_masks,
)
from adze_t.packing import pack_values
from adze_t.state import CommittedStructure, PredictedStructure
from adze_t.unpool import unpool_values


def _committed(batch: int = 1) -> CommittedStructure:
    length = jnp.broadcast_to(jnp.array([[2, 0, 1, 1]], jnp.int32), (batch, 4))
    boundary = jnp.broadcast_to(jnp.array([[0, 1, 0, 1]], jnp.int32), (batch, 4))
    return CommittedStructure(boundary, length > 0, length)


def _predicted(boundary, length) -> PredictedStructure:
    return PredictedStructure(
        jax.nn.one_hot(jnp.asarray(boundary), 2) * 12.0,
        jax.nn.one_hot(jnp.asarray(length), 4) * 12.0,
    )


def _context(batch: int = 1, **changes) -> StructureKeyContext:
    values: dict[str, Any] = dict(
        root_key=jax.random.key(7),
        occurrence=3,
        global_example_id=jnp.arange(batch, dtype=jnp.uint32),
        actual_s_index=1,
        refinement_index=2,
        commit_occurrence=4,
    )
    values.update(changes)
    return StructureKeyContext(**values)


def _commit(previous, predicted, *, policy: CommitPolicy = "argmax", masks=None, **kwargs):
    shape = (previous.c_b.shape[0], previous.c_b.shape[1])
    return commit_predicted_structure(
        previous,
        predicted,
        full_commit_masks(shape) if masks is None else masks,
        policy=policy,
        max_extent=3,
        estimator_owner=kwargs.pop("estimator_owner", StructureEstimatorOwner.NONE),
        allow_reactivation=kwargs.pop("allow_reactivation", True),
        **kwargs,
    )


def test_prediction_only_preserves_explicit_committed_state_exactly():
    previous = _committed()
    result = _commit(previous, _predicted([[1, 0, 1, 0]], [[0, 3, 0, 2]]), policy="prediction_only")
    assert jax.tree_util.tree_all(jax.tree.map(jnp.array_equal, result.committed, previous))
    assert not bool(jnp.any(result.changed))


def test_argmax_commit_separates_predictions_and_commit_and_forces_terminal():
    previous = _committed()
    result = _commit(previous, _predicted([[1, 0, 1, 0]], [[0, 3, 0, 2]]))
    assert result.sampled.boundary.tolist() == [[1, 0, 1, 1]]
    assert result.sampled.length.tolist() == [[0, 3, 0, 2]]
    assert result.reactivated.tolist() == [[False, True, False, False]]
    assert result.deactivated.tolist() == [[True, False, True, False]]


def test_closed_masks_preserve_both_exterior_edges_and_terminal():
    masks = closed_commit_masks(jnp.array([[False, True, True, False]]))
    assert masks.extent_site.tolist() == [[False, True, True, False]]
    assert masks.boundary_edge.tolist() == [[False, True, False, False]]
    result = _commit(
        _committed(),
        _predicted([[1, 0, 1, 0]], [[0, 3, 0, 2]]),
        masks=masks,
    )
    assert result.committed.c_b.tolist() == [[0, 0, 0, 1]]
    assert result.committed.length is not None
    assert result.committed.length.tolist() == [[2, 3, 0, 1]]


def test_task_commit_ownership_fixes_mandatory_scaffold_and_scores_only_optional_edges():
    cfg = replace_config()
    previous = neutral_structure(1, cfg)
    masks = task_structure_commit_masks(1, cfg)
    mandatory = (jnp.arange(cfg.carrier.C) + 1) % cfg.packing.K == 0
    assert jnp.all(~masks.boundary_edge[0, mandatory])
    assert jnp.all(masks.boundary_edge[0, ~mandatory])
    assert jnp.all(masks.extent_site)
    predicted = PredictedStructure(
        jax.nn.one_hot(jnp.zeros((1, cfg.carrier.C), jnp.int32), 2) * 4.0,
        jnp.zeros((1, cfg.carrier.C, cfg.carrier.L_max + 1)),
    )
    sampled = commit_predicted_structure_core(
        previous,
        predicted,
        masks,
        policy="sample",
        max_extent=cfg.carrier.L_max,
        key_context=_context(),
        temperature=1.0,
        estimator_owner=StructureEstimatorOwner.LOCAL_SCORE_BRIDGE,
        allow_reactivation=True,
    )
    assert jnp.all(sampled.committed.c_b[0, mandatory] == 1)
    assert jnp.all(sampled.sampled.boundary_log_prob[0, mandatory] == 0)
    routed = generated_routing_core(
        sampled.committed,
        M_max=cfg.packing.M_max,
        K=cfg.packing.K,
        max_extent=cfg.carrier.L_max,
        upstream_valid=sampled.valid,
        upstream_error=sampled.error_code,
    )
    assert bool(routed.executed)


def test_reactivation_policy_has_distinct_fail_closed_code():
    previous = _committed()
    result = commit_predicted_structure_core(
        previous,
        _predicted(previous.c_b, [[2, 3, 1, 1]]),
        full_commit_masks((previous.c_b.shape[0], previous.c_b.shape[1])),
        policy="argmax",
        max_extent=3,
        estimator_owner=StructureEstimatorOwner.NONE,
        allow_reactivation=False,
    )
    assert not bool(result.valid[0])
    assert int(result.error_code[0]) == COMMIT_REACTIVATION_FORBIDDEN
    with pytest.raises(ValueError, match="policy failure"):
        _commit(
            previous,
            _predicted(previous.c_b, [[2, 3, 1, 1]]),
            allow_reactivation=False,
        )


def test_committed_dtypes_fail_closed_in_core_and_eager_wrapper():
    malformed = CommittedStructure(
        _committed().c_b.astype(jnp.float32),
        _committed().activity,
        _committed().length,
    )
    routed = generated_routing_core(malformed, M_max=4, K=4, max_extent=3)
    assert not bool(routed.executed)
    assert int(routed.error_code[0]) == ROUTING_DTYPE
    with pytest.raises(ValueError, match="error codes"):
        generated_routing(malformed, M_max=4, K=4, max_extent=3)


def test_sample_temperature_and_structural_logit_dtypes_are_validated():
    integer_logits = PredictedStructure(
        jnp.zeros((1, 4, 2), jnp.int32), jnp.zeros((1, 4, 4), jnp.int32)
    )
    with pytest.raises(ValueError, match="floating point"):
        _commit(_committed(), integer_logits)
    with pytest.raises(ValueError, match="finite, and positive"):
        _commit(
            _committed(),
            _predicted([[0, 1, 0, 1]], [[2, 0, 1, 1]]),
            policy="sample",
            key_context=_context(),
            temperature=0.0,
            estimator_owner=StructureEstimatorOwner.LOCAL_SCORE_BRIDGE,
        )


def test_compiled_commit_invalid_numerics_fail_before_generated_scatter():
    previous = _committed()
    predicted = _predicted([[0, 1, 0, 1]], [[2, 0, 1, 1]])
    predicted = replace(predicted, b_logits=predicted.b_logits.at[0, 0, 0].set(jnp.nan))
    committed = commit_predicted_structure_core(
        previous,
        predicted,
        full_commit_masks((1, 4)),
        policy="argmax",
        max_extent=3,
        estimator_owner=StructureEstimatorOwner.NONE,
        allow_reactivation=True,
    )
    assert not bool(committed.valid[0])
    assert int(committed.error_code[0]) == COMMIT_LOGITS_NONFINITE
    routed = generated_routing_core(
        committed.committed,
        M_max=4,
        K=4,
        max_extent=3,
        upstream_valid=committed.valid,
        upstream_error=committed.error_code,
    )
    assert not bool(routed.executed)
    assert not bool(jnp.any(routed.metadata.slot_valid))

    zero_temperature = commit_predicted_structure_core(
        previous,
        _predicted([[0, 1, 0, 1]], [[2, 0, 1, 1]]),
        full_commit_masks((1, 4)),
        policy="sample",
        max_extent=3,
        key_context=_context(),
        temperature=0.0,
        estimator_owner=StructureEstimatorOwner.LOCAL_SCORE_BRIDGE,
        allow_reactivation=True,
    )
    assert not bool(zero_temperature.valid[0])
    assert int(zero_temperature.error_code[0]) == COMMIT_TEMPERATURE_INVALID
    assert jnp.all(jnp.isfinite(zero_temperature.sampled.descendant_log_prob))


def test_hysteresis_rejection_preserves_prior_switch_and_confidence():
    previous = _committed()
    state = StructureHysteresisState(
        jnp.zeros((1, 4), jnp.int32),
        jnp.full((1, 4), 0.25),
        jnp.zeros((1, 4), jnp.int32),
        jnp.full((1, 4), 0.5),
    )
    result = _commit(
        previous,
        _predicted([[1, 0, 1, 0]], [[0, 3, 0, 2]]),
        policy="hysteresis",
        key_context=_context(),
        hysteresis_config=HysteresisConfig(0.999999, 0.99, 10),
        hysteresis_state=state,
    )
    assert result.hysteresis_state is not None
    assert jnp.array_equal(
        result.hysteresis_state.boundary_last_confidence, state.boundary_last_confidence
    )
    assert jnp.array_equal(
        result.hysteresis_state.extent_last_confidence, state.extent_last_confidence
    )


def test_sampled_structure_uses_public_factor_and_explicit_local_owner():
    logits = jnp.asarray([0.2, -0.3, 0.7])
    factor = CategoricalCommitFactor(logits)
    action = factor.sample(jax.random.key(9), {"logits": logits}, None, {"temperature": 1.0})
    expected = jax.nn.log_softmax(logits)[action]
    assert (
        factor.log_probability({"logits": logits}, action, None, {"temperature": 1.0}) == expected
    )
    predicted = PredictedStructure(jnp.zeros((1, 4, 2)), jnp.zeros((1, 4, 4)))
    result = _commit(
        _committed(),
        predicted,
        policy="sample",
        key_context=_context(),
        temperature=1.0,
        estimator_owner=StructureEstimatorOwner.LOCAL_SCORE_BRIDGE,
    )
    assert result.sampled.boundary_log_prob[0, -1] == 0
    assert result.estimator_owner is StructureEstimatorOwner.LOCAL_SCORE_BRIDGE
    with pytest.raises(ValueError, match="local_score_bridge ownership"):
        _commit(
            _committed(),
            predicted,
            policy="sample",
            key_context=_context(),
            temperature=1.0,
            estimator_owner=StructureEstimatorOwner.NATIVE_TORX,
        )


def test_structure_keys_reproduce_and_separate_every_coordinate_and_namespace():
    baseline = _context().keys("boundary", 4)
    assert jnp.array_equal(baseline, _context().keys("boundary", 4))
    assert not jnp.array_equal(baseline, _context(occurrence=4).keys("boundary", 4))
    assert not jnp.array_equal(baseline, _context(actual_s_index=2).keys("boundary", 4))
    assert not jnp.array_equal(baseline, _context(refinement_index=3).keys("boundary", 4))
    assert not jnp.array_equal(baseline, _context(commit_occurrence=5).keys("boundary", 4))
    assert not jnp.array_equal(baseline, _context().keys("extent", 4))
    assert not jnp.array_equal(baseline[:, 0], baseline[:, 1])


@pytest.mark.parametrize(
    "mutator,match",
    [
        (lambda c: replace(c, c_b=c.c_b.at[:, -1].set(0)), "error codes"),
        (lambda c: replace(c, c_b=c.c_b.at[:, 0].set(2)), "error codes"),
        (lambda c: replace(c, length=c.length.at[:, 0].set(-1)), "error codes"),
        (lambda c: replace(c, activity=jnp.zeros_like(c.activity)), "error codes"),
    ],
)
def test_invalid_committed_state_is_rejected(mutator, match):
    with pytest.raises(ValueError, match=match):
        generated_routing(mutator(_committed()), M_max=4, K=4, max_extent=3)


def test_phase_g_rejects_implicit_lengths_nonfinite_logits_and_nonboolean_masks():
    previous = replace(_committed(), length=None)
    with pytest.raises(ValueError, match="explicit"):
        generated_routing(previous, M_max=4, K=4, max_extent=3)
    predicted = _predicted([[0, 1, 0, 1]], [[2, 0, 1, 1]])
    predicted = replace(predicted, b_logits=predicted.b_logits.at[0, 0, 0].set(jnp.nan))
    with pytest.raises(ValueError, match="finite"):
        _commit(_committed(), predicted)
    bad_masks = replace(full_commit_masks((1, 4)), extent_site=jnp.ones((1, 4), jnp.int32))
    with pytest.raises(ValueError, match="boolean"):
        _commit(_committed(), _predicted([[0, 1, 0, 1]], [[2, 0, 1, 1]]), masks=bad_masks)


def test_both_overflow_modes_return_nonconsumable_empty_metadata_and_eager_rejects():
    count = CommittedStructure(
        jnp.ones((1, 4), jnp.int32), jnp.ones((1, 4), bool), jnp.ones((1, 4), jnp.int32)
    )
    count_core = generated_routing_core(count, M_max=2, K=4, max_extent=3)
    assert not bool(count_core.executed)
    assert int(count_core.error_code[0]) == ROUTING_BLOCK_COUNT_OVERFLOW
    assert not bool(jnp.any(count_core.metadata.slot_valid))
    with pytest.raises(ValueError, match="capacity"):
        generated_routing(count, M_max=2, K=4, max_extent=3)

    long_block = CommittedStructure(
        jnp.array([[0, 0, 0, 1]], jnp.int32),
        jnp.ones((1, 4), bool),
        jnp.ones((1, 4), jnp.int32),
    )
    length_core = generated_routing_core(long_block, M_max=2, K=2, max_extent=3)
    assert not bool(length_core.executed)
    assert int(length_core.error_code[0]) == ROUTING_BLOCK_LENGTH_OVERFLOW
    assert not bool(jnp.any(length_core.metadata.slot_valid))


def test_routing_records_execute_under_jit_and_scan():
    structure = _committed(batch=2)
    compiled = jax.jit(lambda value: generated_routing_core(value, M_max=4, K=4, max_extent=3))(
        structure
    )
    assert bool(compiled.executed)

    def body(carry, _):
        result = generated_routing_core(carry, M_max=4, K=4, max_extent=3)
        return result.structure, result.executed

    final, executed = jax.lax.scan(body, structure, xs=None, length=2)
    assert jnp.all(executed)
    assert final.length is not None and structure.length is not None
    assert jnp.array_equal(final.length, structure.length)


def test_ell_zero_nonemitting_reactivation_and_pack_unpool_identity_batch_gt_one():
    route = generated_routing(_committed(batch=2), M_max=4, K=4, max_extent=3)
    for batch in range(2):
        m = int(route.metadata.carrier_to_m[batch, 1])
        k = int(route.metadata.carrier_to_k[batch, 1])
        assert bool(route.metadata.query_mask[batch, m, k])
        assert not bool(route.metadata.kv_mask[batch, m, k])
        assert not bool(route.metadata.pool_mask[batch, m, k])
        assert not bool(route.metadata.emit_mask[batch, m, k])


def test_changed_boundaries_repack_and_unpool_the_same_carrier_identities():
    previous = _committed()
    changed = CommittedStructure(
        jnp.asarray([[1, 0, 0, 1]], jnp.int32), previous.activity, previous.length
    )
    before = generated_routing(previous, M_max=4, K=4, max_extent=3)
    after = generated_routing(changed, M_max=4, K=4, max_extent=3)
    assert not jnp.array_equal(before.metadata.carrier_to_m, after.metadata.carrier_to_m)
    values = jnp.arange(12, dtype=jnp.float32).reshape(1, 4, 3)
    for route in (before, after):
        packed = pack_values(values, route.metadata)
        restored = unpool_values(packed, route.metadata, C=4)
        assert jnp.array_equal(restored, values)


def test_score_bridge_has_zero_primal_and_rejects_nonlocal_or_nonsampled_ownership():
    predicted = PredictedStructure(jnp.zeros((1, 4, 2)), jnp.zeros((1, 4, 4)))
    commit = _commit(
        _committed(),
        predicted,
        policy="sample",
        key_context=_context(),
        temperature=1.0,
        estimator_owner=StructureEstimatorOwner.LOCAL_SCORE_BRIDGE,
    )
    descendant = jnp.asarray([2.5])
    assert jnp.array_equal(score_corrected_structure_loss(descendant, commit), descendant)
    argmax = _commit(_committed(), predicted)
    with pytest.raises(ValueError, match="local score"):
        score_corrected_structure_loss(descendant, argmax)


def test_extreme_calibration_nll_uses_log_softmax_without_probability_floor():
    metrics = categorical_calibration(
        jnp.asarray([[[1000.0, -1000.0]]]),
        jnp.asarray([[1]]),
        jnp.asarray([[True]]),
    )
    assert metrics["nll"] == pytest.approx(2000.0)


def test_generated_length_loss_compacts_globally_and_penalizes_missing_extra():
    logits = jnp.full((1, 2, 2, 256), -10.0)
    logits = logits.at[0, 0, 0, 3].set(10.0)
    logits = logits.at[0, 1, 0, 4].set(10.0)
    target = jnp.array([[[3, 0], [4, 0]]], jnp.int32)
    mask = jnp.array([[[True, False], [True, False]]])
    exact = generated_byte_metrics(logits, jnp.array([[1, 1]]), target, mask)
    assert exact.accuracy_per_example[0] == 1
    assert exact.missing_count[0] == 0
    missing = generated_byte_metrics(logits, jnp.array([[1, 0]]), target, mask)
    assert missing.missing_count[0] == 1
    assert missing.loss_per_example[0] > exact.loss_per_example[0]
    empty = generated_byte_metrics(
        logits, jnp.zeros((1, 2), jnp.int32), target, jnp.zeros_like(mask)
    )
    assert empty.loss_per_example[0] == 0


def test_churn_reports_separate_transition_classes_and_neutral_structure_contract():
    previous = _committed()
    current = _commit(previous, _predicted([[1, 1, 0, 1]], [[0, 2, 1, 0]])).committed
    metrics = structure_churn(previous, current)
    assert metrics["boundary_change_rate"] == pytest.approx(1 / 3)
    assert metrics["reactivation_rate"] == pytest.approx(0.25)
    neutral = neutral_structure(1, replace_config())
    assert neutral.c_b[0, -1] == 1
    assert jnp.all(neutral.length == 4)


def replace_config():
    from adze_t.config import REFERENCE_SMALL_V0

    return REFERENCE_SMALL_V0
