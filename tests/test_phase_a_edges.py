import jax.numpy as jnp
import pytest

from adze_t.packing import build_pack_metadata
from adze_t.state import (
    CarrierState,
    CommittedStructure,
    ObservedStructure,
    PredictedStructure,
    validate_carrier_state,
)


def test_consecutive_inactive_holes_keep_all_identity_slots():
    c_b = jnp.array([[0, 0, 1, 0, 1]], dtype=jnp.int32)
    activity = jnp.array([[1, 0, 0, 1, 1]], dtype=jnp.int32)
    meta = build_pack_metadata(c_b, activity, M_max=5, K=5)
    assert sorted(meta.packed_to_carrier[0][meta.slot_valid[0]].tolist()) == [0, 1, 2, 3, 4]
    assert meta.query_mask[0, 0, 1]
    assert not meta.kv_mask[0, 0, 1]
    assert not meta.kv_mask[0, 0, 2]


def test_predicted_structure_shape_is_validated_without_becoming_routing():
    state = CarrierState(
        h=jnp.zeros((1, 3, 2)),
        observed=ObservedStructure(jnp.zeros((1, 3)), jnp.zeros((1, 3))),
        committed=CommittedStructure(
            jnp.array([[0, 0, 1]], dtype=jnp.int32),
            jnp.ones((1, 3), dtype=jnp.int32),
        ),
    )
    validate_carrier_state(state)

    bad = CarrierState(
        h=state.h,
        observed=state.observed,
        committed=state.committed,
        predicted=PredictedStructure(jnp.zeros((1, 2)), jnp.zeros((1, 3))),
    )
    with pytest.raises(ValueError):
        validate_carrier_state(bad)
