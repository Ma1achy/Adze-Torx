import jax.numpy as jnp
import pytest

from adze_t.state import (
    CarrierState,
    CommittedStructure,
    ObservedStructure,
    validate_carrier_state,
)


def make_state():
    return CarrierState(
        h=jnp.zeros((2, 4, 8)),
        observed=ObservedStructure(
            s_b=jnp.zeros((2, 4), dtype=jnp.int32),
            s_l=jnp.ones((2, 4), dtype=jnp.int32),
        ),
        committed=CommittedStructure(
            c_b=jnp.array([[0, 0, 0, 1], [0, 1, 0, 1]], dtype=jnp.int32),
            activity=jnp.ones((2, 4), dtype=jnp.int32),
        ),
    )


def test_valid_carrier_state():
    validate_carrier_state(make_state())


def test_terminal_boundary_required():
    state = make_state()
    bad = CarrierState(
        h=state.h,
        observed=state.observed,
        committed=CommittedStructure(
            c_b=jnp.zeros((2, 4), dtype=jnp.int32),
            activity=state.committed.activity,
        ),
    )
    with pytest.raises(ValueError):
        validate_carrier_state(bad)
