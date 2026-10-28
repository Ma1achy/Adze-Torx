"""Independent G21 analytic/statistical checks, not neural trainability evidence."""

import math

import jax.numpy as jnp
import numpy as np
import pytest

from adze_t.phase_g21 import (
    actions,
    bernstein_radius,
    finite_difference,
    from_vector,
    pack_parameters,
    route,
    tolerance,
    wilson,
)
from adze_t.phase_g_oracle import fixture_parameters, OracleActions
from adze_t.phase_g2_oracle import descendant
from adze_t.phase_g2 import named_key


def test_bernstein_radius_solves_exponent_and_handles_zero_variance():
    v, bound, n = np.array([0.0, 0.02, 3.0]), np.array([0.0, 20.0, 100.0]), 262144
    radius = bernstein_radius(v, bound, n)
    assert radius[0] == 0
    exponent = n * radius[1:] ** 2 / (2 * v[1:] + 2 * bound[1:] * radius[1:] / 3)
    np.testing.assert_allclose(exponent, math.log(2 * 42 / 0.01), atol=1e-12)
    assert 2 * 42 * np.exp(-exponent[0]) <= 0.010000000000001
    with pytest.raises(ValueError):
        bernstein_radius(-1, 1, n)
    with pytest.raises(ValueError):
        bernstein_radius(1, 1, 0)


@pytest.mark.parametrize("fixture", ["uniform", "asymmetric", "peaked"])
def test_independent_actions_loss_and_analytic_gradient(fixture):
    p = fixture_parameters(fixture)
    table = from_vector(pack_parameters(p))
    assert table["gradient"].shape == (108, 14)
    assert len(np.unique(actions(), axis=0)) == 108
    assert np.isfinite(table["loss"]).all()
    fd = finite_difference(pack_parameters(p))
    np.testing.assert_allclose(fd, table["mean"], atol=2e-8, rtol=1e-4)
    routing = jnp.asarray(np.stack([route(row) for row in actions()]))
    production = np.asarray(descendant(p, OracleActions(routing, jnp.asarray(actions()[:, 2:]))))
    assert np.all(np.abs(production - table["loss"]) <= tolerance(table["loss"]))
    # Action-independent score baselines have zero expected contribution.
    np.testing.assert_allclose(table["probability"] @ table["score"], 0, atol=1e-12)
    # Mechanically forced middle cut is not a new stochastic action.
    assert route(np.array([0, 0, 0, 0, 0])).tolist() == [0, 1, 1]


def test_joint_family_bound_does_not_studentize_away_rare_events():
    table = from_vector(pack_parameters(fixture_parameters("peaked")))
    component = 6  # extent site0/class2, not a cherry-picked new acceptance rule
    n = 65536
    assert table["variance"][component] > 0.01
    assert np.sqrt(table["variance"][component] / n) > 0.0004
    assert bernstein_radius(table["variance"], table["maximum"], n)[component] > 0.001
    # A constant error beyond the fixed region is rejected, even with zero
    # observed root variance; the region never uses empirical root variance.
    radius = bernstein_radius(table["variance"], table["maximum"], 262144)
    shift = radius + tolerance(table["mean"]) + 0.1
    assert np.all(shift > radius + tolerance(table["mean"]))


def test_wilson_support_and_multinomial_count_conservation():
    from adjudicate_phase_g21 import count_draws

    assert wilson(0, 2048)[1] < 0.002
    with pytest.raises(ValueError):
        wilson(2, 1)
    conditional = jnp.ones(108).at[1:].set(0.0)
    counts = np.asarray(count_draws(named_key("g21_unit_count"), conditional, trials=32, roots=1))
    assert counts.shape == (256, 1, 108)
    assert np.all(counts.sum(-1) == 32)
    assert np.all(counts[:, :, 0] == 32) and np.all(counts[:, :, 1:] == 0)
