from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from adze_t.phase_g_oracle import (
    ORACLE_ACTION_COUNT,
    descendant_fixture,
    enumerate_actions,
    enumerated_bridge_gradient,
    exact_value_and_gradient,
    fixture_parameters,
    monte_carlo_conformance,
)


def _allclose(left, right, *, atol=1.0e-6):
    return jax.tree_util.tree_all(
        jax.tree.map(lambda x, y: jnp.allclose(x, y, rtol=1.0e-6, atol=atol), left, right)
    )


def test_oracle_enumerates_exactly_108_actions_with_forced_terminal():
    actions = enumerate_actions()
    assert actions.boundary.shape == (ORACLE_ACTION_COUNT, 3)
    assert actions.extent.shape == (ORACLE_ACTION_COUNT, 3)
    assert jnp.all(actions.boundary[:, -1] == 1)
    assert (
        len(
            {
                (tuple(boundary), tuple(extent))
                for boundary, extent in zip(
                    actions.boundary.tolist(), actions.extent.tolist(), strict=True
                )
            }
        )
        == 108
    )


@pytest.mark.parametrize("parameter_fixture", ["uniform", "asymmetric", "peaked"])
@pytest.mark.parametrize(
    "descendant_name",
    ["inactive", "reactivation", "zero", "missing", "extra", "direct_dependencies", "baseline"],
)
def test_enumerated_local_bridge_matches_independent_exact_gradient(
    parameter_fixture, descendant_name
):
    parameters = fixture_parameters(parameter_fixture)
    descendant = descendant_fixture(descendant_name)
    _, exact = exact_value_and_gradient(parameters, descendant)
    bridge = enumerated_bridge_gradient(parameters, descendant)
    assert _allclose(bridge, exact, atol=2.0e-6)


def test_omitted_and_duplicated_occurrence_mutations_fail_exact_oracle():
    parameters = fixture_parameters("asymmetric")
    descendant = descendant_fixture("direct_dependencies")
    _, exact = exact_value_and_gradient(parameters, descendant)
    omitted = enumerated_bridge_gradient(parameters, descendant, mutation="omit_first_boundary")
    duplicated = enumerated_bridge_gradient(parameters, descendant, mutation="duplicate_all")
    assert not _allclose(omitted, exact, atol=1.0e-4)
    assert not _allclose(duplicated, exact, atol=1.0e-4)


@pytest.mark.slow
@pytest.mark.parametrize("parameter_fixture", ["uniform", "asymmetric", "peaked"])
@pytest.mark.parametrize(
    "descendant_name",
    ["inactive", "reactivation", "zero", "missing", "extra", "direct_dependencies", "baseline"],
)
def test_monte_carlo_bridge_is_within_four_root_standard_errors(parameter_fixture, descendant_name):
    result = monte_carlo_conformance(
        fixture_parameters(parameter_fixture),
        descendant_fixture(descendant_name),
        root_seed=8100,
    )
    assert bool(result["accepted"]), result
