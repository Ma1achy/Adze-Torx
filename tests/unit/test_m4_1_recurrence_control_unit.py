import jax
import jax.numpy as jnp

from adze_t.model.core import (
    RecurrentCoreConfig,
    deterministic_core,
    effective_linear_map,
    initialise_params,
    nominal_accumulated_variance,
    per_step_variance,
)


def test_identity_initialization_is_q_invariant():
    state = jnp.linspace(-1.0, 1.0, 6)
    outputs = []
    for family in ("identity_residual", "q_normalized_residual"):
        for q in (1, 2, 4):
            config = RecurrentCoreConfig(width=6, q=q, family=family, eta=0.25)
            params = initialise_params(config, jax.random.key(10 + q))
            outputs.append(deterministic_core(config, params, state))
            assert float(jnp.linalg.norm(outputs[-1] - state)) < 1e-6
            assert float(jnp.linalg.norm(effective_linear_map(config, params) - jnp.eye(6))) < 1e-6
    assert all(bool(jnp.array_equal(outputs[0], output)) for output in outputs[1:])


def test_q_normalized_step_scale_is_eta_over_q():
    from adze_t.model.core import _gate_params

    config = RecurrentCoreConfig(width=2, q=4, family="q_normalized_residual", eta=0.8)
    params = {"A": jnp.eye(2), "b": jnp.ones(2), "log_var": jnp.zeros(2)}
    effective = _gate_params(config, params, 0)
    expected = jnp.eye(2) + 0.2 * jnp.eye(2)
    assert bool(jnp.array_equal(effective["A"], expected))
    assert bool(jnp.array_equal(effective["b"], jnp.full((2,), 0.2)))


def test_noise_is_independent_of_residual_eta_and_uses_variance_scaling():
    config_a = RecurrentCoreConfig(
        width=2, q=4, family="identity_residual", eta=0.1, total_variance=0.8
    )
    config_b = RecurrentCoreConfig(
        width=2, q=4, family="identity_residual", eta=0.9, total_variance=0.8
    )
    assert per_step_variance(config_a) == per_step_variance(config_b) == 0.2
    assert nominal_accumulated_variance(config_a) == nominal_accumulated_variance(config_b) == 0.8
