import jax
import jax.numpy as jnp
import pytest

from adze_t.model.core import (
    RecurrentCoreConfig,
    deterministic_core,
    nominal_accumulated_variance,
    per_step_variance,
)


def test_fixed_total_noise_scales_variance_not_standard_deviation():
    config = RecurrentCoreConfig(width=4, q=4, total_variance=0.8, noise_mode="fixed_total")
    assert per_step_variance(config) == pytest.approx(0.2)
    assert nominal_accumulated_variance(config) == pytest.approx(0.8)


def test_fixed_per_cycle_noise_accumulates_nominal_variance():
    config = RecurrentCoreConfig(width=4, q=4, total_variance=0.2, noise_mode="fixed_per_cycle")
    assert per_step_variance(config) == pytest.approx(0.2)
    assert nominal_accumulated_variance(config) == pytest.approx(0.8)


def test_residual_eta_zero_is_identity_mean_dynamics():
    config = RecurrentCoreConfig(width=3, q=4, family="residual", eta=0.0)
    from adze_t.model.core import initialise_params

    params = initialise_params(config, jax.random.key(1))
    state = jnp.arange(3.0)
    observed = deterministic_core(config, params, state)
    assert bool(jnp.array_equal(observed, state))


def test_q_one_residual_returns_one_step_trajectory():
    config = RecurrentCoreConfig(width=3, q=1, family="residual", eta=0.25)
    from adze_t.model.core import apply_core, initialise_params

    params = initialise_params(config, jax.random.key(2))
    trajectory = apply_core(config, params, jnp.ones(3), jax.random.key(3), True)
    assert trajectory.shape == (2, 3)


def test_invalid_recurrent_configuration_is_rejected():
    with pytest.raises(ValueError):
        RecurrentCoreConfig(width=3, q=0)
    with pytest.raises(ValueError):
        RecurrentCoreConfig(width=3, family="other")
