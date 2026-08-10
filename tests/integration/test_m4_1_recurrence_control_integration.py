import jax
import jax.numpy as jnp

from adze_t.model.core import RecurrentCoreConfig, apply_core, deterministic_core, initialise_params
from experiments.m4_1_recurrence_control.diagnostics import initial_map_report


def test_m4_1_public_torx_variants_are_finite_and_fixed_shape():
    for family in ("identity_residual", "q_normalized_residual"):
        for q in (1, 2, 4):
            config = RecurrentCoreConfig(width=6, q=q, family=family, eta=0.25)
            params = initialise_params(config, jax.random.key(20 + q))
            state = jnp.arange(6.0)
            sampled = apply_core(config, params, state, jax.random.key(30 + q))
            mean = deterministic_core(config, params, state)
            assert sampled.shape == mean.shape == state.shape
            assert bool(jnp.all(jnp.isfinite(sampled)))
            assert bool(jnp.all(jnp.isfinite(mean)))
            report = initial_map_report(config, jax.random.key(40 + q))
            assert report["state_error"] < 1e-6
            assert report["linear_error"] < 1e-6


def test_m4_1_fixed_seed_reproduces_public_sampling():
    config = RecurrentCoreConfig(width=4, q=4, family="q_normalized_residual")
    params = initialise_params(config, jax.random.key(50))
    state = jnp.ones(4)
    first = apply_core(config, params, state, jax.random.key(51))
    second = apply_core(config, params, state, jax.random.key(51))
    different = apply_core(config, params, state, jax.random.key(52))
    assert bool(jnp.array_equal(first, second))
    assert not bool(jnp.array_equal(first, different))
