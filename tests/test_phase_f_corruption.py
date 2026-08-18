"""Contracts for the standalone Phase-F V0 corruption substrate."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from adze_t.backends.torx import TorxOperatorConfig, TorxOps
from adze_t.corruption import (
    DiffusionEtaMode,
    DiffusionKeyContext,
    DiffusionStage,
    alpha,
    corrupt_h,
    diffusion_key,
    phase_f_schedule,
    recorrupt_h,
    sample_initial_corruption,
    sample_recorruption,
    sigma,
)


@pytest.mark.parametrize("nu", [0.0, 0.25, 0.5, 0.75, 1.0])
def test_alpha_sigma_exact_law_and_unit_energy(nu):
    expected_alpha = jnp.cos(jnp.pi * jnp.asarray(nu) / 2.0)
    expected_sigma = jnp.sin(jnp.pi * jnp.asarray(nu) / 2.0)
    assert jnp.allclose(alpha(nu), expected_alpha, rtol=0.0, atol=1.0e-7)
    assert jnp.allclose(sigma(nu), expected_sigma, rtol=0.0, atol=1.0e-7)
    assert jnp.allclose(alpha(nu) ** 2 + sigma(nu) ** 2, 1.0, rtol=0.0, atol=1.0e-7)


def test_alpha_sigma_endpoints_arrays_jit_and_invalid_levels():
    levels = jnp.asarray([0.0, 0.25, 0.5, 0.75, 1.0], dtype=jnp.float32)
    actual_alpha, actual_sigma = jax.jit(lambda x: (alpha(x), sigma(x)))(levels)
    assert actual_alpha[0] == 1.0
    assert jnp.abs(actual_alpha[-1]) < 1.0e-7
    assert actual_sigma[0] == 0.0
    assert actual_sigma[-1] == 1.0
    with pytest.raises(ValueError, match="nu must lie"):
        alpha(-0.01)
    with pytest.raises(ValueError, match="nu must lie"):
        sigma(1.01)
    with pytest.raises(ValueError, match="nu must be finite"):
        alpha(jnp.nan)


def test_forward_corruption_matches_independent_reference_and_broadcasts_batch_levels():
    h0 = jnp.arange(24, dtype=jnp.float32).reshape(2, 3, 4) / 10.0
    epsilon = jnp.flip(h0, axis=-1) - 0.3
    nu = jnp.asarray([0.25, 0.75], dtype=jnp.float32)
    angle = jnp.pi * nu[:, None, None] / 2.0
    expected = jnp.cos(angle) * h0 + jnp.sin(angle) * epsilon
    actual = jax.jit(corrupt_h)(h0, nu, epsilon)
    assert jnp.allclose(actual, expected, rtol=0.0, atol=1.0e-7)


def test_corruption_has_no_hidden_dimension_or_compute_scaling():
    h0 = jnp.ones((2, 5, 7), dtype=jnp.float32)
    epsilon = jnp.full_like(h0, 2.0)
    nu = 0.5
    expected_scalar = jnp.cos(jnp.pi / 4) + 2.0 * jnp.sin(jnp.pi / 4)
    assert jnp.allclose(corrupt_h(h0, nu, epsilon), expected_scalar, atol=1.0e-7)


def test_corruption_rejects_bad_carrier_shapes_and_dtypes():
    with pytest.raises(ValueError, match="identical shapes"):
        corrupt_h(jnp.ones((2, 3)), 0.5, jnp.ones((2, 1)))
    with pytest.raises(TypeError, match="floating dtype"):
        corrupt_h(jnp.ones((2, 3), dtype=jnp.int32), 0.5, jnp.ones((2, 3)))


def test_recorruption_eta_zero_is_exact_mean_and_eta_one_is_full_forward_kernel():
    h_hat = jnp.arange(12, dtype=jnp.float32).reshape(3, 4) / 7.0
    epsilon_a = jnp.ones_like(h_hat) * 3.0
    epsilon_b = jnp.ones_like(h_hat) * -9.0
    nu = 0.6
    mean = alpha(nu) * h_hat
    eta_zero_a = recorrupt_h(h_hat, nu, epsilon_a, DiffusionEtaMode.DETERMINISTIC_MEAN)
    eta_zero_b = recorrupt_h(h_hat, nu, epsilon_b, 0)
    assert jnp.array_equal(eta_zero_a, mean)
    assert jnp.array_equal(eta_zero_a, eta_zero_b)
    assert jnp.allclose(
        recorrupt_h(h_hat, nu, epsilon_a, DiffusionEtaMode.FULL_RENOISE),
        mean + jnp.sin(jnp.pi * nu / 2.0) * epsilon_a,
        rtol=0.0,
        atol=1.0e-7,
    )
    with pytest.raises(ValueError, match="eta_diff only"):
        recorrupt_h(h_hat, nu, epsilon_a, 0.5)


def test_diffusion_key_identity_and_namespace_contract():
    root = jax.random.key(71)
    context = DiffusionKeyContext(root, global_example_id=19)
    initial = context.key_for(DiffusionStage.INITIAL_CORRUPTION, denoise_step=0)
    repeated = diffusion_key(
        root,
        global_example_id=19,
        stage=DiffusionStage.INITIAL_CORRUPTION,
        denoise_step=0,
    )
    other_example = DiffusionKeyContext(root, 20).key_for(DiffusionStage.INITIAL_CORRUPTION, 0)
    other_stage = context.key_for(DiffusionStage.RECORRUPTION, 0)
    other_step = context.key_for(DiffusionStage.RECORRUPTION, 1)
    assert jnp.array_equal(initial, repeated)
    assert not jnp.array_equal(initial, other_example)
    assert not jnp.array_equal(initial, other_stage)
    assert not jnp.array_equal(other_stage, other_step)

    operator_key = TorxOps.create(
        root,
        config=TorxOperatorConfig(operator_stochasticity=True, lambda_op=1.0),
        global_example_id=19,
    ).context.key_for("phase_f.representative_operator")
    assert not jnp.array_equal(initial, operator_key)


def test_diffusion_prefix_randomness_does_not_depend_on_total_s():
    context = DiffusionKeyContext(jax.random.key(72), global_example_id=5)

    def keys_for_execution(total_s):
        return [
            context.key_for(DiffusionStage.RECORRUPTION, denoise_step=step)
            for step in range(1, total_s)
        ]

    keys_s2 = keys_for_execution(2)
    keys_s3 = keys_for_execution(3)
    keys_s4 = keys_for_execution(4)
    assert jnp.array_equal(keys_s2[0], keys_s3[0])
    assert jnp.array_equal(keys_s2[0], keys_s4[0])
    assert jnp.array_equal(keys_s3[1], keys_s4[1])
    assert not jnp.array_equal(keys_s4[0], keys_s4[1])


def test_sampling_helpers_are_reproducible_and_expose_epsilon():
    h0 = jnp.arange(12, dtype=jnp.float32).reshape(3, 4)
    key = DiffusionKeyContext(jax.random.key(73), 4).key_for(DiffusionStage.INITIAL_CORRUPTION, 0)
    first = sample_initial_corruption(h0, 0.5, key)
    second = sample_initial_corruption(h0, 0.5, key)
    assert jnp.array_equal(first.value, second.value)
    assert jnp.array_equal(first.epsilon, second.epsilon)
    assert jnp.array_equal(first.value, corrupt_h(h0, 0.5, first.epsilon))

    other = sample_recorruption(h0, 0.25, jax.random.key(74), 1)
    assert jnp.array_equal(other.value, recorrupt_h(h0, 0.25, other.epsilon, 1))


def test_schedule_is_frozen_monotone_and_strictly_prefix_compatible():
    full = phase_f_schedule(0.8)
    assert jnp.allclose(full, jnp.asarray([0.8, 0.6, 0.4, 0.2]), atol=1.0e-7)
    assert jnp.all(jnp.diff(full) < 0)
    for steps in (1, 2, 3):
        assert jnp.array_equal(phase_f_schedule(0.8, steps), full[:steps])
    batched = jax.jit(phase_f_schedule)(jnp.asarray([0.4, 0.8]))
    assert batched.shape == (2, 4)
    with pytest.raises(ValueError, match="s_max"):
        phase_f_schedule(0.8, 0)
    with pytest.raises(ValueError, match="s_max"):
        phase_f_schedule(0.8, 5)


def test_fixed_epsilon_pathwise_gradients_match_independent_formulas():
    h0 = jnp.asarray([-0.4, 0.2, 0.9], dtype=jnp.float32)
    epsilon = jnp.asarray([0.5, -1.2, 0.3], dtype=jnp.float32)
    nu = jnp.asarray(0.37, dtype=jnp.float32)

    actual_h_grad = jax.grad(lambda h: jnp.sum(corrupt_h(h, nu, epsilon)))(h0)
    expected_h_grad = jnp.full_like(h0, jnp.cos(jnp.pi * nu / 2.0))
    assert jnp.allclose(actual_h_grad, expected_h_grad, rtol=0.0, atol=1.0e-7)

    actual_nu_grad = jax.grad(lambda level: jnp.sum(corrupt_h(h0, level, epsilon)))(nu)
    expected_nu_grad = jax.grad(
        lambda level: jnp.sum(
            jnp.cos(jnp.pi * level / 2.0) * h0 + jnp.sin(jnp.pi * level / 2.0) * epsilon
        )
    )(nu)
    assert jnp.allclose(actual_nu_grad, expected_nu_grad, rtol=0.0, atol=1.0e-7)

    for eta in (0, 1):
        actual = jax.grad(lambda h: jnp.sum(recorrupt_h(h, nu, epsilon, eta)))(h0)
        assert jnp.allclose(actual, expected_h_grad, rtol=0.0, atol=1.0e-7)


@pytest.mark.slow
def test_initial_and_recorruption_empirical_moments():
    sample_count = 8192
    h0 = jnp.asarray([0.7], dtype=jnp.float32)
    nu = 0.5
    keys = jax.random.split(jax.random.key(75), sample_count)
    initial = jax.vmap(lambda key: sample_initial_corruption(h0, nu, key).value[0])(keys)
    expected_mean = jnp.cos(jnp.pi / 4.0) * h0[0]
    expected_variance = jnp.sin(jnp.pi / 4.0) ** 2
    assert jnp.abs(jnp.mean(initial) - expected_mean) < 0.03
    assert jnp.abs(jnp.var(initial, ddof=1) - expected_variance) < 0.03

    renoised = jax.vmap(lambda key: sample_recorruption(h0, nu, key, 1).value[0])(keys)
    deterministic = jax.vmap(lambda key: sample_recorruption(h0, nu, key, 0).value[0])(keys)
    assert jnp.abs(jnp.mean(renoised) - expected_mean) < 0.03
    assert jnp.abs(jnp.var(renoised, ddof=1) - expected_variance) < 0.03
    assert jnp.all(deterministic == deterministic[0])
    assert jnp.var(deterministic, ddof=1) < 1.0e-12
