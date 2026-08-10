import jax
import jax.numpy as jnp
import pytest
from torx.psc import PNOT

from adze_t.train.score_bridge import score_corrected_loss
from experiments.m1_trainability.discrete import (
    estimate,
    make_keys,
    manual_score_estimate,
)
from experiments.m1_trainability.mixed import (
    estimate_bridge,
    estimate_uncorrected,
)
from experiments.m1_trainability.oracles import markov_objective, mixed_value_and_grad


def _summary(values, exact):
    values = jnp.stack(values)
    mean = jnp.mean(values, axis=0)
    std = jnp.std(values, axis=0, ddof=1)
    stderr = std / jnp.sqrt(values.shape[0])
    z = (mean - exact) / stderr
    return mean, std, stderr, z


@pytest.mark.slow
def test_mixed_bridge_matches_oracle_and_preserves_continuous_gradients():
    params = jnp.array([0.2, 0.8, 0.4, -1.0])
    exact = mixed_value_and_grad(params, depth=4)[1]
    before = [estimate_uncorrected(params, make_keys(seed, 4096), 4)[1] for seed in range(16)]
    after = [estimate_bridge(params, make_keys(seed, 4096), 4)[1] for seed in range(16)]
    before_mean, _, _, _ = _summary(before, exact)
    after_mean, _, _, after_z = _summary(after, exact)
    assert bool(jnp.abs(before_mean[0]) < 1e-12)
    assert bool(jnp.all(jnp.abs(after_z) <= 4.0))
    assert bool(jnp.allclose(after_mean[1:], before_mean[1:], rtol=0.0, atol=1e-12))


@pytest.mark.slow
def test_bridge_recurrence_depths_match_exact_mixed_oracle():
    params = jnp.array([0.2, 0.8, 0.4, -1.0])
    for depth in (1, 2, 4, 8, 16, 32):
        exact = mixed_value_and_grad(params, depth)[1]
        estimates = [estimate_bridge(params, make_keys(seed, 4096), depth)[1] for seed in range(8)]
        mean, _, stderr, z = _summary(estimates, exact)
        accepted = (stderr == 0) & (mean == exact) | (jnp.abs(z) <= 4.0)
        assert bool(jnp.all(accepted)), (depth, z)


@pytest.mark.slow
def test_discrete_bridge_matches_exact_and_native_torx_route():
    theta = 0.35
    depth = 4
    exact_grad = jax.grad(lambda value: markov_objective(value, depth))(theta)
    bridge = [manual_score_estimate(theta, make_keys(seed, 4096), depth)[1] for seed in range(16)]
    native = [estimate(depth, theta, jax.random.key(seed), 4096)[1] for seed in range(16)]
    bridge_mean, _, _, bridge_z = _summary(bridge, exact_grad)
    native_mean, _, _, native_z = _summary(native, exact_grad)
    assert bool(jnp.abs(bridge_z) <= 4.0)
    assert bool(jnp.abs(native_z) <= 4.0)
    assert bool(
        jnp.abs(bridge_mean - native_mean)
        <= 4.0
        * jnp.sqrt(
            jnp.var(jnp.stack(bridge), ddof=1) / 16 + jnp.var(jnp.stack(native), ddof=1) / 16
        )
    )


@pytest.mark.slow
def test_omitting_one_recurrent_log_probability_fails_oracle_comparison():
    gate = PNOT(0)
    theta = jnp.asarray(0.35, dtype=jnp.float64)
    depth = 2

    def mutated_loss(key):
        state = jnp.zeros((1,), dtype=gate.input_ports["in"].dtype)
        log_prob_sum = jnp.asarray(0.0, dtype=theta.dtype)
        for step, step_key in enumerate(jax.random.split(key, depth)):
            previous_state = state
            state = gate.sample(step_key, {"in": previous_state}, jnp.reshape(theta, (1,)))
            if step != 0:  # test-only mutation: intentionally omit one occurrence
                log_prob_sum = log_prob_sum + gate.log_probability(
                    {"in": previous_state}, state, jnp.reshape(theta, (1,))
                )
        return score_corrected_loss(state[0].astype(theta.dtype), log_prob_sum)

    def gradient_for_key(key):
        return jax.grad(lambda value: mutated_loss(key))(theta)

    estimates = jax.vmap(gradient_for_key)(make_keys(0, 4096))
    exact = jax.grad(lambda value: markov_objective(value, depth))(theta)
    stderr = jnp.std(estimates, ddof=1) / jnp.sqrt(estimates.shape[0])
    assert bool(jnp.abs(jnp.mean(estimates) - exact) > 4.0 * stderr)
