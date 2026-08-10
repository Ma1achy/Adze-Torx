import jax
import jax.numpy as jnp
import pytest
from torx.psc import PNOT

from adze_t.train.score_bridge import score_corrected_loss
from experiments.m1_trainability.mixed import sample_trajectory


def test_score_corrected_loss_preserves_scalar_primal_exactly():
    loss = jnp.asarray(2.5)
    log_prob = jnp.asarray(-1.2)
    corrected = score_corrected_loss(loss, log_prob)
    assert bool(jnp.array_equal(corrected, loss))


def test_score_corrected_loss_preserves_per_trajectory_primal_exactly():
    loss = jnp.array([1.0, 2.0, 3.0])
    log_prob = jnp.array([-0.1, -0.2, -0.3])
    corrected = score_corrected_loss(loss, log_prob)
    assert bool(jnp.array_equal(corrected, loss))


def test_score_corrected_loss_rejects_ambiguous_shapes():
    with pytest.raises(ValueError, match="same per-trajectory shape"):
        score_corrected_loss(jnp.asarray(1.0), jnp.ones(3))
    with pytest.raises(ValueError, match="same per-trajectory shape"):
        score_corrected_loss(jnp.ones(3), jnp.ones((3, 1)))


def test_score_correction_is_applied_per_trajectory_before_reduction():
    losses = jnp.array([1.0, 2.0])

    def batch_loss(theta):
        log_probs = theta * jnp.array([1.0, 2.0])
        return jnp.mean(score_corrected_loss(losses, log_probs))

    assert jnp.allclose(jax.grad(batch_loss)(0.0), 2.5)


def test_constant_loss_score_is_zero_only_in_expectation_not_per_sample():
    loss = jnp.asarray(3.0)
    gradient = jax.grad(lambda theta: score_corrected_loss(loss, theta))(0.0)
    assert gradient == 3.0


def test_constant_or_disconnected_loss_has_zero_expected_score_contribution():
    gate = PNOT(0)
    theta = jnp.asarray(0.35)
    theta_param = jnp.reshape(theta, (1,))
    inputs = {"in": jnp.zeros((1,), dtype=gate.input_ports["in"].dtype)}
    expected = jnp.asarray(0.0)
    for state_index in (0, 1):
        output = gate.get_nth_output_state(state_index)
        log_prob = gate.log_probability(inputs, output, theta_param)
        probability = jnp.exp(log_prob)

        def score_for_value(value, selected_output=output):
            log_prob = gate.log_probability(inputs, selected_output, jnp.reshape(value, (1,)))
            return score_corrected_loss(jnp.asarray(3.0), log_prob)

        contribution = jax.grad(score_for_value)(theta)
        expected = expected + probability * contribution
    assert jnp.allclose(expected, 0.0)


def test_mixed_trajectory_is_finite_with_accumulated_log_probability():
    params = jnp.array([0.2, 0.8, 0.4, -1.0])
    trajectory = sample_trajectory(params, jax.random.key(3), depth=4)
    assert bool(jnp.all(jnp.isfinite(trajectory.loss)))
    assert bool(jnp.all(jnp.isfinite(trajectory.log_prob_sum)))
