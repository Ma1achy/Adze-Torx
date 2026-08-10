import jax
import jax.numpy as jnp
import pytest

from experiments.m1_trainability.discrete import estimate
from experiments.m1_trainability.oracles import markov_value_and_grad


@pytest.mark.slow
def test_discrete_gradient_conforms_within_measured_uncertainty():
    exact = float(markov_value_and_grad(0.35, 4)[1])
    values = jnp.array([estimate(4, 0.35, jax.random.key(seed), 4096)[1] for seed in range(16)])
    mean = float(jnp.mean(values))
    stderr = float(jnp.std(values, ddof=1) / jnp.sqrt(values.shape[0]))
    assert abs(mean - exact) <= 4.0 * stderr
