import jax.numpy as jnp

from experiments.m1_trainability.discrete import exact
from experiments.m1_trainability.oracles import markov_value_and_grad


def test_discrete_torx_exact_route_matches_markov_oracle():
    value, grad = exact(4, 0.35)
    oracle_value, oracle_grad = markov_value_and_grad(0.35, 4)
    assert jnp.allclose(value, oracle_value)
    assert jnp.allclose(grad, oracle_grad)


def test_markov_oracle_is_finite_near_saturation():
    for theta in (-30.0, 30.0):
        value, grad = markov_value_and_grad(theta, 32)
        assert bool(jnp.isfinite(value) & jnp.isfinite(grad))
