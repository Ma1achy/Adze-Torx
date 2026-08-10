import jax.numpy as jnp

from experiments.m1_trainability.continuous import run


def test_affine_gaussian_public_moment_route_matches_analytic_oracle():
    value, grad, oracle_value, oracle_grad = run()
    assert jnp.allclose(value, oracle_value)
    observed = jnp.array([grad["A"][0, 0], grad["b"][0], grad["log_var"][0]])
    assert jnp.allclose(observed, oracle_grad)
