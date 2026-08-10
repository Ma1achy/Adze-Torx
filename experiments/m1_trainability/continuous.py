"""Public-Torx affine-Gaussian experiment."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from torx.psc import AffineGaussianGate, AffineGaussianSimulator, HybridPCircuit

from .oracles import gaussian_value_and_grad


def build():
    return HybridPCircuit([AffineGaussianGate(sites=[0], dims=(1,))])


def exact_torx(params):
    circuit = build()
    simulator = AffineGaussianSimulator()
    compiled = simulator.build_circuit(circuit, [params])
    moments = simulator.propagate(compiled, jnp.array([0.0], dtype=jnp.float64))
    mean = moments.mean[0]
    variance = moments.covariance[0, 0]
    return variance + (mean - 0.7) ** 2


def run(params=None):
    params = (
        {
            "A": jnp.array([[0.8]], dtype=jnp.float64),
            "b": jnp.array([0.25], dtype=jnp.float64),
            "log_var": jnp.array([-1.0], dtype=jnp.float64),
        }
        if params is None
        else params
    )
    # The public exact simulator differentiates through its moment calculation;
    # this is the supported route, not a sampled pathwise claim.
    value, grad = jax.value_and_grad(exact_torx)(params)
    oracle_params = jnp.array([params["A"][0, 0], params["b"][0], params["log_var"][0]])
    oracle_value, oracle_grad = gaussian_value_and_grad(oracle_params)
    return value, grad, oracle_value, oracle_grad
