from typing import cast

import jax
import jax.numpy as jnp

from experiments.m1_trainability.mixed import forward_sample, oracle, public_gradient_route


def test_mixed_public_forward_is_defined_and_finite():
    params = [
        jnp.array([0.2]),
        {"means": jnp.array([[0.0], [0.4]]), "log_vars": jnp.array([[-1.0], [-1.0]])},
    ]
    result = cast(dict[str, jax.Array], forward_sample(2, params, jax.random.key(11)))
    assert set(result) == {"discrete", "continuous"}
    assert bool(jnp.all(jnp.isfinite(result["continuous"])))


def test_mixed_conditional_moment_oracle_is_finite():
    value, grad = oracle(4, jnp.array([0.2, 0.8, 0.4, -1.0]))
    assert bool(jnp.isfinite(value))
    assert bool(jnp.all(jnp.isfinite(grad)))


def test_mixed_route_inventory_is_explicit():
    assert "No unified public mixed" in public_gradient_route()


def test_plain_jax_mixed_sample_has_zero_discrete_gradient():
    params = [
        jnp.array([0.2]),
        {"means": jnp.array([[0.0], [0.4]]), "log_vars": jnp.array([[-1.0], [-1.0]])},
    ]
    circuit = __import__("experiments.m1_trainability.mixed", fromlist=["build"]).build(1)
    inputs = {
        "discrete": jnp.array([0], dtype=circuit.gates[0].input_ports["in"].dtype),
        "continuous": jnp.array([0.0], dtype=circuit.input_ports["continuous"].dtype),
    }

    def loss(p):
        return circuit.sample(jax.random.key(0), inputs, p)["continuous"][0]

    grad = jax.grad(loss)(params)
    assert grad[0][0] == 0.0
