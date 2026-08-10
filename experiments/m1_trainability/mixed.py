"""Public-Torx mixed forward counterexample and independent oracle."""

from __future__ import annotations

import jax.numpy as jnp
from torx.psc import PNOT, HybridPCircuit, MixtureGaussianGate

from .oracles import mixed_value_and_grad


def build(depth: int = 1) -> HybridPCircuit:
    return HybridPCircuit(
        [
            PNOT(0),
            MixtureGaussianGate(sites=([0], [0]), dims=(1,), num_components=2),
        ],
        reps=depth,
    )


def forward_sample(depth: int, params, key):
    circuit = build(depth)
    inputs = {
        "discrete": jnp.array([0], dtype=circuit.gates[0].input_ports["in"].dtype),
        "continuous": jnp.array([0.0], dtype=circuit.input_ports["continuous"].dtype),
    }
    return circuit.sample(key, inputs, params)


def oracle(depth: int, params):
    return mixed_value_and_grad(params, depth)


def public_gradient_route():
    return (
        "No unified public mixed stochastic-gradient simulator found: "
        "BranchingSimulator accepts DiscretePCircuit, while "
        "AffineGaussianSimulator accepts only affine-Gaussian HybridPCircuit gates."
    )
