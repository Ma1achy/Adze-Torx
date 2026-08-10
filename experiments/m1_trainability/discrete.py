"""Public-Torx discrete recurrence experiment."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from torx.psc import PNOT, BranchingSimulator, DiscretePCircuit, StateVectorSimulator

from adze_t.train.score_bridge import score_corrected_loss

from .oracles import markov_value_and_grad


def build(depth: int) -> DiscretePCircuit:
    return DiscretePCircuit([PNOT(0)], reps=depth)


def exact(depth: int, theta: float):
    simulator = StateVectorSimulator()
    circuit = build(depth)
    initial = jnp.array([1.0, 0.0], dtype=jnp.float64)

    def objective(thetas):
        compiled = simulator.build_circuit(circuit, thetas)
        return simulator.expval_all(compiled, initial)[0]

    value, grad = jax.value_and_grad(objective)([jnp.array([theta], dtype=jnp.float64)])
    return value, grad[0][0]


def estimate(
    depth: int,
    theta: float,
    key,
    num_samples: int = 4096,
    method: str = "param_shift_filter",
):
    simulator = BranchingSimulator(diff_method=method, num_samples=num_samples)
    circuit = build(depth)
    initial = jnp.array([0], dtype=jnp.int32)

    def objective(thetas, sample_key):
        compiled = simulator.build_circuit(circuit, thetas)
        return simulator.expval(compiled, initial, 0, sample_key)

    value, grad = jax.value_and_grad(objective)([jnp.array([theta], dtype=jnp.float64)], key)
    return value, grad[0][0]


def manual_score_corrected_loss(theta: jax.Array, key, depth: int) -> jax.Array:
    """Sample a PNOT recurrence and add its per-trajectory score correction."""
    gate = PNOT(0)
    theta_param = jnp.reshape(theta, (1,))
    state = jnp.zeros((1,), dtype=gate.input_ports["in"].dtype)
    log_prob_sum = jnp.asarray(0.0, dtype=theta.dtype)
    keys = jax.random.split(key, depth)
    for step_key in keys:
        previous_state = state
        state = gate.sample(step_key, {"in": previous_state}, theta_param)
        log_prob_sum = log_prob_sum + gate.log_probability(
            {"in": previous_state}, state, theta_param
        )
    loss = state[0].astype(theta.dtype)
    return score_corrected_loss(loss, log_prob_sum)


def manual_score_estimate(theta: float, keys: jax.Array, depth: int):
    theta_array = jnp.asarray(theta, dtype=jnp.float64)

    def batch_loss(t):
        values = jax.vmap(lambda key: manual_score_corrected_loss(t, key, depth))(keys)
        return jnp.mean(values)

    return jax.value_and_grad(batch_loss)(theta_array)


def make_keys(seed: int, n_samples: int) -> jax.Array:
    return jax.random.split(jax.random.key(seed), n_samples)


def run(depths=(1, 2, 4, 8, 16, 32), theta=0.35, samples=4096, seeds=8):
    rows = []
    for depth in depths:
        exact_value, exact_grad = markov_value_and_grad(theta, depth)
        estimates = jnp.array(
            [estimate(depth, theta, jax.random.key(seed), samples)[1] for seed in range(seeds)]
        )
        rows.append(
            {
                "depth": depth,
                "exact_objective": float(exact_value),
                "exact_gradient": float(exact_grad),
                "estimated_objective": float(estimate(depth, theta, jax.random.key(0), samples)[0]),
                "gradient_mean": float(jnp.mean(estimates)),
                "gradient_std": float(jnp.std(estimates, ddof=1)),
                "gradient_stderr": float(jnp.std(estimates, ddof=1) / jnp.sqrt(seeds)),
                "samples": samples,
            }
        )
    return rows
