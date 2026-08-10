"""Independent analytic and exact oracles for the M1 spike."""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp


def binary_kernel(theta: jax.Array) -> jax.Array:
    """Row-stochastic binary flip kernel, independent of Torx."""
    p = jax.nn.sigmoid(theta)
    return jnp.array([[1.0 - p, p], [p, 1.0 - p]])


def markov_objective(theta: jax.Array, depth: int, pi0=None, cost=None) -> jax.Array:
    pi0 = jnp.array([1.0, 0.0]) if pi0 is None else jnp.asarray(pi0)
    cost = jnp.array([0.0, 1.0]) if cost is None else jnp.asarray(cost)
    return pi0 @ jnp.linalg.matrix_power(binary_kernel(theta), depth) @ cost


def markov_value_and_grad(theta: float, depth: int):
    def fn(t):
        return markov_objective(t, depth)

    value, grad = jax.value_and_grad(fn)(jnp.asarray(theta, dtype=jnp.float64))
    return value, grad


def gaussian_objective(params: jax.Array, depth: int = 1) -> jax.Array:
    """E[(H_T-c)^2] for H_{t+1}=alpha H_t+beta+sigma*eps."""
    alpha, beta, log_var = params
    mean = jnp.array(0.0)
    variance = jnp.array(0.0)
    for _ in range(depth):
        mean = alpha * mean + beta
        variance = alpha**2 * variance + jnp.exp(log_var)
    return variance + (mean - 0.7) ** 2


def gaussian_value_and_grad(params, depth: int = 1):
    return jax.value_and_grad(lambda p: gaussian_objective(p, depth))(params)


class MixedMoments(NamedTuple):
    probability: jax.Array
    conditional_mean: jax.Array
    conditional_second: jax.Array


def mixed_objective(params: jax.Array, depth: int = 1) -> jax.Array:
    """Exact X/H conditional-moment recursion for the mixed toy model."""
    theta, alpha, beta, log_var = params
    p = jax.nn.sigmoid(theta)
    transition = jnp.array([[1.0 - p, p], [p, 1.0 - p]])
    prob = jnp.array([1.0, 0.0])
    mean = jnp.zeros(2)
    second = jnp.zeros(2)
    for _ in range(depth):
        next_prob = prob @ transition
        next_mean = jnp.zeros(2)
        next_second = jnp.zeros(2)
        for new_x in range(2):
            transition_weights = transition[:, new_x]
            # H' = alpha H + beta * X' + sigma eps.
            shifted_mean = alpha * mean + beta * new_x * prob
            shifted_second = (
                alpha**2 * second
                + 2.0 * alpha * beta * new_x * mean
                + ((beta * new_x) ** 2 + jnp.exp(log_var)) * prob
            )
            next_mean = next_mean.at[new_x].set(jnp.sum(transition_weights * shifted_mean))
            next_second = next_second.at[new_x].set(jnp.sum(transition_weights * shifted_second))
        prob, mean, second = next_prob, next_mean, next_second
    h_mean = jnp.sum(mean)
    h_second = jnp.sum(second)
    return 0.4 * prob[1] + 0.3 * h_mean + 0.8 * h_second


def mixed_value_and_grad(params, depth: int = 1):
    return jax.value_and_grad(lambda p: mixed_objective(p, depth))(params)
