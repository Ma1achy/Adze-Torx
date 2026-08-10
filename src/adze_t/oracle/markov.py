"""Finite-state Markov-chain oracles for M1 recurrence tests."""

from __future__ import annotations

import jax.numpy as jnp


def terminal_distribution(initial, transition, steps: int):
    """Compute pi_0 K^steps exactly in finite-state floating-point arithmetic."""
    if steps < 0:
        raise ValueError("steps must be non-negative")
    return initial @ jnp.linalg.matrix_power(transition, steps)


def terminal_cost(initial, transition, steps: int, cost):
    return terminal_distribution(initial, transition, steps) @ cost
