"""Hard per-example nonlinear operator-composition tasks."""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

STATE_DIM = 8
OPERATOR_COUNT = 4
WIDTH = 64


class HardBatch(NamedTuple):
    initial: jax.Array
    target: jax.Array
    operators: jax.Array
    intermediates: jax.Array


def operator_family() -> tuple[jax.Array, jax.Array]:
    matrices = jnp.asarray(
        [
            [
                [0.70, 0.08, 0.00, 0.00, 0.04, 0.00, 0.00, 0.00],
                [0.00, 0.62, 0.10, 0.00, 0.00, 0.03, 0.00, 0.00],
                [0.04, 0.00, 0.66, 0.08, 0.00, 0.00, 0.02, 0.00],
                [0.00, 0.03, 0.00, 0.58, 0.09, 0.00, 0.00, 0.02],
                [0.02, 0.00, 0.05, 0.00, 0.64, 0.07, 0.00, 0.00],
                [0.00, 0.04, 0.00, 0.02, 0.00, 0.61, 0.08, 0.00],
                [0.00, 0.00, 0.03, 0.00, 0.04, 0.00, 0.67, 0.06],
                [0.03, 0.00, 0.00, 0.05, 0.00, 0.02, 0.00, 0.59],
            ],
            [
                [0.58, -0.10, 0.06, 0.00, 0.00, 0.05, 0.00, 0.02],
                [0.08, 0.65, 0.00, 0.04, -0.03, 0.00, 0.02, 0.00],
                [0.00, 0.05, 0.55, 0.12, 0.00, 0.00, 0.04, 0.00],
                [0.02, 0.00, 0.08, 0.63, 0.06, -0.02, 0.00, 0.00],
                [0.00, 0.03, 0.00, 0.07, 0.57, 0.10, 0.00, 0.04],
                [0.04, 0.00, 0.02, 0.00, 0.05, 0.60, 0.09, 0.00],
                [0.00, 0.02, 0.05, 0.00, 0.00, 0.08, 0.59, 0.07],
                [0.06, 0.00, 0.00, 0.03, 0.02, 0.00, 0.05, 0.62],
            ],
            [
                [0.62, 0.00, -0.08, 0.05, 0.00, 0.00, 0.06, 0.00],
                [0.02, 0.57, 0.09, 0.00, 0.05, 0.00, 0.00, 0.03],
                [0.06, 0.04, 0.60, 0.00, 0.00, 0.07, 0.00, 0.00],
                [0.00, 0.02, 0.00, 0.68, 0.04, 0.00, 0.08, 0.00],
                [0.03, 0.00, 0.04, 0.03, 0.61, 0.00, 0.00, 0.09],
                [0.00, 0.06, 0.00, 0.05, 0.02, 0.56, 0.10, 0.00],
                [0.01, 0.00, 0.05, 0.00, 0.06, 0.04, 0.63, 0.02],
                [0.00, 0.03, 0.00, 0.06, 0.00, 0.08, 0.02, 0.58],
            ],
            [
                [0.66, 0.07, 0.00, 0.03, 0.00, -0.06, 0.00, 0.05],
                [0.00, 0.59, -0.04, 0.08, 0.00, 0.00, 0.07, 0.00],
                [0.03, 0.00, 0.64, 0.05, 0.08, 0.00, 0.00, 0.02],
                [0.00, 0.04, 0.02, 0.60, 0.00, 0.09, 0.03, 0.00],
                [0.05, 0.00, 0.00, 0.06, 0.58, 0.03, 0.00, 0.07],
                [0.00, 0.02, 0.06, 0.00, 0.05, 0.65, 0.04, 0.00],
                [0.00, 0.05, 0.00, 0.02, 0.00, 0.07, 0.60, 0.08],
                [0.04, 0.00, 0.03, 0.00, 0.07, 0.00, 0.05, 0.62],
            ],
        ]
    )
    biases = jnp.asarray(
        [
            [0.14, -0.08, 0.05, 0.03, -0.06, 0.04, 0.02, -0.03],
            [-0.10, 0.06, 0.03, -0.05, 0.08, -0.02, 0.04, 0.01],
            [0.05, 0.03, -0.09, 0.07, 0.02, 0.06, -0.04, 0.08],
            [-0.04, 0.09, 0.02, 0.05, -0.03, -0.07, 0.06, 0.04],
        ]
    )
    return matrices, biases


def _code(raw: jax.Array, k: int, residue: int) -> jax.Array:
    total = OPERATOR_COUNT**k
    return (OPERATOR_COUNT * raw + residue) % total


def _decode(codes: jax.Array, k: int) -> jax.Array:
    powers = OPERATOR_COUNT ** jnp.arange(k - 1, -1, -1)
    return (codes[:, None] // powers[None, :]) % OPERATOR_COUNT


def _one(state, operators, matrices, biases):
    def step(carry, operator):
        next_state = jnp.tanh(1.25 * (matrices[operator] @ carry) + biases[operator])
        return next_state, next_state

    _, rest = jax.lax.scan(step, state, operators)
    return jnp.concatenate((state[None, :], rest), axis=0)


def make_data(key: jax.Array, n: int, k: int, train: bool, width: int = WIDTH) -> HardBatch:
    if k < 1 or 3 + STATE_DIM + OPERATOR_COUNT * k > width:
        raise ValueError("k or width cannot represent the fixed operator-conditioned carrier")
    state_key, sequence_key = jax.random.split(key)
    matrices, biases = operator_family()
    total = OPERATOR_COUNT**k
    raw = jax.random.randint(sequence_key, (n,), 0, max(total // 4, 1))
    residue = 1 if train else 0
    operators = _decode(_code(raw, k, residue), k).astype(jnp.int32)
    states = jax.random.uniform(state_key, (n, STATE_DIM), minval=-0.95, maxval=0.95)
    intermediates = jax.vmap(lambda state, ops: _one(state, ops, matrices, biases))(
        states, operators
    )
    features = jax.nn.one_hot(operators, OPERATOR_COUNT).reshape((n, OPERATOR_COUNT * k))
    initial = jnp.zeros((n, width)).at[:, :STATE_DIM].set(states)
    initial = initial.at[:, STATE_DIM : STATE_DIM + OPERATOR_COUNT * k].set(features)
    target = jnp.zeros((n, width)).at[:, :STATE_DIM].set(intermediates[:, -1])
    return HardBatch(initial, target, operators, intermediates)
