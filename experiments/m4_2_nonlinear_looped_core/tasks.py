"""Per-example operator-sequence composition task."""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.nn as jnn
import jax.numpy as jnp


class CompositionBatch(NamedTuple):
    initial: jax.Array
    target: jax.Array
    operators: jax.Array
    intermediates: jax.Array


def operator_family() -> tuple[jax.Array, jax.Array]:
    matrices = jnp.asarray(
        [
            [[0.65, 0.10, 0.00], [0.00, 0.55, 0.12], [0.08, 0.00, 0.60]],
            [[0.45, -0.20, 0.08], [0.15, 0.60, 0.00], [0.00, 0.18, 0.50]],
            [[0.55, 0.00, -0.15], [0.12, 0.48, 0.10], [-0.10, 0.08, 0.58]],
        ]
    )
    biases = jnp.asarray([[0.10, -0.05, 0.03], [-0.08, 0.06, 0.02], [0.04, 0.02, -0.07]])
    return matrices, biases


def _sequence_codes(k: int, train: bool) -> jax.Array:
    total = 3**k
    if k == 1:
        return jnp.arange(total, dtype=jnp.int32)
    values = tuple(code for code in range(total) if (code % 5 != 0) == train)
    return jnp.asarray(values, dtype=jnp.int32)


def _decode(codes: jax.Array, k: int) -> jax.Array:
    powers = 3 ** jnp.arange(k - 1, -1, -1)
    return (codes[:, None] // powers[None, :]) % 3


def _one(initial: jax.Array, operators: jax.Array, matrices: jax.Array, biases: jax.Array):
    def step(state, op):
        next_state = jnp.tanh(matrices[op] @ state + biases[op])
        return next_state, next_state

    _, rest = jax.lax.scan(step, initial, operators)
    return jnp.concatenate((initial[None, :], rest), axis=0)


def make_data(key: jax.Array, n: int, k: int, train: bool, width: int = 18) -> CompositionBatch:
    state_key, sequence_key = jax.random.split(key)
    matrices, biases = operator_family()
    choices = jax.random.randint(sequence_key, (n,), 0, _sequence_codes(k, train).shape[0])
    codes = _sequence_codes(k, train)[choices]
    operators = _decode(codes, k).astype(jnp.int32)
    states = jax.random.uniform(state_key, (n, 3), minval=-0.8, maxval=0.8)
    intermediates = jax.vmap(lambda state, ops: _one(state, ops, matrices, biases))(
        states, operators
    )
    operator_features = jnn.one_hot(operators, 3).reshape((n, 3 * k))
    if 3 + 3 * k > width:
        raise ValueError("width is too small to encode the operator sequence")
    initial = jnp.zeros((n, width)).at[:, :3].set(states)
    initial = initial.at[:, 3 : 3 + 3 * k].set(operator_features)
    target = jnp.zeros((n, width)).at[:, :3].set(intermediates[:, -1])
    return CompositionBatch(initial, target, operators, intermediates)


def task_loss(params, batch: CompositionBatch, keys, apply_fn):
    predictions = jax.vmap(lambda state, key: apply_fn(params, state, key))(batch.initial, keys)
    return jnp.mean((predictions[:, :3] - batch.target[:, :3]) ** 2)


def task_metrics(params, batch: CompositionBatch, keys, apply_fn) -> dict[str, jax.Array]:
    predictions = jax.vmap(lambda state, key: apply_fn(params, state, key))(batch.initial, keys)
    per_example = jnp.mean((predictions[:, :3] - batch.target[:, :3]) ** 2, axis=-1)
    return {
        "loss": jnp.mean(per_example),
        "success_rate": jnp.mean(per_example < 0.01),
    }
