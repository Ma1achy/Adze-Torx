"""Exact fixed-width decimal addition and carry-cursor inputs."""

from __future__ import annotations

from typing import NamedTuple, cast

import jax
import jax.numpy as jnp

DIGITS = 12
RESULT_DIGITS = 13
BASE = 10


class ArithmeticBatch(NamedTuple):
    a: jax.Array
    b: jax.Array
    result: jax.Array
    carry: jax.Array
    all_conditioning: jax.Array
    cursor_conditioning: jax.Array
    initial_dynamic: jax.Array
    carry_depth: jax.Array


def _add_one(a: jax.Array, b: jax.Array) -> tuple[jax.Array, jax.Array]:
    def step(carry: jax.Array, values: jax.Array):
        ai, bi = values
        total = ai + bi + carry
        return total // BASE, (total % BASE, total // BASE)

    final_carry, outputs = jax.lax.scan(
        step, jnp.asarray(0, dtype=jnp.int32), jnp.stack((a, b), axis=-1)
    )
    digits, carry_out = outputs
    result = jnp.concatenate((digits, final_carry[None]), axis=0)
    carry_in = jnp.concatenate((jnp.zeros((1,), dtype=jnp.int32), carry_out[:-1]))
    return result, jnp.stack((carry_in, carry_out), axis=-1)


def exact_add(a: jax.Array, b: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Return complete result digits and carry-in/out states."""
    return jax.vmap(_add_one)(a, b)


def _one(key: jax.Array, depth: int) -> tuple[jax.Array, jax.Array]:
    digit_key, start_key = jax.random.split(key)
    raw = jax.random.randint(digit_key, (2, DIGITS), 0, 5)
    start = jax.random.randint(start_key, (), 0, DIGITS - depth + 1)
    positions = jnp.arange(DIGITS)
    chain = (positions >= start) & (positions < start + depth)
    initiator = positions == start
    terminator = positions == start + depth
    a = raw[0]
    b = raw[1]
    a = jnp.where(chain, 9, a)
    b = jnp.where(chain, jnp.where(initiator, 1, 0), b)
    a = jnp.where(terminator, 0, a)
    b = jnp.where(terminator, 0, b)
    return a.astype(jnp.int32), b.astype(jnp.int32)


def make_data(key: jax.Array, n: int, depth: int) -> ArithmeticBatch:
    if depth not in (1, 2, 4, 8, 12):
        raise ValueError("depth must be one of 1, 2, 4, 8, or 12")
    a, b = jax.vmap(lambda k: _one(k, depth))(jax.random.split(key, n))
    result, carry = exact_add(a, b)
    pos = jnp.arange(DIGITS, dtype=jnp.float32) / (DIGITS - 1)
    position = jnp.stack((pos, 1.0 - pos), axis=-1).reshape(-1)
    position = jnp.broadcast_to(position, (n, position.size))
    all_conditioning = jnp.concatenate(
        (a.astype(jnp.float32) / 9.0, b.astype(jnp.float32) / 9.0, position), axis=-1
    )
    cursor = jnp.zeros((n, DIGITS, 5), dtype=jnp.float32)
    cursor = cursor.at[:, :, 0].set(a / 9.0)
    cursor = cursor.at[:, :, 1].set(b / 9.0)
    cursor = cursor.at[:, :, 2].set(jnp.arange(DIGITS, dtype=jnp.float32) / (DIGITS - 1))
    cursor = cursor.at[:, :, 3].set(1.0)
    done = jnp.zeros((n, 1, 5), dtype=jnp.float32).at[:, :, 4].set(1.0)
    cursor_conditioning = jnp.concatenate((cursor, done), axis=1)
    initial_dynamic = jnp.zeros((n, 32), dtype=jnp.float32)

    def run_length(carry_out: jax.Array) -> jax.Array:
        def step(run, value):
            next_run = jnp.where(value == 1, run + 1, 0)
            return next_run, next_run

        _, runs = jax.lax.scan(step, jnp.asarray(0, dtype=jnp.int32), carry_out)
        return jnp.max(cast(jax.Array, runs))

    computed_depth = jnp.asarray(jax.vmap(run_length)(carry[:, :, 1]))
    return ArithmeticBatch(
        a, b, result, carry, all_conditioning, cursor_conditioning, initial_dynamic, computed_depth
    )


def exact_accuracy(logits: jax.Array, result: jax.Array) -> jax.Array:
    prediction = jnp.argmax(logits, axis=-1)
    return jnp.mean(jnp.all(prediction == result, axis=-1))


def digit_accuracy(logits: jax.Array, result: jax.Array) -> jax.Array:
    return jnp.mean(jnp.argmax(logits, axis=-1) == result)
