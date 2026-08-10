"""Bijective modular register-machine programs."""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

MODULUS = 16
OP_NAMES = (
    "INC_A",
    "DEC_A",
    "INC_B",
    "DEC_B",
    "ADD_A_B",
    "ADD_B_A",
    "SUB_A_B",
    "SUB_B_A",
    "SWAP",
    "NEG_A",
    "NEG_B",
)
OPS = len(OP_NAMES)
MAX_LENGTH = 12


class ProgramBatch(NamedTuple):
    initial_registers: jax.Array
    programs: jax.Array
    valid: jax.Array
    final_registers: jax.Array
    intermediates: jax.Array
    all_conditioning: jax.Array
    cursor_conditioning: jax.Array
    initial_dynamic: jax.Array


def apply_instruction(registers: jax.Array, instruction: jax.Array) -> jax.Array:
    a, b = registers
    # Every branch is a bijection on Z_16^2.
    candidates = jnp.stack(
        (
            jnp.asarray(((a + 1) % MODULUS, b)),
            jnp.asarray(((a - 1) % MODULUS, b)),
            jnp.asarray((a, (b + 1) % MODULUS)),
            jnp.asarray((a, (b - 1) % MODULUS)),
            jnp.asarray(((a + b) % MODULUS, b)),
            jnp.asarray((a, (a + b) % MODULUS)),
            jnp.asarray(((a - b) % MODULUS, b)),
            jnp.asarray((a, (b - a) % MODULUS)),
            jnp.asarray((b, a)),
            jnp.asarray(((-a) % MODULUS, b)),
            jnp.asarray((a, (-b) % MODULUS)),
        )
    )
    return candidates[instruction]


def execute_program(initial: jax.Array, program: jax.Array) -> tuple[jax.Array, jax.Array]:
    def step(registers, instruction):
        next_registers = apply_instruction(registers, instruction)
        return next_registers, next_registers

    final, rest = jax.lax.scan(step, initial, program)
    return jnp.concatenate((initial[None], rest), axis=0), final


def _sample_program(key: jax.Array, length: int) -> jax.Array:
    return jax.random.randint(key, (length,), 0, OPS)


def make_data(key: jax.Array, n: int, length: int) -> ProgramBatch:
    if length not in (4, 8, 12):
        raise ValueError("length must be 4, 8, or 12")
    init_key, program_key = jax.random.split(key)
    initial = jax.random.randint(init_key, (n, 2), 0, MODULUS)
    programs = jax.vmap(lambda k: _sample_program(k, length))(jax.random.split(program_key, n))
    intermediates, final = jax.vmap(execute_program)(initial, programs)
    padded = jnp.zeros((n, MAX_LENGTH), dtype=jnp.int32).at[:, :length].set(programs)
    valid = jnp.zeros((n, MAX_LENGTH), dtype=jnp.float32).at[:, :length].set(1.0)
    features = jax.nn.one_hot(padded, OPS).reshape((n, MAX_LENGTH * OPS))
    all_conditioning = jnp.concatenate((features, valid), axis=-1)
    cursor = jax.nn.one_hot(padded, OPS)
    done = (1.0 - valid)[..., None]
    cursor_conditioning = jnp.concatenate((cursor, done), axis=-1)
    initial_dynamic = jnp.zeros((n, 32), dtype=jnp.float32).at[:, :2].set(initial / (MODULUS - 1))
    return ProgramBatch(
        initial,
        padded,
        valid,
        final,
        intermediates,
        all_conditioning,
        cursor_conditioning,
        initial_dynamic,
    )


def mask_all_conditioning(conditioning: jax.Array) -> jax.Array:
    """Zero padded instruction one-hots while retaining the explicit mask."""
    leading = conditioning.shape[:-1]
    features = conditioning[..., : MAX_LENGTH * OPS].reshape((*leading, MAX_LENGTH, OPS))
    valid = conditioning[..., MAX_LENGTH * OPS :]
    masked = jnp.concatenate((features * valid[..., None], valid[..., None]), axis=-1)
    return masked.reshape(conditioning.shape)


def exact_accuracy(logits: jax.Array, target: jax.Array) -> jax.Array:
    prediction = jnp.argmax(logits, axis=-1)
    return jnp.mean(jnp.all(prediction == target, axis=-1))


def register_accuracy(logits: jax.Array, target: jax.Array) -> jax.Array:
    return jnp.mean(jnp.argmax(logits, axis=-1) == target)
