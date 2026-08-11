"""PROVISIONAL_PHASE_B_TEACHER monotonic carrier structure."""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp
from jax import Array

from .config import ReferenceConfig


class TeacherStructure(NamedTuple):
    boundaries: Array
    length: Array
    activity: Array
    slot_mask: Array
    slot_bytes: Array
    capacity_overflow: Array
    prefix_mask_valid: Array


def canonical_teacher_structure_core(
    target_bytes: Array, target_mask: Array, config: ReferenceConfig
) -> TeacherStructure:
    """Pure-JAX teacher mapping with explicit validity flags."""
    batch, width = target_bytes.shape
    capacity, max_length = config.carrier.C, config.carrier.L_max
    total_slots = capacity * max_length
    target_mask = target_mask.astype(bool)
    raw_n_bytes = jnp.sum(target_mask, axis=1)
    capacity_overflow = raw_n_bytes > total_slots
    expected_prefix = jnp.arange(width)[None, :] < raw_n_bytes[:, None]
    prefix_mask_valid = jnp.all(target_mask == expected_prefix, axis=1)
    n_bytes = jnp.minimum(raw_n_bytes, total_slots)
    carrier = jnp.arange(capacity)[None, :]
    length = jnp.clip(n_bytes[:, None] - carrier * max_length, 0, max_length).astype(jnp.int32)
    activity = length > 0

    positions = jnp.arange(capacity)
    base_boundaries = ((positions + 1) % config.packing.K == 0).astype(jnp.int32)
    base_boundaries = base_boundaries.at[-1].set(1)
    boundaries = jnp.broadcast_to(base_boundaries, (batch, capacity))

    padded = jnp.pad(target_bytes, ((0, 0), (0, max(0, total_slots - width))))
    slot_bytes = padded[:, :total_slots].reshape(batch, capacity, max_length)
    slot_mask = (jnp.arange(total_slots)[None, :] < n_bytes[:, None]).reshape(
        batch, capacity, max_length
    )
    return TeacherStructure(
        boundaries,
        length,
        activity,
        slot_mask,
        slot_bytes,
        capacity_overflow,
        prefix_mask_valid,
    )


def canonical_teacher_structure(
    target_bytes: Array, target_mask: Array, config: ReferenceConfig
) -> TeacherStructure:
    """Eager validated wrapper around :func:`canonical_teacher_structure_core`."""
    if target_bytes.ndim != 2 or target_mask.shape != target_bytes.shape:
        raise ValueError("target_bytes and target_mask must have matching [batch, sequence] shape")
    teacher = canonical_teacher_structure_core(target_bytes, target_mask, config)
    if bool(jnp.any(teacher.capacity_overflow)):
        raise ValueError("target byte count exceeds carrier emission capacity")
    if not bool(jnp.all(teacher.prefix_mask_valid)):
        raise ValueError("target_mask must be prefix-valid")
    return teacher
