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


def canonical_teacher_structure(
    target_bytes: Array, target_mask: Array, config: ReferenceConfig
) -> TeacherStructure:
    """Map every prefix-valid byte exactly once to ``(t//L_max,t%L_max)``."""
    batch, width = target_bytes.shape
    capacity, max_length = config.carrier.C, config.carrier.L_max
    total_slots = capacity * max_length
    n_bytes = jnp.minimum(jnp.sum(target_mask, axis=1), total_slots)
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
    return TeacherStructure(boundaries, length, activity, slot_mask, slot_bytes)
