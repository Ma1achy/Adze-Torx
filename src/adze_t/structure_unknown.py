"""Versioned structural-UNKNOWN encoding for Phase H candidates.

Accepted checkpoints are never edited in place.  The migration helpers return
new deterministic mean-parameter trees whose existing embedding rows are
bitwise preserved and whose new UNKNOWN rows are initialized to zero.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from jax import Array
import jax.numpy as jnp


PHASE_H_STRUCTURE_UNKNOWN_V1 = "PHASE_H_STRUCTURE_UNKNOWN_V1"
BOUNDARY_UNKNOWN = 2
BOUNDARY_LEFT_BOS = 2
BOUNDARY_LEFT_UNKNOWN = 3


@dataclass(frozen=True)
class StructureUnknownCodes:
    """Valid categorical codes used only by observed/corrupted structure."""

    boundary: int
    extent: int


def structure_unknown_codes(max_extent: int) -> StructureUnknownCodes:
    if max_extent < 1:
        raise ValueError("max_extent must be positive")
    return StructureUnknownCodes(boundary=BOUNDARY_UNKNOWN, extent=max_extent + 1)


def _append_zero_row(table: Array, expected_rows: int, name: str) -> Array:
    if table.ndim != 2 or table.shape[0] != expected_rows:
        raise ValueError(f"{name} must have shape [{expected_rows}, width]")
    return jnp.concatenate((table, jnp.zeros_like(table[:1])), axis=0)


def migrate_dit_structure_unknown_embeddings(
    dit_params: dict[str, Any], *, max_extent: int
) -> dict[str, Any]:
    """Return candidate DiT means with distinct UNKNOWN rows appended.

    The expected input is an unmigrated deterministic parameter tree.  Calling
    the migration twice is rejected so checkpoint provenance cannot silently
    mix schemas.
    """
    codes = structure_unknown_codes(max_extent)
    required = ("length_embed", "boundary_left_embed", "boundary_right_embed")
    if any(name not in dit_params for name in required):
        raise ValueError("DiT parameter tree is missing structural embeddings")
    length = dit_params["length_embed"]
    left = dit_params["boundary_left_embed"]
    right = dit_params["boundary_right_embed"]
    if not all(isinstance(value, Array) for value in (length, left, right)):
        raise TypeError("migrate deterministic mean parameters before Torx conversion")
    migrated = dict(dit_params)
    migrated["length_embed"] = _append_zero_row(length, codes.extent, "length_embed")
    migrated["boundary_left_embed"] = _append_zero_row(
        left, BOUNDARY_LEFT_UNKNOWN, "boundary_left_embed"
    )
    migrated["boundary_right_embed"] = _append_zero_row(
        right, BOUNDARY_UNKNOWN, "boundary_right_embed"
    )
    return migrated


def migrate_model_structure_unknown_embeddings(
    params: dict[str, Any], *, max_extent: int
) -> dict[str, Any]:
    """Migrate a copy of a deterministic model tree without mutating input."""
    if "dit" not in params or not isinstance(params["dit"], dict):
        raise ValueError("model parameter tree has no deterministic DiT subtree")
    return {
        **params,
        "dit": migrate_dit_structure_unknown_embeddings(params["dit"], max_extent=max_extent),
    }


def resolve_structural_embedding_indices(
    observed_b: Array,
    observed_l: Array,
    safe_carrier: Array,
    *,
    batch: int,
) -> tuple[Array, Array, Array]:
    """Map observed codes to right/left/extent embedding indices.

    Boundary UNKNOWN is code 2.  The historical left-edge BOS remains row 2,
    while non-BOS UNKNOWN uses the appended left-embedding row 3.
    """
    rows = jnp.arange(batch)[:, None]
    right = observed_b[rows, safe_carrier].astype(jnp.int32)
    previous_carrier = jnp.maximum(safe_carrier - 1, 0)
    previous = observed_b[rows, previous_carrier].astype(jnp.int32)
    left = jnp.where(previous == BOUNDARY_UNKNOWN, BOUNDARY_LEFT_UNKNOWN, previous)
    left = jnp.where(safe_carrier == 0, BOUNDARY_LEFT_BOS, left)
    extent = observed_l[rows, safe_carrier].astype(jnp.int32)
    return right, left, extent
