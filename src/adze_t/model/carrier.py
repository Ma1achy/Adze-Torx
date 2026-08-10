"""Persistent fixed-capacity carrier construction and invariants."""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp

BOUNDARY_UNKNOWN = 2


@dataclass(frozen=True)
class StructureConfig:
    """Fixed categorical supports for the M3 structure channels."""

    capacity: int = 6
    latent_dim: int = 3
    max_length: int = 3

    def __post_init__(self) -> None:
        if min(self.capacity, self.latent_dim, self.max_length) < 1:
            raise ValueError("capacity, latent_dim, and max_length must be positive")

    @property
    def length_unknown(self) -> int:
        return self.max_length + 1

    @property
    def length_observed_classes(self) -> int:
        return self.max_length + 2


def validate_carrier_shapes(
    h: jax.Array,
    b: jax.Array,
    length: jax.Array,
    config: StructureConfig,
) -> None:
    """Validate fixed carrier shapes and structural supports."""
    if h.shape[-2:] != (config.capacity, config.latent_dim):
        raise ValueError(f"h must end in {(config.capacity, config.latent_dim)}, got {h.shape}")
    if b.shape[-1:] != (config.capacity,) or length.shape[-1:] != (config.capacity,):
        raise ValueError("b and length must end in the fixed carrier capacity")
    if bool(jnp.any((b < 0) | (b > BOUNDARY_UNKNOWN))):
        raise ValueError("boundary values must be 0, 1, or UNKNOWN")
    if bool(jnp.any((length < 0) | (length > config.length_unknown))):
        raise ValueError("length values must be 0..max_length or UNKNOWN")


def initialise_carrier(*args, **kwargs):
    """Create a zero-content, unknown-structure fixed carrier."""
    config = kwargs.get("config", args[0] if args else StructureConfig())
    h = jnp.zeros((config.capacity, config.latent_dim))
    b = jnp.full((config.capacity,), BOUNDARY_UNKNOWN, dtype=jnp.int32)
    length = jnp.full((config.capacity,), config.length_unknown, dtype=jnp.int32)
    return h, b, length
