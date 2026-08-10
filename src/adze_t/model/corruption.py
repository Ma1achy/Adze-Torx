"""Fixed-shape content and absorbing structural corruption."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from .carrier import BOUNDARY_UNKNOWN, StructureConfig


def corrupt_content(clean: jax.Array, key: jax.Array, alpha: float, sigma: float) -> jax.Array:
    return alpha * clean + sigma * jax.random.normal(key, clean.shape, dtype=clean.dtype)


def corrupt_boundary(clean: jax.Array, key: jax.Array, rho: float) -> jax.Array:
    mask = jax.random.uniform(key, clean.shape) < rho
    return jnp.where(mask, BOUNDARY_UNKNOWN, clean).astype(jnp.int32)


def corrupt_length(
    clean: jax.Array, key: jax.Array, rho: float, config: StructureConfig
) -> jax.Array:
    mask = jax.random.uniform(key, clean.shape) < rho
    return jnp.where(mask, config.length_unknown, clean).astype(jnp.int32)


def corrupt_structure(
    clean_b: jax.Array,
    clean_length: jax.Array,
    key: jax.Array,
    rho_b: float,
    rho_length: float,
    config: StructureConfig,
) -> tuple[jax.Array, jax.Array]:
    """Corrupt b and length independently with absorbing UNKNOWN masks."""
    b_key, length_key = jax.random.split(key)
    return (
        corrupt_boundary(clean_b, b_key, rho_b),
        corrupt_length(clean_length, length_key, rho_length, config),
    )
