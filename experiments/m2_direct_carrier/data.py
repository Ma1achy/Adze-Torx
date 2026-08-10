"""Explicit-key synthetic structured carrier data for M2."""

from __future__ import annotations

import jax
import jax.numpy as jnp


def make_data(key: jax.Array, n: int, capacity: int = 4, latent_dim: int = 3) -> jax.Array:
    """Generate smooth low-rank carriers with independent keyed noise."""
    z_key, noise_key = jax.random.split(key)
    z = jax.random.normal(z_key, (n, latent_dim))
    positions = jnp.linspace(-1.0, 1.0, capacity)
    basis = jnp.stack([jnp.ones_like(positions), positions, positions**2], axis=0)[:latent_dim]
    clean = z @ basis
    clean = clean[..., None] * jnp.linspace(0.8, 1.2, latent_dim)[None, None, :]
    clean = clean.reshape(n, capacity * latent_dim)
    return clean + 0.02 * jax.random.normal(noise_key, clean.shape)


def corrupt(clean: jax.Array, key: jax.Array, alpha: float, sigma: float) -> jax.Array:
    """Apply h_corrupt = alpha*h_clean + sigma*epsilon."""
    return alpha * clean + sigma * jax.random.normal(key, clean.shape, dtype=clean.dtype)


def corruption_levels() -> dict[str, tuple[float, float]]:
    return {"low": (0.9, 0.10), "medium": (0.6, 0.50), "high": (0.3, 0.90)}
