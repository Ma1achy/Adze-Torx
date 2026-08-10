"""Synthetic fixed-carrier segments with recoverable local boundaries."""

from __future__ import annotations

from typing import NamedTuple, cast

import jax
import jax.numpy as jnp

from adze_t.model.carrier import StructureConfig
from adze_t.model.corruption import corrupt_content, corrupt_structure


class StructuredCarrier(NamedTuple):
    h: jax.Array
    b: jax.Array
    length: jax.Array


def _one_example(key: jax.Array, config: StructureConfig) -> StructuredCarrier:
    boundary_key, latent_key, type_key, noise_key = jax.random.split(key, 4)
    boundary_draws = jax.random.uniform(boundary_key, (config.capacity,))
    candidates = jax.random.normal(latent_key, (config.capacity, config.latent_dim))
    candidate_types = jax.random.randint(type_key, (config.capacity,), 0, config.max_length + 1)
    noise = 0.025 * jax.random.normal(noise_key, (config.capacity, config.latent_dim))
    positions = jnp.linspace(-1.0, 1.0, config.capacity)

    def step(carry, values):
        previous_latent, previous_type, index = carry
        boundary_draw, candidate, candidate_type = values
        boundary_draw = cast(jax.Array, boundary_draw)
        candidate = cast(jax.Array, candidate)
        candidate_type = cast(jax.Array, candidate_type)
        is_first = index == 0
        boundary = jnp.where(is_first, 1, boundary_draw < 0.35).astype(jnp.int32)
        latent = jnp.where(boundary == 1, candidate, previous_latent + 0.025 * candidate)
        chunk_type = jnp.where(boundary == 1, candidate_type, previous_type)
        return (latent, chunk_type, index + 1), (boundary, latent, chunk_type)

    initial = (candidates[0], candidate_types[0], jnp.asarray(0))
    _, (boundaries, latents, chunk_types) = jax.lax.scan(
        step,
        initial,
        (boundary_draws, candidates, candidate_types),
    )
    type_array = cast(jax.Array, chunk_types)
    latent_array = cast(jax.Array, latents)
    boundary_array = cast(jax.Array, boundaries)
    type_effect = type_array[:, None] * jnp.array([0.8, -0.6, 0.4])
    boundary_effect = boundary_array[:, None] * jnp.array([1.8, -1.45, 1.0])
    position_effect = positions[:, None] * jnp.array([0.08, -0.04, 0.05])
    h = latent_array + type_effect + boundary_effect + position_effect + noise
    return StructuredCarrier(h, boundary_array, type_array.astype(jnp.int32))


def make_data(key: jax.Array, n: int, config: StructureConfig | None = None) -> StructuredCarrier:
    config = StructureConfig() if config is None else config
    return jax.vmap(lambda sample_key: _one_example(sample_key, config))(jax.random.split(key, n))


def corrupt_batch(
    data: StructuredCarrier,
    key: jax.Array,
    alpha: float,
    sigma: float,
    rho_b: float,
    rho_length: float,
    config: StructureConfig,
) -> StructuredCarrier:
    h_key, structure_key = jax.random.split(key)
    observed_b, observed_length = corrupt_structure(
        data.b, data.length, structure_key, rho_b, rho_length, config
    )
    return StructuredCarrier(
        corrupt_content(data.h, h_key, alpha, sigma), observed_b, observed_length
    )


def shuffle_structure(data: StructuredCarrier, key: jax.Array) -> StructuredCarrier:
    """Shuffle targets across examples for the no-leakage control."""
    b_key, length_key = jax.random.split(key)
    return StructuredCarrier(
        data.h,
        data.b[jax.random.permutation(b_key, data.b.shape[0])],
        data.length[jax.random.permutation(length_key, data.length.shape[0])],
    )
