"""Deterministic DiT conditioning embeddings."""

from jax import Array
import jax.numpy as jnp


def sinusoidal_embedding(value: Array, width: int) -> Array:
    value = jnp.asarray(value, dtype=jnp.float32).reshape((-1, 1))
    half = max(width // 2, 1)
    frequencies = jnp.exp(-jnp.log(10000.0) * jnp.arange(half) / max(half - 1, 1))
    emb = jnp.concatenate([jnp.sin(value * frequencies), jnp.cos(value * frequencies)], axis=-1)
    return emb[:, :width]


def build_conditioning(
    prompt_global: Array,
    noise: Array | float,
    mode: Array | int,
    denoise_step: Array | int,
    refinement_step: Array | int,
    effective_depth: Array | int,
) -> Array:
    """Build the shared conditioning vector while retaining all Phase B axes."""
    batch = prompt_global.shape[0]
    scalars = [noise, mode, denoise_step, refinement_step, effective_depth]
    embeddings = [
        sinusoidal_embedding(jnp.broadcast_to(jnp.asarray(x), (batch,)), 16) for x in scalars
    ]
    return jnp.concatenate([prompt_global, *embeddings], axis=-1)
