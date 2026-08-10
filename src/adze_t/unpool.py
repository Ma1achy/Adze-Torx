"""Scatter packed values back to persistent carrier identities."""

from jax import Array
import jax.numpy as jnp

from .packing import PackMetadata


def unpool_values(packed: Array, metadata: PackMetadata, *, C: int) -> Array:
    """Scatter `[B,M_max,K,...]` values back to `[B,C,...]` exactly once per site."""
    packed = jnp.asarray(packed)
    if packed.ndim < 3:
        raise ValueError("packed must have shape [batch, M_max, K, ...]")
    if packed.shape[:3] != metadata.packed_to_carrier.shape:
        raise ValueError("packed shape does not match metadata")
    if C != metadata.carrier_to_m.shape[1]:
        raise ValueError("C does not match metadata carrier capacity")
    batch = packed.shape[0]
    m = metadata.carrier_to_m
    k = metadata.carrier_to_k
    batch_ids = jnp.arange(batch)[:, None]
    restored = packed[batch_ids, m, k]
    return restored
