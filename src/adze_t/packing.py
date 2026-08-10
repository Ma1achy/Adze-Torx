"""Boundary-to-block construction and hard-pack metadata.

The exact slot-level mapping is a provisional reference implementation choice
documented in architecture v3 section 55. It must not be misrepresented as an
original-Adze fact.
"""

from dataclasses import dataclass
from jax import Array
import jax.numpy as jnp


@dataclass(frozen=True)
class BlockLayout:
    block_id_per_carrier: Array
    within_block_pos: Array
    block_lengths: Array
    n_blocks: Array


@dataclass(frozen=True)
class PackMetadata:
    block_valid: Array
    slot_valid: Array
    query_mask: Array
    kv_mask: Array
    pool_mask: Array
    emit_mask: Array
    carrier_to_m: Array
    carrier_to_k: Array
    packed_to_carrier: Array
    block_id: Array
    carrier_id: Array
    within_block_pos: Array


def boundaries_to_blocks(c_b: Array) -> BlockLayout:
    """Convert committed `cut after i` boundaries to logical contiguous blocks."""
    c_b = jnp.asarray(c_b)
    if c_b.ndim != 2:
        raise ValueError("c_b must have shape [batch, carrier]")
    if c_b.shape[1] == 0:
        raise ValueError("carrier capacity must be positive")
    if not bool(jnp.all((c_b == 0) | (c_b == 1))):
        raise ValueError("c_b must contain only binary values")
    if not bool(jnp.all(c_b[:, -1] == 1)):
        raise ValueError("c_b must have a terminal cut after the final carrier")

    block_id = jnp.cumsum(c_b, axis=1) - c_b
    carrier_pos = jnp.broadcast_to(jnp.arange(c_b.shape[1]), c_b.shape)
    # A cut after i starts the next block at i+1.  The first carrier always
    # starts block zero, regardless of the first edge value.
    previous_cut = jnp.concatenate(
        [jnp.zeros((c_b.shape[0], 1), dtype=c_b.dtype), c_b[:, :-1]], axis=1
    )
    starts = jnp.where(previous_cut == 1, carrier_pos, 0)
    start_for_site = jnp.maximum.accumulate(starts, axis=1)
    within = carrier_pos - start_for_site
    block_lengths = jnp.sum(jax_one_hot(block_id, int(c_b.shape[1])), axis=1, dtype=jnp.int32)
    n_blocks = jnp.sum(c_b, axis=1, dtype=jnp.int32)
    return BlockLayout(block_id, within, block_lengths, n_blocks)


def jax_one_hot(indices: Array, size: int) -> Array:
    """Small local one-hot helper avoiding a dependency on a model utility."""
    return (jnp.arange(size)[None, None, :] == indices[..., None]).astype(jnp.int32)


def build_pack_metadata(
    c_b: Array,
    activity: Array,
    *,
    M_max: int,
    K: int,
) -> PackMetadata:
    """Build the hard-pack maps/masks without truncation."""
    c_b = jnp.asarray(c_b)
    activity = jnp.asarray(activity)
    if c_b.ndim != 2 or activity.shape != c_b.shape:
        raise ValueError("c_b and activity must both have shape [batch, carrier]")
    if M_max <= 0 or K <= 0:
        raise ValueError("M_max and K must be positive")
    if not bool(jnp.all((activity == 0) | (activity == 1))):
        raise ValueError("activity must contain only binary values")

    layout = boundaries_to_blocks(c_b)
    if int(layout.n_blocks.max()) > M_max:
        raise ValueError(
            f"generated block count exceeds M_max={M_max}: {int(layout.n_blocks.max())}"
        )
    if int(layout.block_lengths.max()) > K:
        raise ValueError(f"generated block length exceeds K={K}: {int(layout.block_lengths.max())}")

    batch, capacity = c_b.shape
    packed_to_carrier = jnp.full((batch, M_max, K), -1, dtype=jnp.int32)
    packed_to_carrier = packed_to_carrier.at[
        jnp.arange(batch)[:, None], layout.block_id_per_carrier, layout.within_block_pos
    ].set(jnp.broadcast_to(jnp.arange(capacity), (batch, capacity)))
    carrier_to_m = layout.block_id_per_carrier.astype(jnp.int32)
    carrier_to_k = layout.within_block_pos.astype(jnp.int32)

    block_valid = jnp.arange(M_max)[None, :] < layout.n_blocks[:, None]
    slot_valid = packed_to_carrier >= 0
    safe_carrier = jnp.maximum(packed_to_carrier, 0)
    activity_packed = jnp.take_along_axis(activity[:, None, :], safe_carrier, axis=2)
    query_mask = slot_valid
    kv_mask = slot_valid & (activity_packed.astype(bool))
    pool_mask = kv_mask
    emit_mask = kv_mask
    block_id = jnp.broadcast_to(jnp.arange(M_max)[None, :, None], (batch, M_max, K))
    carrier_id = packed_to_carrier
    within_block_pos = jnp.broadcast_to(jnp.arange(K)[None, None, :], (batch, M_max, K))
    return PackMetadata(
        block_valid,
        slot_valid,
        query_mask,
        kv_mask,
        pool_mask,
        emit_mask,
        carrier_to_m,
        carrier_to_k,
        packed_to_carrier,
        block_id,
        carrier_id,
        within_block_pos,
    )


def pack_values(values: Array, metadata: PackMetadata) -> Array:
    """Gather `[B,C,...]` carrier values into `[B,M_max,K,...]`."""
    values = jnp.asarray(values)
    if values.ndim < 2:
        raise ValueError("values must have shape [batch, carrier, ...]")
    if values.shape[0] != metadata.packed_to_carrier.shape[0]:
        raise ValueError("values and metadata batch dimensions must match")
    safe = jnp.maximum(metadata.packed_to_carrier, 0)
    packed = values[jnp.arange(values.shape[0])[:, None, None], safe]
    return jnp.where(metadata.slot_valid[(...,) + (None,) * (values.ndim - 2)], packed, 0)
