"""Boundary-to-block construction and hard-pack metadata.

The exact slot-level mapping is a provisional reference implementation choice
documented in architecture v3 section 55. It must not be misrepresented as an
original-Adze fact.
"""

from dataclasses import dataclass
from jax import Array
import jax
import jax.numpy as jnp


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class BlockLayout:
    block_id_per_carrier: Array
    within_block_pos: Array
    block_lengths: Array
    n_blocks: Array

    def tree_flatten(self):
        return (
            self.block_id_per_carrier,
            self.within_block_pos,
            self.block_lengths,
            self.n_blocks,
        ), None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        del aux_data
        return cls(*children)


@jax.tree_util.register_pytree_node_class
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
    block_count_overflow: Array
    block_length_overflow: Array

    def tree_flatten(self):
        return tuple(self.__dict__.values()), None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        del aux_data
        return cls(*children)


def boundaries_to_blocks(c_b: Array) -> BlockLayout:
    """Pure-JAX conversion from cut-after-carrier boundaries to blocks."""
    c_b = jnp.asarray(c_b)
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
    block_lengths = jnp.sum(jax_one_hot(block_id, c_b.shape[1]), axis=1, dtype=jnp.int32)
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
    """Eager wrapper around :func:`build_pack_metadata_core`.

    Host-side validation raises explicit errors. Compiled callers should use
    ``build_pack_metadata_core`` and inspect its overflow flags instead.
    """
    c_b = jnp.asarray(c_b)
    activity = jnp.asarray(activity)
    if c_b.ndim != 2 or activity.shape != c_b.shape:
        raise ValueError("c_b and activity must both have shape [batch, carrier]")
    if M_max <= 0 or K <= 0:
        raise ValueError("M_max and K must be positive")
    if not bool(jnp.all((c_b == 0) | (c_b == 1))):
        raise ValueError("c_b must contain only binary values")
    if not bool(jnp.all(c_b[:, -1] == 1)):
        raise ValueError("c_b must have a terminal cut after the final carrier")
    if not bool(jnp.all((activity == 0) | (activity == 1))):
        raise ValueError("activity must contain only binary values")

    metadata = build_pack_metadata_core(c_b, activity, M_max=M_max, K=K)
    if bool(jnp.any(metadata.block_count_overflow)):
        raise ValueError("generated block count exceeds M_max")
    if bool(jnp.any(metadata.block_length_overflow)):
        raise ValueError("generated block length exceeds K")
    return metadata


def build_pack_metadata_core(c_b: Array, activity: Array, *, M_max: int, K: int) -> PackMetadata:
    """Pure-JAX hard-pack construction with explicit overflow flags.

    ``M_max`` and ``K`` are static capacity arguments. The returned flags are
    per batch item and remain usable under ``jax.jit``; no Python conversion or
    host-side exception is performed here.
    """
    c_b = jnp.asarray(c_b)
    activity = jnp.asarray(activity)

    layout = boundaries_to_blocks(c_b)
    block_count_overflow = layout.n_blocks > M_max
    block_length_overflow = jnp.max(layout.block_lengths, axis=1) > K

    batch, capacity = c_b.shape
    packed_to_carrier = jnp.full((batch, M_max, K), -1, dtype=jnp.int32)
    packed_to_carrier = packed_to_carrier.at[
        jnp.arange(batch)[:, None],
        jnp.clip(layout.block_id_per_carrier, 0, M_max - 1),
        jnp.clip(layout.within_block_pos, 0, K - 1),
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
        block_count_overflow,
        block_length_overflow,
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
