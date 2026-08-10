"""Exact draft/refine attention masks for packed carrier positions."""

from jax import Array
import jax.numpy as jnp


def draft_attention_mask(
    block_id: Array,
    query_mask: Array,
    kv_mask: Array,
) -> Array:
    """Return boolean allowed-attention mask.

    Draft semantics:
    - within a logical block: bidirectional;
    - across blocks: keys may come only from the same or earlier block.
    """
    block_id, query_mask, kv_mask = (_flatten(block_id), _flatten(query_mask), _flatten(kv_mask))
    allowed = block_id[:, :, None] >= block_id[:, None, :]
    allowed &= query_mask[:, :, None] & kv_mask[:, None, :]
    return allowed


def refine_attention_mask(
    query_mask: Array,
    kv_mask: Array,
) -> Array:
    """Return global refine-mode boolean allowed-attention mask."""
    query_mask, kv_mask = _flatten(query_mask), _flatten(kv_mask)
    return query_mask[:, :, None] & kv_mask[:, None, :]


def _flatten(value: Array) -> Array:
    value = jnp.asarray(value)
    if value.ndim < 2:
        raise ValueError("mask inputs must have a batch dimension and positions")
    return value.reshape((value.shape[0], -1))
