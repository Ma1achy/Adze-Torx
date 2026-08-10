import jax
import jax.numpy as jnp

from adze_t.masking import draft_attention_mask, refine_attention_mask
from adze_t.packing import (
    boundaries_to_blocks,
    build_pack_metadata_core,
    pack_values,
)
from adze_t.unpool import unpool_values
from adze_t.state import carrier_invariant_flags


def _inputs():
    c_b = jnp.array([[0, 1, 0, 0, 1, 1]], dtype=jnp.int32)
    activity = jnp.array([[1, 1, 0, 1, 1, 1]], dtype=jnp.int32)
    return c_b, activity


def test_boundary_to_blocks_jit():
    out = jax.jit(boundaries_to_blocks)(_inputs()[0])
    assert out.block_id_per_carrier.tolist() == [[0, 0, 1, 1, 1, 2]]
    assert out.within_block_pos.tolist() == [[0, 1, 0, 1, 2, 0]]


def test_carrier_invariant_flags_jit():
    c_b, activity = _inputs()
    binary, terminal = jax.jit(carrier_invariant_flags)(c_b, activity)
    assert bool(binary)
    assert bool(terminal)


def test_pack_values_and_unpool_jit_with_inactive_hole():
    c_b, activity = _inputs()
    build = jax.jit(build_pack_metadata_core, static_argnames=("M_max", "K"))
    metadata = build(c_b, activity, M_max=4, K=4)
    values = jnp.arange(6, dtype=jnp.float32)[None, :, None]
    packed = jax.jit(pack_values)(values, metadata)
    restored = jax.jit(unpool_values, static_argnames=("C",))(packed, metadata, C=6)
    assert restored.tolist() == values.tolist()
    m, k = int(metadata.carrier_to_m[0, 2]), int(metadata.carrier_to_k[0, 2])
    assert metadata.query_mask[0, m, k]
    assert not metadata.kv_mask[0, m, k]


def test_masks_jit():
    block = jnp.array([[0, 0, 1, 1, 2]], dtype=jnp.int32)
    query = jnp.ones_like(block, dtype=bool)
    kv = jnp.array([[1, 0, 1, 1, 1]], dtype=bool)
    draft = jax.jit(draft_attention_mask)(block, query, kv)[0]
    refine = jax.jit(refine_attention_mask)(query, kv)[0]
    assert draft[0].tolist() == [True, False, False, False, False]
    assert refine[0].tolist() == [True, False, True, True, True]


def test_overflow_flags_are_reported_under_jit_without_tracer_errors():
    too_long = jnp.array([[0, 0, 0, 0, 1]], dtype=jnp.int32)
    activity = jnp.ones_like(too_long)
    build = jax.jit(build_pack_metadata_core, static_argnames=("M_max", "K"))
    metadata = build(too_long, activity, M_max=8, K=4)
    assert metadata.block_count_overflow.tolist() == [False]
    assert metadata.block_length_overflow.tolist() == [True]

    too_many = jnp.ones((1, 5), dtype=jnp.int32)
    metadata = build(too_many, activity, M_max=4, K=4)
    assert metadata.block_count_overflow.tolist() == [True]
    assert metadata.block_length_overflow.tolist() == [False]
