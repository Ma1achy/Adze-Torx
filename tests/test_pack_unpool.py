import jax.numpy as jnp
import pytest

from adze_t.packing import build_pack_metadata, pack_values
from adze_t.unpool import unpool_values


def test_pack_unpool_roundtrip_carrier_ids():
    c_b = jnp.array([[0, 1, 0, 0, 1, 1]], dtype=jnp.int32)
    activity = jnp.array([[1, 1, 0, 1, 1, 1]], dtype=jnp.int32)
    meta = build_pack_metadata(c_b, activity, M_max=4, K=4)

    values = jnp.arange(6, dtype=jnp.float32)[None, :, None]
    packed = pack_values(values, meta)
    restored = unpool_values(packed, meta, C=6)

    assert restored.shape == values.shape
    assert restored.tolist() == values.tolist()


def test_inactive_hole_is_not_dropped_from_identity_map():
    c_b = jnp.array([[0, 1, 0, 0, 1, 1]], dtype=jnp.int32)
    activity = jnp.array([[1, 1, 0, 1, 1, 1]], dtype=jnp.int32)
    meta = build_pack_metadata(c_b, activity, M_max=4, K=4)

    # Carrier 2 is inactive but must have a packed query slot.
    m = int(meta.carrier_to_m[0, 2])
    k = int(meta.carrier_to_k[0, 2])
    assert int(meta.packed_to_carrier[0, m, k]) == 2
    assert bool(meta.query_mask[0, m, k])
    assert not bool(meta.kv_mask[0, m, k])
    assert not bool(meta.pool_mask[0, m, k])
    assert not bool(meta.emit_mask[0, m, k])


def test_overflow_is_explicit_not_truncation():
    c_b = jnp.array([[0, 0, 0, 0, 1]], dtype=jnp.int32)
    activity = jnp.ones((1, 5), dtype=jnp.int32)
    with pytest.raises((ValueError, RuntimeError)):
        build_pack_metadata(c_b, activity, M_max=4, K=4)


def test_block_length_overflow_is_explicit():
    c_b = jnp.array([[0, 0, 0, 0, 1]], dtype=jnp.int32)
    activity = jnp.ones((1, 5), dtype=jnp.int32)
    with pytest.raises((ValueError, RuntimeError), match="block length"):
        build_pack_metadata(c_b, activity, M_max=8, K=4)


def test_block_count_overflow_is_explicit():
    c_b = jnp.ones((1, 5), dtype=jnp.int32)
    activity = jnp.ones((1, 5), dtype=jnp.int32)
    with pytest.raises((ValueError, RuntimeError), match="block count"):
        build_pack_metadata(c_b, activity, M_max=4, K=4)
