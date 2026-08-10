import jax.numpy as jnp

from adze_t.packing import build_pack_metadata


def test_inactive_site_query_active_kv_pool_emit_inactive():
    c_b = jnp.array([[0, 0, 1]], dtype=jnp.int32)
    activity = jnp.array([[1, 0, 1]], dtype=jnp.int32)
    meta = build_pack_metadata(c_b, activity, M_max=3, K=3)

    m = int(meta.carrier_to_m[0, 1])
    k = int(meta.carrier_to_k[0, 1])

    assert bool(meta.query_mask[0, m, k]) is True
    assert bool(meta.kv_mask[0, m, k]) is False
    assert bool(meta.pool_mask[0, m, k]) is False
    assert bool(meta.emit_mask[0, m, k]) is False
