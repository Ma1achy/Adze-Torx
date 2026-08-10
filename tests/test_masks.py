import jax.numpy as jnp

from adze_t.masking import draft_attention_mask, refine_attention_mask


def test_draft_block_causal_mask():
    # flattened positions: block 0 has two sites; block 1 has two; block 2 has one.
    block = jnp.array([[0, 0, 1, 1, 2]], dtype=jnp.int32)
    q = jnp.ones_like(block, dtype=bool)
    kv = jnp.ones_like(block, dtype=bool)

    mask = draft_attention_mask(block, q, kv)[0]

    # query in block 0: only block 0
    assert mask[0].tolist() == [True, True, False, False, False]
    # query in block 1: blocks 0 and 1
    assert mask[2].tolist() == [True, True, True, True, False]
    # query in block 2: all earlier and same blocks
    assert mask[4].tolist() == [True, True, True, True, True]


def test_refine_is_global_over_active_kv():
    q = jnp.array([[1, 1, 1]], dtype=bool)
    kv = jnp.array([[1, 0, 1]], dtype=bool)
    mask = refine_attention_mask(q, kv)[0]
    assert mask.tolist() == [
        [True, False, True],
        [True, False, True],
        [True, False, True],
    ]


def test_invalid_query_has_no_attention():
    q = jnp.array([[1, 0]], dtype=bool)
    kv = jnp.array([[1, 1]], dtype=bool)
    mask = refine_attention_mask(q, kv)[0]
    assert mask[0].tolist() == [True, True]
    assert mask[1].tolist() == [False, False]
