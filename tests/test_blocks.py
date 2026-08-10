import jax.numpy as jnp

from adze_t.packing import boundaries_to_blocks


def test_blocks_worked_example():
    # cut-after-i convention; final entry is terminal sentinel.
    c_b = jnp.array([[0, 1, 0, 0, 1, 1]], dtype=jnp.int32)
    out = boundaries_to_blocks(c_b)
    assert out.block_id_per_carrier.tolist() == [[0, 0, 1, 1, 1, 2]]
    assert out.within_block_pos.tolist() == [[0, 1, 0, 1, 2, 0]]
    assert out.n_blocks.tolist() == [3]


def test_no_internal_cuts_is_one_block():
    c_b = jnp.array([[0, 0, 0, 1]], dtype=jnp.int32)
    out = boundaries_to_blocks(c_b)
    assert out.block_id_per_carrier.tolist() == [[0, 0, 0, 0]]
    assert out.n_blocks.tolist() == [1]


def test_cut_after_every_site():
    c_b = jnp.array([[1, 1, 1, 1]], dtype=jnp.int32)
    out = boundaries_to_blocks(c_b)
    assert out.block_id_per_carrier.tolist() == [[0, 1, 2, 3]]
    assert out.n_blocks.tolist() == [4]
