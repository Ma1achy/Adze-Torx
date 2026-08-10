import jax
import jax.numpy as jnp
from torx import ChainFactor, DeterministicFactor


def _base():
    def fn(inputs, _site_info):
        return inputs["x"] + 1.0

    return DeterministicFactor(
        fn=fn,
        input_ports={"x": jax.ShapeDtypeStruct((), jnp.float32)},
        output_spec=jax.ShapeDtypeStruct((), jnp.float32),
    )


def test_chain_depth_one_matches_base():
    base = _base()
    chain = ChainFactor(base, 1, "x", weight_tied=True)
    inputs = {"x": jnp.array(2.0, dtype=jnp.float32)}
    key = jnp.array([0, 1], dtype=jnp.uint32)
    assert jnp.array_equal(base.sample(key, inputs, None), chain.sample(key, inputs, None))


def test_chain_recurrence_matches_expected_manual_recurrence():
    chain = ChainFactor(_base(), 4, "x", weight_tied=True)
    out = chain.sample(
        jnp.array([0, 1], dtype=jnp.uint32),
        {"x": jnp.array(0.0, dtype=jnp.float32)},
        None,
    )
    assert out == 4.0
