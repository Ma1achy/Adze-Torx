import jax
import jax.numpy as jnp
from torx import DeterministicFactor, TiledFactor


def _base():
    def fn(inputs, _site_info):
        return inputs["x"] + 1.0

    return DeterministicFactor(
        fn=fn,
        input_ports={"x": jax.ShapeDtypeStruct((), jnp.float32)},
        output_spec=jax.ShapeDtypeStruct((), jnp.float32),
    )


def test_tiled_one_matches_base():
    base = _base()
    tiled = TiledFactor(base, 1, weight_tied=True)
    key = jnp.array([0, 1], dtype=jnp.uint32)
    assert jnp.array_equal(
        tiled.sample(key, {"x": jnp.array([2.0], dtype=jnp.float32)}, None),
        jnp.array([base.sample(key, {"x": jnp.array(2.0, dtype=jnp.float32)}, None)]),
    )


def test_tiled_manual_values_match():
    tiled = TiledFactor(_base(), 3, weight_tied=True)
    out = tiled.sample(
        jnp.array([0, 1], dtype=jnp.uint32),
        {"x": jnp.array([0.0, 1.0, 2.0], dtype=jnp.float32)},
        None,
    )
    assert jnp.array_equal(out, jnp.array([1.0, 2.0, 3.0]))
