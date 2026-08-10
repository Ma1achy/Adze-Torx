import jax
import jax.numpy as jnp
import pytest

from adze_t.model.carrier import BOUNDARY_UNKNOWN, StructureConfig, validate_carrier_shapes
from adze_t.model.corruption import corrupt_boundary, corrupt_length, corrupt_structure
from experiments.m3_carrier_structure.data import make_data, shuffle_structure


def test_m3_generator_has_fixed_shapes_recoverable_boundaries_and_zero_lengths():
    config = StructureConfig()
    data = make_data(jax.random.key(10), 512, config)
    validate_carrier_shapes(data.h, data.b, data.length, config)
    boundary_jump = jnp.linalg.norm(data.h[:, 1:] - data.h[:, :-1], axis=-1)
    boundary_mask = data.b[:, 1:] == 1
    assert bool(jnp.all(data.b[:, 0] == 1))
    assert bool(jnp.any(data.length == 0))
    assert float(jnp.mean(boundary_jump[boundary_mask])) > float(
        jnp.mean(boundary_jump[~boundary_mask])
    )


def test_unknown_is_distinct_from_valid_boundary_and_length_zero():
    config = StructureConfig()
    assert BOUNDARY_UNKNOWN not in (0, 1)
    assert config.length_unknown != 0
    data = make_data(jax.random.key(11), 16, config)
    observed_b, observed_length = corrupt_structure(
        data.b, data.length, jax.random.key(12), 1.0, 1.0, config
    )
    assert bool(jnp.all(observed_b == BOUNDARY_UNKNOWN))
    assert bool(jnp.all(observed_length == config.length_unknown))
    assert bool(jnp.any(data.length == 0))
    assert data.h.shape == (*observed_b.shape, config.latent_dim)


def test_structural_corruption_limits_and_reproducibility():
    config = StructureConfig()
    data = make_data(jax.random.key(13), 8, config)
    b0 = corrupt_boundary(data.b, jax.random.key(14), 0.0)
    l0 = corrupt_length(data.length, jax.random.key(15), 0.0, config)
    b1 = corrupt_boundary(data.b, jax.random.key(14), 1.0)
    l1 = corrupt_length(data.length, jax.random.key(15), 1.0, config)
    assert bool(jnp.array_equal(b0, data.b))
    assert bool(jnp.array_equal(l0, data.length))
    assert bool(jnp.all(b1 == BOUNDARY_UNKNOWN))
    assert bool(jnp.all(l1 == config.length_unknown))
    assert bool(jnp.array_equal(b1, corrupt_boundary(data.b, jax.random.key(14), 1.0)))


def test_shuffle_control_preserves_shapes_without_clean_observation_semantics():
    config = StructureConfig()
    data = make_data(jax.random.key(16), 12, config)
    shuffled = shuffle_structure(data, jax.random.key(17))
    assert shuffled.h.shape == data.h.shape
    assert shuffled.b.shape == data.b.shape
    assert shuffled.length.shape == data.length.shape
    assert bool(jnp.array_equal(shuffled.h, data.h))


def test_invalid_structure_shape_is_rejected():
    config = StructureConfig()
    with pytest.raises(ValueError):
        validate_carrier_shapes(
            jnp.zeros((config.capacity, config.latent_dim + 1)),
            jnp.zeros(config.capacity, dtype=jnp.int32),
            jnp.zeros(config.capacity, dtype=jnp.int32),
            config,
        )
