from typing import cast

import jax
import jax.numpy as jnp

from adze_t.model.carrier import StructureConfig
from experiments.m3_carrier_structure.data import StructuredCarrier, make_data
from experiments.m3_carrier_structure.model import (
    M3Config,
    initialise_params,
    loss_one,
    predict_one,
)


def _sample():
    structure = StructureConfig()
    data = make_data(jax.random.key(20), 2, structure)
    target = StructuredCarrier(data.h[0], data.b[0], data.length[0])
    return structure, target


def test_full_m3_forward_and_gradient_are_finite_and_fixed_shape():
    structure, target = _sample()
    config = M3Config(structure=structure)
    params = initialise_params(config, jax.random.key(21))
    value, grads = jax.value_and_grad(loss_one)(
        params, target, jax.random.key(22), config, 0.6, 0.5, 1.0, 1.0
    )
    prediction = predict_one(params, target, jax.random.key(22), config, 0.6, 0.5, 1.0, 1.0)
    assert prediction.h.shape == target.h.shape
    boundary_logits = cast(jax.Array, prediction.boundary_logits)
    length_logits = cast(jax.Array, prediction.length_logits)
    assert boundary_logits.shape == (structure.capacity, 2)
    assert length_logits.shape == (structure.capacity, structure.max_length + 1)
    assert bool(jnp.isfinite(value))
    assert bool(
        jnp.all(jnp.asarray([jnp.all(jnp.isfinite(x)) for x in jax.tree_util.tree_leaves(grads)]))
    )


def test_structural_heads_do_not_change_torx_content_core_output():
    structure, target = _sample()
    config = M3Config(structure=structure)
    params = initialise_params(config, jax.random.key(23))
    changed = dict(params)
    changed["boundary"] = {key: value + 1.0 for key, value in params["boundary"].items()}
    changed["length"] = {key: value - 1.0 for key, value in params["length"].items()}
    first = predict_one(params, target, jax.random.key(24), config, 0.6, 0.5, 1.0, 1.0)
    second = predict_one(changed, target, jax.random.key(24), config, 0.6, 0.5, 1.0, 1.0)
    assert bool(jnp.array_equal(first.h, second.h))
    first_boundary = cast(jax.Array, first.boundary_logits)
    second_boundary = cast(jax.Array, second.boundary_logits)
    first_length = cast(jax.Array, first.length_logits)
    second_length = cast(jax.Array, second.length_logits)
    assert not bool(jnp.array_equal(first_boundary, second_boundary))
    assert not bool(jnp.array_equal(first_length, second_length))
