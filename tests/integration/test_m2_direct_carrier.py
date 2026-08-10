from typing import cast

import jax
import jax.numpy as jnp

from adze_t.model.direct_carrier import DirectCarrierConfig, apply_chain, apply_core, make_gate


def test_affine_gaussian_public_core_has_fixed_shape_and_finite_output():
    config = DirectCarrierConfig(capacity=4, latent_dim=3, q=2)
    gate = make_gate(config)
    params = gate.init_params(jax.random.key(1))
    state = jnp.zeros(config.width)
    output = apply_core(config, params, state, jax.random.key(2))
    assert output.shape == state.shape
    assert bool(jnp.all(jnp.isfinite(output)))


def test_fixed_seed_and_split_keys_have_expected_public_sampling_semantics():
    config = DirectCarrierConfig(q=2)
    gate = make_gate(config)
    params = gate.init_params(jax.random.key(3))
    state = jnp.zeros(config.width)
    first = apply_core(config, params, state, jax.random.key(4))
    second = apply_core(config, params, state, jax.random.key(4))
    different = apply_core(config, params, state, jax.random.key(5))
    assert bool(jnp.array_equal(first, second))
    assert not bool(jnp.array_equal(first, different))


def test_q_one_is_one_public_base_transition():
    config = DirectCarrierConfig(q=1)
    gate = make_gate(config)
    params = gate.init_params(jax.random.key(6))
    state = jnp.arange(config.width, dtype=jnp.float32)
    inputs = {
        "continuous": state,
        "discrete": jnp.empty((0,), dtype=gate.input_ports["discrete"].dtype),
    }
    expected = cast(jax.Array, gate.sample(jax.random.key(7), inputs, params))
    observed = apply_core(config, params, state, jax.random.key(7))
    assert bool(jnp.array_equal(expected, observed))


def test_manual_tied_recurrence_and_public_chainfactor_are_finite():
    config = DirectCarrierConfig(q=2, tied=True)
    gate = make_gate(config)
    params = gate.init_params(jax.random.key(8))
    state = jnp.zeros(config.width)
    manual = apply_core(config, params, state, jax.random.key(9))
    composite = apply_chain(config, params, state, jax.random.key(9))
    assert manual.shape == composite.shape == state.shape
    assert bool(jnp.all(jnp.isfinite(manual)))
    assert bool(jnp.all(jnp.isfinite(composite)))
