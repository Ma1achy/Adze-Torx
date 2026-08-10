import importlib.util

import pytest

pytestmark = pytest.mark.contract


def test_required_public_torx_symbols_when_installed():
    if importlib.util.find_spec("torx") is None:
        pytest.skip("Torx optional dependency is not installed")

    from adze_t.torx_api.contracts import inspect_public_surface

    surface = inspect_public_surface()
    missing = [name for name, present in surface.items() if not present]
    assert not missing, f"Missing public Torx symbols: {missing}"


def test_pnot_sample_and_log_probability_agree():
    import jax.numpy as jnp
    from torx.psc import PNOT

    gate = PNOT(0)
    theta = jnp.array([0.35])
    inputs = {"in": jnp.array([0], dtype=gate.input_ports["in"].dtype)}
    output = gate.sample(jnp.array([0, 1], dtype=jnp.uint32), inputs, theta)
    log_p = gate.log_probability(inputs, output, theta)
    log_dist = gate.get_log_output_distribution(inputs, theta)
    assert bool(jnp.isfinite(log_p))
    assert bool(jnp.allclose(jnp.exp(log_dist).sum(), 1.0))
    assert bool(jnp.allclose(log_p, gate.log_probability(inputs, output, theta)))


def test_public_seed_reproducibility_and_split_independence():
    import jax
    import jax.numpy as jnp
    from torx.psc import PNOT

    gate = PNOT(0)
    inputs = {"in": jnp.array([0], dtype=gate.input_ports["in"].dtype)}
    theta = jnp.array([0.35])
    key = jax.random.key(10)
    first = gate.sample(key, inputs, theta)
    second = gate.sample(key, inputs, theta)
    assert bool(jnp.array_equal(first, second))
    k1, k2 = jax.random.split(key)
    assert not bool(jnp.array_equal(k1, k2))
