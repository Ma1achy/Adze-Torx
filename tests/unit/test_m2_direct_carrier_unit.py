import jax
import jax.numpy as jnp
import pytest

from adze_t.model.direct_carrier import DirectCarrierConfig
from experiments.m2_direct_carrier.data import corrupt, make_data
from experiments.m2_direct_carrier.model import no_update_prediction, reconstruction_loss


def test_m2_config_and_data_shapes_are_fixed():
    config = DirectCarrierConfig(capacity=5, latent_dim=2, q=4)
    data = make_data(jax.random.key(1), 7, config.capacity, config.latent_dim)
    assert data.shape == (7, config.width)


def test_corruption_is_reproducible_and_no_update_is_exactly_corruption():
    clean = make_data(jax.random.key(2), 4)
    key = jax.random.key(3)
    expected = corrupt(clean, key, 0.6, 0.5)
    observed = no_update_prediction(clean, key, 0.6, 0.5)
    assert bool(jnp.array_equal(expected, observed))
    assert bool(jnp.array_equal(observed, no_update_prediction(clean, key, 0.6, 0.5)))


def test_zero_corruption_no_update_is_near_perfect():
    clean = make_data(jax.random.key(4), 8)
    observed = no_update_prediction(clean, jax.random.key(5), 1.0, 0.0)
    assert float(reconstruction_loss(observed, clean)) == 0.0


def test_invalid_q_is_rejected():
    with pytest.raises(ValueError):
        DirectCarrierConfig(q=0)
