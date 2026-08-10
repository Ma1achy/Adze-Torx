import jax
import jax.numpy as jnp

from adze_t.model.direct_carrier import DirectCarrierConfig
from experiments.m2_direct_carrier.data import make_data
from experiments.m2_direct_carrier.train import TrainConfig, run


def test_tiny_direct_carrier_training_reduces_loss():
    train = make_data(jax.random.key(10), 64)
    validation = make_data(jax.random.key(11), 64)
    config = DirectCarrierConfig(q=2, tied=True)
    _, metrics = run(config, TrainConfig(steps=12, batch_size=16, seed=2), train, validation)
    assert bool(jnp.isfinite(metrics["loss"]))
    assert bool(jnp.isfinite(metrics["train_loss_final"]))
    assert float(metrics["train_loss_final"]) < float(metrics["train_loss_initial"])
