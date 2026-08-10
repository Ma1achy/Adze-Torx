import jax

from experiments.m3_carrier_structure.data import make_data
from experiments.m4_recurrent_core.model import M4Config, initialise_params
from experiments.m4_recurrent_core.train import TrainConfig, run


def test_m4_tiny_training_step_is_finite_and_reduces_loss():
    config = M4Config()
    train_data = make_data(jax.random.key(100), 64, config.structure)
    val_data = make_data(jax.random.key(101), 32, config.structure)
    _, metrics = run(
        config,
        TrainConfig(steps=4, batch_size=16, seed=3),
        train_data,
        val_data,
        initialise_params,
    )
    assert float(metrics["train_loss_final"]) < float(metrics["train_loss_initial"])
    assert float(metrics["h_mse"]) == float(metrics["h_mse"])
