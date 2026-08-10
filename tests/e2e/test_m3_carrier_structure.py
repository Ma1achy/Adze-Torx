import jax
import jax.numpy as jnp

from adze_t.model.carrier import StructureConfig
from experiments.m3_carrier_structure.data import make_data
from experiments.m3_carrier_structure.model import M3Config
from experiments.m3_carrier_structure.train import TrainConfig, run


def test_m3_h_only_and_full_carrier_train_on_tiny_data():
    structure = StructureConfig()
    train = make_data(jax.random.key(30), 48, structure)
    validation = make_data(jax.random.key(31), 48, structure)
    for config in (
        M3Config(structure=structure, predict_boundary=False, predict_length=False),
        M3Config(structure=structure),
    ):
        _, metrics = run(
            config,
            TrainConfig(steps=8, batch_size=12, seed=32, rho_b=1.0, rho_length=1.0),
            train,
            validation,
        )
        assert bool(jnp.isfinite(metrics["h_mse"]))
        assert float(metrics["train_loss_final"]) < float(metrics["train_loss_initial"])
