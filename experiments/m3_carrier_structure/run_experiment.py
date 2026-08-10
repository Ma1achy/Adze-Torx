"""Run the M3 ablations, corruption sweep, and shuffled-target control."""

from __future__ import annotations

import jax

from adze_t.model.carrier import StructureConfig
from experiments.m3_carrier_structure.data import make_data, shuffle_structure
from experiments.m3_carrier_structure.model import M3Config
from experiments.m3_carrier_structure.train import TrainConfig, class_baselines, run


def _variant(name: str) -> M3Config:
    return M3Config(
        predict_boundary="b" in name,
        predict_length="length" in name,
    )


def main() -> None:
    config = StructureConfig(capacity=6, latent_dim=3, max_length=3)
    train = make_data(jax.random.key(300), 256, config)
    validation = make_data(jax.random.key(301), 256, config)
    print("jax/device", jax.__version__, jax.devices())
    print("class_baselines", class_baselines(validation))

    variants = ("h", "h+b", "h+length", "h+b+length")
    for name in variants:
        values = []
        for seed in (0, 1, 2):
            _, metrics = run(
                _variant(name),
                TrainConfig(seed=seed, rho_b=0.5, rho_length=0.5),
                train,
                validation,
            )
            values.append({key: float(value) for key, value in metrics.items()})
        print("variant", name, values)

    for rho in (0.0, 0.25, 0.5, 0.75, 1.0):
        _, metrics = run(
            _variant("h+b+length"),
            TrainConfig(seed=0, rho_b=rho, rho_length=rho),
            train,
            validation,
        )
        print("sweep", rho, {key: float(value) for key, value in metrics.items()})

    for rho_b, rho_length in ((1.0, 0.25), (0.25, 1.0)):
        _, metrics = run(
            _variant("h+b+length"),
            TrainConfig(seed=0, rho_b=rho_b, rho_length=rho_length),
            train,
            validation,
        )
        print(
            "asymmetric", rho_b, rho_length, {key: float(value) for key, value in metrics.items()}
        )

    shuffled = shuffle_structure(train, jax.random.key(999))
    _, metrics = run(
        _variant("h+b+length"),
        TrainConfig(seed=0, rho_b=1.0, rho_length=1.0),
        shuffled,
        validation,
    )
    print(
        "shuffled_targets_unknown_observed", {key: float(value) for key, value in metrics.items()}
    )


if __name__ == "__main__":
    main()
