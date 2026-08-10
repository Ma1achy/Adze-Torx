"""Run the compact M2 comparison and print machine-readable summaries."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from adze_t.model.direct_carrier import DirectCarrierConfig
from experiments.m2_direct_carrier.data import corruption_levels, make_data
from experiments.m2_direct_carrier.train import TrainConfig, control_loss, run


def main() -> None:
    print("python/jax/device", jax.__version__, jax.devices())
    train = make_data(jax.random.key(20), 256)
    validation = make_data(jax.random.key(21), 256)
    for level, (alpha, sigma) in corruption_levels().items():
        baseline = control_loss(validation, 700, alpha, sigma)
        print("level", level, "baseline", baseline)
        for q in (1, 2, 4):
            for tied in (True, False) if q == 4 else (True,):
                values = []
                for seed in (0, 1, 2):
                    model = DirectCarrierConfig(q=q, tied=tied)
                    _, metrics = run(
                        model, TrainConfig(seed=seed, alpha=alpha, sigma=sigma), train, validation
                    )
                    values.append(float(metrics["loss"]))
                print("model", q, tied, "val", values, "mean", float(jnp.mean(jnp.array(values))))
        deterministic = []
        for seed in (0, 1, 2):
            _, metrics = run(
                DirectCarrierConfig(q=1),
                TrainConfig(seed=seed, alpha=alpha, sigma=sigma, stochastic=False),
                train,
                validation,
            )
            deterministic.append(float(metrics["loss"]))
        print("deterministic", deterministic, "mean", float(jnp.mean(jnp.array(deterministic))))


if __name__ == "__main__":
    main()
