"""Run the M4 diagnostics and final comparisons."""

from __future__ import annotations

import argparse
from typing import cast

import jax

from adze_t.model.core import RecurrentCoreConfig
from experiments.m3_carrier_structure.data import make_data
from experiments.m4_recurrent_core.diagnostics import trajectory_summary
from experiments.m4_recurrent_core.model import M4Config, initialise_params
from experiments.m4_recurrent_core.train import TrainConfig, run


def _value(value):
    return float(value)


def _data():
    structure = M4Config().structure
    return make_data(jax.random.key(1201), 512, structure), make_data(
        jax.random.key(1202), 256, structure
    )


def _config(family: str, q: int, tied: bool = True, eta: float = 0.25, cycle: bool = False):
    core = RecurrentCoreConfig(
        width=18,
        q=q,
        tied=tied,
        family=family,
        eta=eta,
        total_variance=0.01,
        noise_mode="fixed_total",
        cycle_conditioning=cycle,
    )
    return M4Config(core=core)


def _run_one(config, train_data, val_data, seed: int):
    _, metrics = run(
        config,
        TrainConfig(seed=seed),
        train_data,
        val_data,
        initialise_params,
    )
    keys = jax.random.split(jax.random.key(seed + 50000), val_data.h.shape[0])
    return metrics, keys


def _print_main(train_data, val_data):
    print("MAIN family q tied eta seed h_mse b_f1 len_acc params runtime")
    for family, eta in (("current", 1.0), ("residual", 0.25)):
        for q in (1, 2, 4):
            for seed in (0, 1, 2):
                config = _config(family, q, eta=eta)
                metrics, _ = _run_one(config, train_data, val_data, seed)
                print(
                    "MAIN",
                    family,
                    q,
                    True,
                    eta,
                    seed,
                    *(
                        f"{_value(metrics.get(name, float('nan'))):.8f}"
                        for name in (
                            "h_mse",
                            "boundary_unknown_f1",
                            "length_unknown_accuracy",
                            "parameter_count",
                            "compile_seconds_included",
                        )
                    ),
                )


def _print_eta(train_data, val_data):
    print("ETA eta q h_mse b_f1 len_acc params")
    for eta in (0.1, 0.25, 0.5, 1.0):
        config = _config("residual", 4, eta=eta)
        metrics, _ = _run_one(config, train_data, val_data, 0)
        print(
            "ETA",
            eta,
            4,
            *(
                f"{_value(metrics.get(name, float('nan'))):.8f}"
                for name in (
                    "h_mse",
                    "boundary_unknown_f1",
                    "length_unknown_accuracy",
                    "parameter_count",
                )
            ),
        )


def _print_ablation(train_data, val_data):
    print("ABLATION family q tied cycle h_mse b_f1 len_acc params")
    for family in ("current", "residual"):
        eta = 0.25 if family == "residual" else 1.0
        for q in (2, 4):
            for tied in (True, False):
                config = _config(family, q, tied=tied, eta=eta)
                metrics, _ = _run_one(config, train_data, val_data, 0)
                print(
                    "ABLATION",
                    family,
                    q,
                    tied,
                    False,
                    *(
                        f"{_value(metrics.get(name, float('nan'))):.8f}"
                        for name in (
                            "h_mse",
                            "boundary_unknown_f1",
                            "length_unknown_accuracy",
                            "parameter_count",
                        )
                    ),
                )
            config = _config(family, q, eta=eta, cycle=True)
            metrics, _ = _run_one(config, train_data, val_data, 0)
            print(
                "ABLATION",
                family,
                q,
                True,
                True,
                *(
                    f"{_value(metrics.get(name, float('nan'))):.8f}"
                    for name in (
                        "h_mse",
                        "boundary_unknown_f1",
                        "length_unknown_accuracy",
                        "parameter_count",
                    )
                ),
            )


def _print_dynamics(train_data, val_data):
    print(
        "DYNAMICS family q stochastic h_mse update_norm state_norm variance mean_h_mse nominal_var"
    )
    for family, eta in (("current", 1.0), ("residual", 0.25)):
        for q in (1, 2, 4):
            config = _config(family, q, eta=eta)
            params, _ = run(
                config,
                TrainConfig(seed=0),
                train_data,
                val_data,
                initialise_params,
            )
            target = val_data.h[0].reshape(-1)
            corruption_key = jax.random.key(9000 + q)
            initial = 0.6 * target + 0.5 * jax.random.normal(
                corruption_key, target.shape, dtype=target.dtype
            )
            summary = trajectory_summary(
                config.core, params["core"], initial, target, jax.random.key(9100 + q)
            )
            for stochastic in (True, False):
                key = "h_mse_by_cycle" if stochastic else "mean_h_mse_by_cycle"
                print(
                    "DYNAMICS",
                    family,
                    q,
                    stochastic,
                    f"{_value(cast(jax.Array, summary[key])[-1]):.8f}",
                    f"{_value(cast(jax.Array, summary['update_norm_by_cycle'])[-1]):.8f}",
                    f"{_value(cast(jax.Array, summary['state_norm_by_cycle'])[-1]):.8f}",
                    f"{_value(cast(jax.Array, summary['variance_by_cycle'])[-1]):.8f}",
                    f"{_value(cast(jax.Array, summary['mean_h_mse_by_cycle'])[-1]):.8f}",
                    f"{_value(summary['nominal_accumulated_variance']):.8f}",
                )


def _print_trajectory(train_data, val_data):
    print("TRAJECTORY family q cycle h_mse mean_h_mse update_norm state_norm variance")
    for family, eta in (("current", 1.0), ("residual", 0.25)):
        config = _config(family, 4, eta=eta)
        params, _ = run(config, TrainConfig(seed=0), train_data, val_data, initialise_params)
        target = val_data.h[0].reshape(-1)
        initial = 0.6 * target + 0.5 * jax.random.normal(
            jax.random.key(9200), target.shape, dtype=target.dtype
        )
        summary = trajectory_summary(
            config.core, params["core"], initial, target, jax.random.key(9201)
        )
        for cycle in range(5):
            print(
                "TRAJECTORY",
                family,
                4,
                cycle,
                *(
                    f"{_value(cast(jax.Array, summary[name])[cycle]):.8f}"
                    for name in (
                        "h_mse_by_cycle",
                        "mean_h_mse_by_cycle",
                        "update_norm_by_cycle",
                        "state_norm_by_cycle",
                        "variance_by_cycle",
                    )
                ),
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--section",
        choices=("main", "eta", "ablation", "dynamics", "trajectory", "all"),
        default="all",
    )
    args = parser.parse_args()
    train_data, val_data = _data()
    if args.section in {"main", "all"}:
        _print_main(train_data, val_data)
    if args.section in {"eta", "all"}:
        _print_eta(train_data, val_data)
    if args.section in {"ablation", "all"}:
        _print_ablation(train_data, val_data)
    if args.section in {"dynamics", "all"}:
        _print_dynamics(train_data, val_data)
    if args.section in {"trajectory", "all"}:
        _print_trajectory(train_data, val_data)


if __name__ == "__main__":
    main()
