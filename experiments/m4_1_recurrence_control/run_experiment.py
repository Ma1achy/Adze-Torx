"""M4.1 controlled recurrence experiment runner."""

from __future__ import annotations

import argparse

import jax

from adze_t.model.core import RecurrentCoreConfig
from experiments.m3_carrier_structure.data import make_data
from experiments.m4_1_recurrence_control.diagnostics import initial_map_report, trajectory_report
from experiments.m4_1_recurrence_control.model import M4Config, initialise_params
from experiments.m4_1_recurrence_control.toy_control import run_toy
from experiments.m4_1_recurrence_control.train import TrainConfig, keys_for, run


def data():
    structure = M4Config().structure
    return make_data(jax.random.key(1201), 512, structure), make_data(
        jax.random.key(1202), 256, structure
    )


def config(family: str, q: int) -> M4Config:
    return M4Config(
        core=RecurrentCoreConfig(
            width=18,
            q=q,
            family=family,
            eta=0.25,
            total_variance=0.01,
            noise_mode="fixed_total",
        )
    )


def run_one(
    family: str, q: int, train_data, val_data, seed: int, steps: int = 60, lr: float = 0.03
):
    return run(
        config(family, q),
        TrainConfig(seed=seed, steps=steps, learning_rate=lr),
        train_data,
        val_data,
        initialise_params,
    )


def print_initialization():
    for family in ("current", "residual", "identity_residual", "q_normalized_residual"):
        for q in (1, 2, 4):
            report = initial_map_report(config(family, q).core, jax.random.key(500 + q))
            print(
                "INIT",
                family,
                q,
                *(
                    f"{report[name]:.10f}"
                    for name in (
                        "state_error",
                        "linear_error",
                        "spectral_radius",
                        "largest_singular",
                        "nominal_variance",
                    )
                ),
            )


def print_toy():
    for family in ("current", "residual", "identity_residual", "q_normalized_residual"):
        for q in (1, 2, 4):
            print("TOY", family, q, f"{run_toy(family, q):.10f}")


def print_final(train_data, val_data):
    for family in ("identity_residual", "q_normalized_residual"):
        for q in (1, 2, 4):
            for seed in (0, 1, 2):
                _, metrics = run_one(family, q, train_data, val_data, seed, steps=240)
                by_step = {row["step"]: row["h_mse"] for row in metrics["validation_curve"]}
                print(
                    "FINAL",
                    family,
                    q,
                    seed,
                    f"{by_step.get(60, float('nan')):.8f}",
                    f"{metrics['best_val_h_mse']:.8f}",
                    metrics["best_val_step"],
                    f"{metrics['final_val_h_mse']:.8f}",
                    f"{metrics['total_train_seconds']:.4f}",
                    f"{metrics['steady_step_seconds']:.6f}",
                    metrics["parameter_count"],
                )


def print_lr(train_data, val_data):
    for lr in (0.01, 0.03, 0.1):
        for family in ("identity_residual", "q_normalized_residual"):
            for q in (1, 2, 4):
                _, metrics = run_one(family, q, train_data, val_data, 0, steps=60, lr=lr)
                print("LR", lr, family, q, f"{metrics['final_val_h_mse']:.8f}")


def print_early(train_data, val_data):
    for family in ("identity_residual", "q_normalized_residual"):
        for q in (1, 2, 4):
            _, metrics = run_one(family, q, train_data, val_data, 0, steps=60)
            records = {row["step"]: row for row in metrics["train_curve"]}
            for step in (0, 1, 5, 10, 20, 40, 59):
                row = records[step]
                print(
                    "EARLY",
                    family,
                    q,
                    step,
                    f"{row['loss']:.8f}",
                    f"{row['grad_norm']:.8f}",
                    f"{row['core_grad_norm']:.8f}",
                    f"{row['update_norm']:.8f}",
                    f"{row['delta_update_norm']:.8f}",
                    row["delta_update_parameter_ratio"],
                    f"{row['update_over_state_scale']:.8f}",
                    f"{row['step_seconds']:.6f}",
                )


def print_trajectories(train_data, val_data):
    for family in ("identity_residual", "q_normalized_residual"):
        for q in (2, 4):
            params, _ = run_one(family, q, train_data, val_data, 0, steps=240)
            reports = trajectory_report(
                config(family, q),
                params,
                val_data,
                keys_for(50000, 0, val_data.h.shape[0]),
            )
            for report in reports:
                print(
                    "TRAJ",
                    family,
                    q,
                    *(
                        report[name]
                        for name in (
                            "cycle",
                            "h_mse",
                            "mean_h_mse",
                            "delta_mse",
                            "improve_fraction",
                            "boundary_f1",
                            "length_accuracy",
                            "update_norm",
                            "state_norm",
                            "variance",
                        )
                    ),
                )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--section",
        choices=("init", "toy", "final", "lr", "early", "trajectory", "all"),
        default="all",
    )
    args = parser.parse_args()
    if args.section == "init" or args.section == "all":
        print_initialization()
    if args.section == "toy" or args.section == "all":
        print_toy()
    train_data, val_data = data()
    if args.section == "final" or args.section == "all":
        print_final(train_data, val_data)
    if args.section == "lr" or args.section == "all":
        print_lr(train_data, val_data)
    if args.section == "early" or args.section == "all":
        print_early(train_data, val_data)
    if args.section == "trajectory" or args.section == "all":
        print_trajectories(train_data, val_data)


if __name__ == "__main__":
    main()
