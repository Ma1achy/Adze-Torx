"""Run the M4.3 Q/depth matrix and write raw numerical evidence."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import jax

from experiments.m4_2_nonlinear_looped_core.model import NonlinearCoreConfig, initialise_params
from experiments.m4_3_hard_task_depth_sweep.diagnostics import (
    convergence_limited,
    initial_report,
    paired_t_interval,
    trajectory_report,
)
from experiments.m4_3_hard_task_depth_sweep.train import TrainConfig, run

OUT = Path(__file__).with_name("raw_results.json")
Q_VALUES = (1, 2, 4, 8, 12)
DEPTHS = (4, 8, 12)
SEEDS = (0, 1, 2)


def summary(result):
    return {
        key: result[key]
        for key in (
            "best",
            "final",
            "curves",
            "validation",
            "finite_failures",
            "parameter_count",
            "compile_seconds",
            "steady_step_seconds",
            "total_seconds",
        )
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args()
    seeds = tuple(range(args.seeds))
    report = {
        "environment": {
            "python": sys.version.split()[0],
            "jax": jax.__version__,
            "devices": [str(device) for device in jax.devices()],
            "dtype": "float32",
        },
        "initialization": {},
        "runs": {},
        "paired": {},
        "trajectories": {},
        "extended": {},
    }
    for nonlinear in (True, False):
        label = "nonlinear" if nonlinear else "affine"
        for q in Q_VALUES:
            config = NonlinearCoreConfig(
                width=64,
                q=q,
                eta=0.75,
                total_variance=0.04,
                nonlinear=nonlinear,
                active_mask=(1.0,) * 8 + (0.0,) * 56,
            )
            params = initialise_params(config, jax.random.PRNGKey(100 + q))
            report["initialization"][f"{label}-q{q}"] = initial_report(
                config, params, jax.random.PRNGKey(200 + q)
            )
            for depth in DEPTHS:
                key = f"{label}-k{depth}-q{q}"
                results = []
                for seed in seeds:
                    result = run(config, TrainConfig(steps=args.steps, seed=seed), depth)
                    results.append(summary(result))
                    if seed == seeds[0] and depth == 8 and q in (1, 8, 12):
                        report["trajectories"][key] = trajectory_report(
                            result, jax.random.split(jax.random.PRNGKey(8000 + q), 512)
                        )
                report["runs"][key] = results
    for nonlinear in (True, False):
        label = "nonlinear" if nonlinear else "affine"
        for depth in DEPTHS:
            q1 = report["runs"][f"{label}-k{depth}-q1"]
            q1_values = [row["best"]["loss"] for row in q1]
            for q in Q_VALUES[1:]:
                values = [row["best"]["loss"] for row in report["runs"][f"{label}-k{depth}-q{q}"]]
                report["paired"][f"{label}-k{depth}-q{q}"] = paired_t_interval(q1_values, values)
            if any(convergence_limited(row["validation"]) for row in q1):
                report["extended"][f"{label}-k{depth}"] = "required_common_extended_budget"
    if report["extended"]:
        extended_results = {}
        for nonlinear in (True, False):
            label = "nonlinear" if nonlinear else "affine"
            for depth in DEPTHS:
                for q in Q_VALUES:
                    config = NonlinearCoreConfig(
                        width=64,
                        q=q,
                        eta=0.75,
                        total_variance=0.04,
                        nonlinear=nonlinear,
                        active_mask=(1.0,) * 8 + (0.0,) * 56,
                    )
                    extended_results[f"{label}-k{depth}-q{q}"] = [
                        summary(run(config, TrainConfig(steps=240, seed=seed), depth))
                        for seed in seeds
                    ]
        report["extended_results"] = extended_results
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
