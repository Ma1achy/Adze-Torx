"""Run the bounded M4.4 semantics/fidelity matrix."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import jax
import jax.numpy as jnp

from experiments.m4_4_faithful_loop_core.diagnostics import initial_report, trajectory
from experiments.m4_4_faithful_loop_core.model import FaithfulConfig, initialise_params
from experiments.m4_4_faithful_loop_core.train import TrainConfig, run

OUT = Path(__file__).with_name("raw_results.json")


def config(q, fixed, progress=False, scratch=24, nonlinear=True):
    return FaithfulConfig(
        q=q,
        fixed_horizon=fixed,
        progress=progress,
        scratch_width=scratch,
        nonlinear=nonlinear,
    )


def summary(result):
    return {
        k: result[k]
        for k in (
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
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args()
    seeds = range(args.seeds)
    report = {
        "environment": {
            "python": sys.version.split()[0],
            "jax": jax.__version__,
            "devices": [str(x) for x in jax.devices()],
            "dtype": "float32",
        },
        "initialization": {},
        "runs": {},
        "trajectories": {},
        "progress_pilot": {},
        "test_time_q": {},
    }
    for fixed in (True, False):
        semantics = "fixed_horizon" if fixed else "compute_scaling"
        for q in (1, 2, 4, 8, 12):
            c = config(q, fixed)
            p = initialise_params(c, jax.random.PRNGKey(100 + q))
            state = jnp.zeros((92,)).at[:8].set(jnp.linspace(-0.5, 0.5, 8))
            report["initialization"][f"faithful-{semantics}-q{q}"] = initial_report(c, p, state)
        for q in (1, 2, 4, 8, 12):
            for depth in (8, 12):
                key = f"faithful-{semantics}-k{depth}-q{q}"
                results = []
                for seed in seeds:
                    results.append(
                        summary(
                            run(config(q, fixed), TrainConfig(steps=args.steps, seed=seed), depth)
                        )
                    )
                report["runs"][key] = results
    for fixed in (True, False):
        semantics = "fixed_horizon" if fixed else "compute_scaling"
        for family in ("minimal", "faithful"):
            for q in (1, 4, 8):
                key = f"{family}-{semantics}-k8-q{q}"
                results = [
                    summary(
                        run(
                            config(q, fixed),
                            TrainConfig(steps=args.steps, seed=seed),
                            8,
                            family=family,
                        )
                    )
                    for seed in seeds
                ]
                report["runs"][key] = results
    for q in (1, 4, 8):
        c = config(q, False)
        result = run(c, TrainConfig(steps=args.steps, seed=0), 8)
        report["trajectories"][f"faithful-compute-k8-q{q}"] = trajectory(
            result, jax.random.split(jax.random.PRNGKey(7000 + q), 256)
        )
    for progress in (False, True):
        key = f"progress-{progress}"
        report["progress_pilot"][key] = [
            summary(
                run(
                    config(8, False, progress=progress), TrainConfig(steps=args.steps, seed=seed), 8
                )
            )
            for seed in seeds
        ]
    for q in (1, 2, 4, 8, 12):
        result = run(config(4, False), TrainConfig(steps=args.steps, seed=0), 8)
        evaluation_config = config(q, False)
        evaluation = result["params"]
        batch = result["validation_batch"]
        from experiments.m4_4_faithful_loop_core.model import deterministic_trace

        outputs = jax.vmap(
            lambda x, current=evaluation, cfg=evaluation_config: deterministic_trace(
                cfg, current, x
            )[0]
        )(batch.initial)
        report["test_time_q"][f"train4-test{q}"] = float(
            jnp.mean((outputs[:, :8] - batch.target[:, :8]) ** 2)
        )
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
