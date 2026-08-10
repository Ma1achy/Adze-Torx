"""Run the bounded M4.2 experiment matrix and emit a JSON report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax

from experiments.m4_2_nonlinear_looped_core.diagnostics import (
    initial_map_report,
    nonlinearity_report,
    trajectory_metrics,
)
from experiments.m4_2_nonlinear_looped_core.model import (
    M42Config,
    NonlinearCoreConfig,
    initialise_params,
)
from experiments.m4_2_nonlinear_looped_core.train import (
    TrainConfig,
    run_composition,
    run_reconstruction,
)

OUT = Path(__file__).with_name("raw_results.json")


def _summary(result):
    return {
        "best": result["best"],
        "final": result["final"],
        "parameter_count": result["parameter_count"],
        "finite_failures": result["finite_failures"],
        "compile_seconds": result["compile_seconds"],
        "steady_step_seconds": result["steady_step_seconds"],
        "total_seconds": result["total_seconds"],
        "curves": result["curves"],
        "validation": result["validation"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args()
    report = {
        "environment": {
            "python": __import__("sys").version.split()[0],
            "jax": jax.__version__,
            "devices": [str(device) for device in jax.devices()],
            "dtype": "float32",
        },
        "initialization": {},
        "reconstruction": {},
        "composition": {},
        "affine_control": {},
        "diagnostics": {},
        "matched_depth": {},
    }
    for nonlinear in (True, False):
        label = "nonlinear" if nonlinear else "affine"
        for q in (1, 2, 4):
            core = NonlinearCoreConfig(nonlinear=nonlinear, q=q)
            params = initialise_params(core, jax.random.PRNGKey(100 + q))
            report["initialization"][f"{label}-q{q}"] = initial_map_report(
                core, params, jax.random.PRNGKey(200 + q)
            )
            if nonlinear:
                runs = []
                for seed in range(args.seeds):
                    result = run_reconstruction(
                        M42Config(core=core), TrainConfig(steps=args.steps, seed=seed)
                    )
                    runs.append(_summary(result))
                report["reconstruction"][f"q{q}"] = runs
            for depth in (1, 2, 4):
                runs = []
                for seed in range(args.seeds):
                    composition_core = NonlinearCoreConfig(
                        nonlinear=nonlinear,
                        q=q,
                        active_mask=(1.0, 1.0, 1.0) + (0.0,) * 15,
                    )
                    result = run_composition(
                        composition_core, TrainConfig(steps=args.steps, seed=seed), depth
                    )
                    runs.append(_summary(result))
                    if nonlinear and seed == 0 and depth == 4 and q in (1, 4):
                        report["diagnostics"][f"nonlinear-q{q}"] = {
                            "nonlinearity": nonlinearity_report(composition_core, result["params"]),
                            "trajectory": trajectory_metrics(
                                composition_core,
                                result["params"],
                                result["validation_batch"],
                                jax.random.split(jax.random.PRNGKey(9900 + q), 512),
                            ),
                        }
                destination = report["composition"] if nonlinear else report["affine_control"]
                destination[f"k{depth}-q{q}"] = runs
    for blocks, q in ((4, 1), (2, 2), (1, 4)):
        core = NonlinearCoreConfig(
            blocks=blocks,
            q=q,
            active_mask=(1.0, 1.0, 1.0) + (0.0,) * 15,
        )
        result = run_composition(core, TrainConfig(steps=args.steps, seed=0), 4)
        report["matched_depth"][f"L{blocks}-q{q}"] = _summary(result)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
