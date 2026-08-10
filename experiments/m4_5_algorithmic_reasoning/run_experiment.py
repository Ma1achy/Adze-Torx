"""Run the bounded M4.5 arithmetic/program matrix."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import jax

from experiments.m4_5_algorithmic_reasoning.arithmetic import make_data as make_arithmetic
from experiments.m4_5_algorithmic_reasoning.model import AlgorithmicConfig
from experiments.m4_5_algorithmic_reasoning.programs import make_data as make_program
from experiments.m4_5_algorithmic_reasoning.train import TrainConfig, fit, make_loss

OUT = Path(__file__).with_name("raw_results.json")
Q_VALUES = (1, 2, 4, 8, 12)
DEPTHS = (1, 2, 4, 8, 12)
PROGRAM_LENGTHS = (4, 8, 12)


def _summary(result):
    return {
        key: result[key]
        for key in (
            "best",
            "final",
            "validation",
            "curves",
            "finite_failures",
            "parameter_count",
            "compile_seconds",
            "steady_step_seconds",
            "total_seconds",
        )
    }


def _run_arithmetic(depth, q, mode, seed, steps, validation_size=64):
    train_batch = make_arithmetic(jax.random.PRNGKey(10000 + seed + depth), 16, depth)
    validation = make_arithmetic(jax.random.PRNGKey(20000 + depth), validation_size, depth)
    condition_width = (
        train_batch.all_conditioning.shape[-1]
        if mode == "all"
        else train_batch.cursor_conditioning.shape[-1]
    )
    config = AlgorithmicConfig(q=q, conditioning_width=condition_width, output_shape=(13, 10))
    loss_fn, metric_fn = make_loss(config, "arithmetic", mode)
    result = fit(
        config,
        TrainConfig(steps=steps, seed=seed),
        lambda key: make_arithmetic(key, 16, depth),
        validation,
        loss_fn,
        metric_fn,
    )
    return _summary(result)


def _run_program(length, q, mode, seed, steps, validation_size=64):
    validation = make_program(jax.random.PRNGKey(30000 + length), validation_size, length)
    train_batch = make_program(jax.random.PRNGKey(40000 + seed + length), 16, length)
    condition_width = (
        train_batch.all_conditioning.shape[-1]
        if mode == "all"
        else train_batch.cursor_conditioning.shape[-1]
    )
    config = AlgorithmicConfig(q=q, conditioning_width=condition_width, output_shape=(2, 16))
    loss_fn, metric_fn = make_loss(config, "program", mode)
    result = fit(
        config,
        TrainConfig(steps=steps, seed=seed),
        lambda key: make_program(key, 16, length),
        validation,
        loss_fn,
        metric_fn,
    )
    return _summary(result)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args()
    report = {
        "environment": {
            "python": sys.version.split()[0],
            "jax": jax.__version__,
            "devices": [str(device) for device in jax.devices()],
            "dtype": "float32",
        },
        "arithmetic_all_at_once": {},
        "arithmetic_cursor": {},
        "program_all_at_once": {},
        "program_cursor": {},
    }
    seeds = range(args.seeds)
    for depth in DEPTHS:
        for q in Q_VALUES:
            report["arithmetic_all_at_once"][f"d{depth}-q{q}"] = [
                _run_arithmetic(depth, q, "all", seed, args.steps) for seed in seeds
            ]
    # Cursor arithmetic is trained at complete Q=12; smaller/larger Q are
    # inference diagnostics and are evaluated by the focused runner below.
    for length in PROGRAM_LENGTHS:
        for q in Q_VALUES:
            report["program_all_at_once"][f"k{length}-q{q}"] = [
                _run_program(length, q, "all", seed, args.steps) for seed in seeds
            ]
        report["program_cursor"][f"k{length}-q{length}"] = [
            _run_program(length, length, "cursor", seed, args.steps) for seed in seeds
        ]
    for depth in (1, 8, 12):
        report["arithmetic_cursor"][f"d{depth}-q12"] = [
            _run_arithmetic(depth, 12, "cursor", seed, args.steps) for seed in seeds
        ]
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
