"""Run paired-root Phase-D zero-shot portability evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import pickle
from typing import Any

import jax
import jax.numpy as jnp

from adze_t.backends.mapping import deterministic_to_torx
from adze_t.config import REFERENCE_SMALL_V0
from adze_t.evaluation import (
    aggregate_root_chunks,
    paired_chunk_statistics,
    phase_d_root,
    phase_d_stage_names,
    student_t_summary,
)

from run_phase_b import dataset


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results" / "phase_d" / "d1"
CHECKPOINTS = ROOT / "results" / "phase_b" / "checkpoints"
LAMBDA_ROOT_COUNTS = ((0.0, 1), (0.1, 16), (0.25, 16), (0.5, 16), (1.0, 32))
TASK_CONFIG = {"copy": (821, 4100), "reverse": (831, 4200)}
CHUNK = 32


def _serialise(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    array = jax.device_get(value)
    return float(array) if getattr(array, "ndim", 0) == 0 else array.tolist()


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, default=_serialise) + "\n",
        encoding="utf-8",
    )


def _load(path: Path) -> Any:
    with path.open("rb") as stream:
        value = pickle.load(stream)  # noqa: S301 - trusted committed artifact
    return jax.tree.map(jnp.asarray, value)


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _root_record(result: dict[str, Any], root_index: int, root: jax.Array) -> dict[str, Any]:
    stage_names = phase_d_stage_names()
    return {
        "root_index": root_index,
        "root_key": jax.random.key_data(root),
        "loss": result["loss"],
        "byte_accuracy": result["byte_accuracy"],
        "exact_sequence_accuracy": result["exact_sequence_accuracy"],
        "nonfinite_rate": result["nonfinite_rate"],
        "stages": {
            name: {"signal_rms": signal, "perturbation_rms": perturbation}
            for name, signal, perturbation in zip(
                stage_names, result["signal_rms"], result["perturbation_rms"], strict=True
            )
        },
    }


def _summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {
        name: student_t_summary([float(record[name]) for record in records])
        for name in ("loss", "byte_accuracy", "exact_sequence_accuracy", "nonfinite_rate")
    }
    summary["stages"] = {
        stage: {
            "signal_rms": student_t_summary(
                [float(record["stages"][stage]["signal_rms"]) for record in records]
            ),
            "perturbation_rms": student_t_summary(
                [float(record["stages"][stage]["perturbation_rms"]) for record in records]
            ),
        }
        for stage in phase_d_stage_names()
    }
    return summary


def run_task(task: str) -> dict[str, Any]:
    validation_seed, base_root_seed = TASK_CONFIG[task]
    checkpoint_path = CHECKPOINTS / f"{task}.pkl"
    deterministic = _load(checkpoint_path)
    params, _ = deterministic_to_torx(deterministic)
    prompt, target = dataset(task, 256, validation_seed)
    evaluate = jax.jit(paired_chunk_statistics, static_argnames=("config",))
    lambda_results = []
    distributions: dict[str, Any] = {}
    max_roots = max(count for _, count in LAMBDA_ROOT_COUNTS)
    roots = [phase_d_root(base_root_seed, index) for index in range(max_roots)]
    for lambda_op, root_count in LAMBDA_ROOT_COUNTS:
        records = []
        for root_index, root in enumerate(roots[:root_count]):
            chunks = []
            weights = []
            for start in range(0, prompt.shape[0], CHUNK):
                chunk_result = evaluate(
                    params,
                    prompt[start : start + CHUNK],
                    target[start : start + CHUNK],
                    root,
                    jnp.arange(start, min(start + CHUNK, prompt.shape[0]), dtype=jnp.uint32),
                    jnp.asarray(lambda_op, dtype=jnp.float32),
                    config=REFERENCE_SMALL_V0,
                )
                chunks.append(chunk_result)
                weights.append(min(CHUNK, prompt.shape[0] - start))
            result = aggregate_root_chunks(chunks, weights)
            jax.block_until_ready(result["loss"])
            record = _root_record(result, root_index, root)
            records.append(record)
            print(
                f"{task} lambda={lambda_op:g} root={root_index + 1}/{root_count} "
                f"byte={float(result['byte_accuracy']):.6f} loss={float(result['loss']):.6f}",
                flush=True,
            )
        summary = _summarize(records)
        lambda_results.append(
            {
                "lambda_op": lambda_op,
                "root_count": root_count,
                "root_indices": list(range(root_count)),
                **summary,
            }
        )
        distributions[str(lambda_op)] = records
    result = {
        "task": task,
        "validation_seed": validation_seed,
        "examples": 256,
        "chunk_size": CHUNK,
        "base_root_seed": base_root_seed,
        "paired_nested_roots": True,
        "all_root_keys": [jax.random.key_data(root) for root in roots],
        "checkpoint": str(checkpoint_path.relative_to(ROOT)),
        "checkpoint_sha256": _hash(checkpoint_path),
        "lambda_results": lambda_results,
    }
    _write(OUTPUT / f"{task}.json", result)
    _write(OUTPUT / f"{task}_root_distributions.json", distributions)
    return result


def _lambda_one(result: dict[str, Any]) -> dict[str, Any]:
    return next(item for item in result["lambda_results"] if item["lambda_op"] == 1.0)


def write_decision() -> None:
    paths = [OUTPUT / f"{task}.json" for task in TASK_CONFIG]
    if not all(path.exists() for path in paths):
        return
    results = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    nonfinite = any(
        item["nonfinite_rate"]["mean"] > 0
        for result in results
        for item in result["lambda_results"]
    )
    lambda_one = {
        result["task"]: _lambda_one(result)["byte_accuracy"]["mean"] for result in results
    }
    if nonfinite:
        decision = "D1_PORTABILITY_NUMERICAL_FAILURE"
    elif all(value >= 0.9 for value in lambda_one.values()):
        decision = "D1_PORTABILITY_PASS"
    else:
        decision = "D1_PORTABILITY_DEGRADED"
    (OUTPUT / "D1_PORTABILITY.md").write_text(
        "# D1 — zero-shot portability\n\n"
        f"Decision: **{decision}**.\n\n"
        f"Lambda-one root-mean byte accuracy: COPY `{lambda_one['copy']:.6f}`, "
        f"REVERSE `{lambda_one['reverse']:.6f}`. Evaluations use paired, nested roots and "
        "32-example chunks; JSON records include Student-t intervals, exact accuracy, "
        "nonfinite rates, and stage RMS diagnostics.\n",
        encoding="utf-8",
    )
    print(decision, flush=True)
    if nonfinite:
        raise SystemExit(2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("tasks", nargs="*")
    args = parser.parse_args()
    tasks = args.tasks or tuple(TASK_CONFIG)
    unknown = set(tasks) - set(TASK_CONFIG)
    if unknown:
        parser.error(f"unknown tasks: {sorted(unknown)}")
    for task in tasks:
        run_task(task)
    write_decision()


if __name__ == "__main__":
    main()
