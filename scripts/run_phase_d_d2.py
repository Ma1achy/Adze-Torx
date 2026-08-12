"""Run resumable Phase-D stochastic continuation from accepted B3 checkpoints."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import pickle
import time
from typing import Any

import jax
import jax.numpy as jnp

from adze_t.backends.mapping import deterministic_to_torx, torx_means_to_deterministic
from adze_t.config import REFERENCE_SMALL_V0
from adze_t.evaluation import (
    aggregate_root_chunks,
    paired_chunk_statistics,
    phase_d_root,
    phase_d_stage_names,
    student_t_summary,
)
from adze_t.objectives import adamw_init
from adze_t.training import make_fixed_structure_batch, stochastic_train_step

from run_phase_b import dataset


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "results" / "phase_d" / "d2"
WORK = ROOT / "results" / "runs" / "phase_d" / "d2"
SOURCE = ROOT / "results" / "phase_b" / "checkpoints"
CHECKPOINTS = (100, 250, 500, 1_000, 2_000, 5_000, 10_000)
TASK_CONFIG = {
    "copy": {"train_seed": 820, "valid_seed": 821, "training_root": 5100, "eval_root": 6100},
    "reverse": {
        "train_seed": 830,
        "valid_seed": 831,
        "training_root": 5200,
        "eval_root": 6200,
    },
}
BATCH_SIZE = 32


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


def _append(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, sort_keys=True, default=_serialise) + "\n")


def _load_params(path: Path) -> Any:
    with path.open("rb") as stream:
        return jax.tree.map(
            jnp.asarray,
            pickle.load(stream),  # noqa: S301 - trusted local working/committed artifact
        )


def _save_work(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as stream:
        pickle.dump(jax.device_get(state), stream, protocol=pickle.HIGHEST_PROTOCOL)


def _load_work(path: Path) -> dict[str, Any]:
    with path.open("rb") as stream:
        state = pickle.load(stream)  # noqa: S301 - trusted local working artifact
    return jax.tree.map(jnp.asarray, state)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _path_values(tree: Any, predicate) -> dict[str, Any]:
    return {
        jax.tree_util.keystr(path): value
        for path, value in jax.tree_util.tree_leaves_with_path(tree)
        if predicate(jax.tree_util.keystr(path))
    }


def _equal_paths(left: Any, right: Any, predicate) -> bool:
    left_values = _path_values(left, predicate)
    right_values = _path_values(right, predicate)
    return left_values.keys() == right_values.keys() and all(
        jnp.array_equal(value, right_values[path]) for path, value in left_values.items()
    )


def _root_record(result: dict[str, Any], root_index: int) -> dict[str, Any]:
    return {
        "root_index": root_index,
        "loss": result["loss"],
        "byte_accuracy": result["byte_accuracy"],
        "exact_sequence_accuracy": result["exact_sequence_accuracy"],
        "nonfinite_rate": result["nonfinite_rate"],
        "stages": {
            name: {"signal_rms": signal, "perturbation_rms": perturbation}
            for name, signal, perturbation in zip(
                phase_d_stage_names(),
                result["signal_rms"],
                result["perturbation_rms"],
                strict=True,
            )
        },
    }


def evaluate_roots(params, prompt, target, base_seed, count, evaluate):
    records = []
    for root_index in range(count):
        root = phase_d_root(base_seed, root_index)
        chunks = []
        weights = []
        for start in range(0, prompt.shape[0], BATCH_SIZE):
            chunks.append(
                evaluate(
                    params,
                    prompt[start : start + BATCH_SIZE],
                    target[start : start + BATCH_SIZE],
                    root,
                    jnp.asarray(1.0, dtype=jnp.float32),
                    config=REFERENCE_SMALL_V0,
                )
            )
            weights.append(min(BATCH_SIZE, prompt.shape[0] - start))
        result = aggregate_root_chunks(chunks, weights)
        jax.block_until_ready(result["loss"])
        records.append(_root_record(result, root_index))
    return records


def summarize(records):
    return {
        name: student_t_summary([float(record[name]) for record in records])
        for name in ("loss", "byte_accuracy", "exact_sequence_accuracy", "nonfinite_rate")
    }


def run_task(task: str, max_steps: int) -> dict[str, Any]:
    task_config = TASK_CONFIG[task]
    source_path = SOURCE / f"{task}.pkl"
    work_path = WORK / f"{task}.pkl"
    curve_path = EVIDENCE / f"{task}.jsonl"
    source_deterministic = _load_params(source_path)
    mapped, _ = deterministic_to_torx(source_deterministic)
    if work_path.exists():
        state = _load_work(work_path)
        params, moments, start_step = state["params"], state["moments"], int(state["step"])
    else:
        params = mapped
        zero = adamw_init(params)
        moments = (zero, zero)
        start_step = 0
        curve_path.unlink(missing_ok=True)
    initial_params = mapped
    initial_moments = (adamw_init(mapped), adamw_init(mapped))
    train_prompt, train_target = dataset(task, 65_536, task_config["train_seed"])
    valid_prompt, valid_target = dataset(task, 256, task_config["valid_seed"])
    update = jax.jit(stochastic_train_step, static_argnames=("config",))
    evaluate = jax.jit(paired_chunk_statistics, static_argnames=("config",))
    training_root = jax.random.PRNGKey(task_config["training_root"])
    started = time.monotonic()
    last_metrics = None
    final_records = None
    final_step = start_step
    for step in range(start_step + 1, max_steps + 1):
        offset = ((step - 1) * BATCH_SIZE) % train_prompt.shape[0]
        batch = make_fixed_structure_batch(
            train_prompt[offset : offset + BATCH_SIZE],
            train_target[offset : offset + BATCH_SIZE],
            config=REFERENCE_SMALL_V0,
        )
        params, moments, metrics = update(
            params,
            moments,
            step,
            batch,
            training_root,
            config=REFERENCE_SMALL_V0,
        )
        last_metrics = metrics
        if step not in CHECKPOINTS and step != max_steps:
            continue
        jax.block_until_ready(metrics["loss"])
        records = evaluate_roots(
            params, valid_prompt, valid_target, task_config["eval_root"], 8, evaluate
        )
        root_summary = summarize(records)
        candidate = float(root_summary["byte_accuracy"]["mean"]) >= 0.9
        if candidate:
            records = evaluate_roots(
                params, valid_prompt, valid_target, task_config["eval_root"], 32, evaluate
            )
            root_summary = summarize(records)
        row = {
            "task": task,
            "step": step,
            "lambda_op": 1.0,
            "root_count": len(records),
            "mc": root_summary,
            "train": metrics,
            "wall_clock_seconds": time.monotonic() - started,
        }
        _append(curve_path, row)
        _save_work(work_path, {"params": params, "moments": moments, "step": step})
        print(
            f"{task} step={step} roots={len(records)} "
            f"byte={root_summary['byte_accuracy']['mean']:.6f}",
            flush=True,
        )
        final_records = records
        final_step = step
        if (
            len(records) == 32
            and float(root_summary["byte_accuracy"]["mean"]) >= 0.9
            and float(root_summary["nonfinite_rate"]["mean"]) == 0.0
        ):
            break
    if last_metrics is None or final_records is None:
        raise RuntimeError("D2 did not reach an evaluation checkpoint")

    def mean_paths(path):
        return "['mean']" in path

    def rho_paths(path):
        return "['rho']" in path

    def teacher_paths(path):
        return path.startswith("['encoder']") and any(
            f"['{name}']" in path
            for name in (
                "byte_embed",
                "frontend",
                "target",
                "target_slot_embed",
                "target_carrier_embed",
                "target_pool",
                "target_h",
                "target_b",
                "target_l",
            )
        )

    mapped_round_trip = torx_means_to_deterministic(initial_params)
    checks = {
        "checkpoint_map_exact": _equal_paths(
            mapped_round_trip, source_deterministic, lambda path: True
        ),
        "rho_parameters_bitwise_fixed": _equal_paths(initial_params, params, rho_paths),
        "rho_first_moments_bitwise_fixed": _equal_paths(initial_moments[0], moments[0], rho_paths),
        "rho_second_moments_bitwise_fixed": _equal_paths(initial_moments[1], moments[1], rho_paths),
        "clean_teacher_parameters_bitwise_fixed": _equal_paths(
            initial_params, params, teacher_paths
        ),
        "permitted_means_changed": not _equal_paths(initial_params, params, mean_paths),
        "raw_rho_connectivity_nonzero": float(last_metrics["grad_rho_raw_norm"]) > 0,
        "applied_rho_gradient_zero": float(last_metrics["grad_rho_applied_norm"]) == 0,
        "finite_training_metrics": all(
            bool(jnp.all(jnp.isfinite(value))) for value in jax.tree_util.tree_leaves(last_metrics)
        ),
    }
    _write(EVIDENCE / f"{task}_final_32_roots.json", final_records)
    final_summary = summarize(final_records)
    result = {
        "task": task,
        "step": final_step,
        "source_checkpoint": str(source_path.relative_to(ROOT)),
        "source_checkpoint_sha256": _sha256(source_path),
        "working_checkpoint": str(work_path.relative_to(ROOT)),
        "working_checkpoint_sha256": _sha256(work_path),
        "training_root": task_config["training_root"],
        "evaluation_root_base": task_config["eval_root"],
        "final_mc": final_summary,
        "checks": checks,
        "passed": all(checks.values())
        and len(final_records) == 32
        and float(final_summary["byte_accuracy"]["mean"]) >= 0.9
        and float(final_summary["nonfinite_rate"]["mean"]) == 0.0,
    }
    _write(EVIDENCE / f"{task}_summary.json", result)
    return result


def write_decision() -> None:
    paths = [EVIDENCE / f"{task}_summary.json" for task in TASK_CONFIG]
    if not all(path.exists() for path in paths):
        return
    results = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    if all(result["passed"] for result in results):
        decision = "D2_STOCHASTIC_CONTINUATION_PASS"
    elif any(result["final_mc"]["nonfinite_rate"]["mean"] > 0 for result in results):
        decision = "D2_STOCHASTIC_CONTINUATION_NUMERICAL_FAILURE"
    else:
        decision = "D2_STOCHASTIC_CONTINUATION_UNRESOLVED"
    (EVIDENCE / "D2_STOCHASTIC_CONTINUATION.md").write_text(
        "# D2 — stochastic continuation\n\n"
        f"Decision: **{decision}**.\n\n"
        + "\n".join(
            f"- {result['task'].upper()}: step {result['step']}, 32-root byte accuracy "
            f"`{result['final_mc']['byte_accuracy']['mean']:.6f}`."
            for result in results
        )
        + "\n",
        encoding="utf-8",
    )
    print(decision, flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("tasks", nargs="*")
    parser.add_argument("--max-steps", type=int, default=10_000)
    args = parser.parse_args()
    tasks = args.tasks or tuple(TASK_CONFIG)
    unknown = set(tasks) - set(TASK_CONFIG)
    if unknown:
        parser.error(f"unknown tasks: {sorted(unknown)}")
    for task in tasks:
        run_task(task, args.max_steps)
    write_decision()


if __name__ == "__main__":
    main()
