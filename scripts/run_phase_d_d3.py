"""Run resumable Phase-D stochastic scratch training with B3-matched initialization."""

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
from adze_t.backends.torx import PHASE_D_INITIAL_SIGMA, sigma_from_rho, stable_occurrence_id
from adze_t.config import REFERENCE_SMALL_V0
from adze_t.evaluation import (
    aggregate_root_chunks,
    paired_chunk_statistics,
    phase_d_root,
    phase_d_stage_names,
    student_t_summary,
)
from adze_t.model import init_model_params
from adze_t.objectives import adamw_init
from adze_t.training import (
    B3_INITIALIZATION_SEED,
    accepted_b3_scratch_initialization,
    codec_update_mask,
    make_fixed_structure_batch,
    stochastic_train_step,
)

from run_phase_b import dataset


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "results" / "phase_d" / "d3"
WORK = ROOT / "results" / "runs" / "phase_d" / "d3"
CODEC = ROOT / "results" / "phase_b" / "checkpoints" / "target_codec_b1.pkl"
CHECKPOINTS = (100, 250, 500, 1_000, 2_000, 5_000, 10_000, 20_000, 40_000, 60_000)
TASK_CONFIG = {
    "copy": {"train_seed": 820, "valid_seed": 821, "task_identity": 0, "eval_base": 7100},
    "reverse": {"train_seed": 830, "valid_seed": 831, "task_identity": 1, "eval_base": 7200},
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


def _load(path: Path) -> Any:
    with path.open("rb") as stream:
        value = pickle.load(stream)  # noqa: S301 - trusted committed/local artifact
    return jax.tree.map(jnp.asarray, value)


def _save_work(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as stream:
        pickle.dump(jax.device_get(state), stream, protocol=pickle.HIGHEST_PROTOCOL)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _tree_sha256(tree: Any) -> str:
    digest = hashlib.sha256()
    for path, value in jax.tree_util.tree_leaves_with_path(tree):
        array = jax.device_get(value)
        digest.update(jax.tree_util.keystr(path).encode("utf-8"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(str(array.shape).encode("ascii"))
        digest.update(array.tobytes())
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


def _root_record(result: dict[str, Any], root_index: int, root: jax.Array) -> dict[str, Any]:
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
                phase_d_stage_names(),
                result["signal_rms"],
                result["perturbation_rms"],
                strict=True,
            )
        },
    }


def evaluate_roots(params, prompt, target, base_seed, count, lambda_op, evaluate):
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
                    jnp.asarray(lambda_op, dtype=jnp.float32),
                    config=REFERENCE_SMALL_V0,
                )
            )
            weights.append(min(BATCH_SIZE, prompt.shape[0] - start))
        result = aggregate_root_chunks(chunks, weights)
        jax.block_until_ready(result["loss"])
        records.append(_root_record(result, root_index, root))
    return records


def summarize(records):
    return {
        name: student_t_summary([float(record[name]) for record in records])
        for name in ("loss", "byte_accuracy", "exact_sequence_accuracy", "nonfinite_rate")
    }


def _initialization() -> tuple[Any, Any, dict[str, Any]]:
    codec = _load(CODEC)
    fresh = init_model_params(jax.random.PRNGKey(B3_INITIALIZATION_SEED), REFERENCE_SMALL_V0)
    deterministic = accepted_b3_scratch_initialization(codec, REFERENCE_SMALL_V0)
    codec_mask = codec_update_mask(fresh)
    codec_exact = all(
        jnp.array_equal(actual, codec_value)
        for actual, codec_value, enabled in zip(
            jax.tree_util.tree_leaves(deterministic),
            jax.tree_util.tree_leaves(codec),
            jax.tree_util.tree_leaves(codec_mask),
            strict=True,
        )
        if bool(enabled)
    )
    noncodec_fresh = all(
        jnp.array_equal(actual, fresh_value)
        for actual, fresh_value, enabled in zip(
            jax.tree_util.tree_leaves(deterministic),
            jax.tree_util.tree_leaves(fresh),
            jax.tree_util.tree_leaves(codec_mask),
            strict=True,
        )
        if not bool(enabled)
    )
    params, _ = deterministic_to_torx(deterministic)
    rho_values = _path_values(params, lambda path: "['rho']" in path)
    rho_policy_exact = all(
        bool(jnp.allclose(sigma_from_rho(value), PHASE_D_INITIAL_SIGMA, rtol=1.0e-6))
        for value in rho_values.values()
    )
    checks = {
        "accepted_codec_leaves_exact": codec_exact,
        "noncodec_generative_leaves_fresh_seed_700": noncodec_fresh,
        "torx_mean_round_trip_exact": _equal_paths(
            torx_means_to_deterministic(params), deterministic, lambda path: True
        ),
        "rho_initialized_to_fixed_sigma_1e_3": rho_policy_exact,
        "no_task_trained_checkpoint_loaded": True,
    }
    metadata = {
        "protocol": "accepted_B3_seed_700_fresh_noncodec_plus_target_codec_b1_masked_overlay",
        "initialization_seed": B3_INITIALIZATION_SEED,
        "codec_checkpoint": str(CODEC.relative_to(ROOT)),
        "codec_checkpoint_sha256": _sha256(CODEC),
        "deterministic_initial_tree_sha256": _tree_sha256(deterministic),
        "fresh_unoverlaid_tree_sha256": _tree_sha256(fresh),
        "checks": checks,
    }
    return params, deterministic, metadata


def run_task(task: str, training_seed: int, max_steps: int) -> dict[str, Any]:
    task_config = TASK_CONFIG[task]
    work_path = WORK / f"{task}_seed{training_seed}.pkl"
    curve_path = EVIDENCE / f"{task}_seed{training_seed}.jsonl"
    params_initial, deterministic_initial, initialization = _initialization()
    if work_path.exists():
        state = _load(work_path)
        params, moments, start_step = state["params"], state["moments"], int(state["step"])
    else:
        params = params_initial
        zero = adamw_init(params)
        moments = (zero, zero)
        start_step = 0
        curve_path.unlink(missing_ok=True)
    initial_moments = (adamw_init(params_initial), adamw_init(params_initial))
    train_prompt, train_target = dataset(task, 65_536, task_config["train_seed"])
    valid_prompt, valid_target = dataset(task, 256, task_config["valid_seed"])
    majority = float(
        jnp.max(jnp.bincount(train_target.reshape(-1), length=256)) / train_target.size
    )
    update = jax.jit(stochastic_train_step, static_argnames=("config",))
    evaluate = jax.jit(paired_chunk_statistics, static_argnames=("config",))
    training_root = jax.random.fold_in(
        jax.random.PRNGKey(training_seed),
        stable_occurrence_id(f"phase_d_d3:{task}"),
    )
    eval_base = task_config["eval_base"] + training_seed
    started = time.monotonic()
    last_metrics = None
    final_zero_records = None
    final_one_records = None
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
        zero_records = evaluate_roots(
            params, valid_prompt, valid_target, eval_base, 4, 0.0, evaluate
        )
        one_records = evaluate_roots(
            params, valid_prompt, valid_target, eval_base, 4, 1.0, evaluate
        )
        zero_summary = summarize(zero_records)
        one_summary = summarize(one_records)
        candidate = (
            float(zero_summary["byte_accuracy"]["mean"]) >= 0.9
            and float(one_summary["byte_accuracy"]["mean"]) >= 0.9
            and float(zero_summary["byte_accuracy"]["mean"]) - majority >= 0.2
        )
        if candidate:
            one_records = evaluate_roots(
                params, valid_prompt, valid_target, eval_base, 32, 1.0, evaluate
            )
            one_summary = summarize(one_records)
        elapsed = time.monotonic() - started
        row = {
            "task": task,
            "training_seed": training_seed,
            "step": step,
            "lambda_zero": zero_summary,
            "lambda_one": one_summary,
            "lambda_zero_root_count": len(zero_records),
            "lambda_one_root_count": len(one_records),
            "majority_byte_baseline": majority,
            "train": metrics,
            "wall_clock_seconds": elapsed,
        }
        _append(curve_path, row)
        _save_work(work_path, {"params": params, "moments": moments, "step": step})
        print(
            f"{task} seed={training_seed} step={step} "
            f"lambda0={float(zero_summary['byte_accuracy']['mean']):.6f} "
            f"lambda1={float(one_summary['byte_accuracy']['mean']):.6f} "
            f"roots={len(one_records)} elapsed={elapsed:.1f}s",
            flush=True,
        )
        final_zero_records = zero_records
        final_one_records = one_records
        final_step = step
        if (
            len(one_records) == 32
            and float(zero_summary["byte_accuracy"]["mean"]) >= 0.9
            and float(one_summary["byte_accuracy"]["mean"]) >= 0.9
            and float(zero_summary["byte_accuracy"]["mean"]) - majority >= 0.2
            and float(one_summary["byte_accuracy"]["mean"]) - majority >= 0.2
            and float(zero_summary["nonfinite_rate"]["mean"]) == 0.0
            and float(one_summary["nonfinite_rate"]["mean"]) == 0.0
        ):
            break
    if last_metrics is None or final_zero_records is None or final_one_records is None:
        raise RuntimeError("D3 did not reach an evaluation checkpoint")

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

    checks = {
        **initialization["checks"],
        "rho_parameters_bitwise_fixed": _equal_paths(params_initial, params, rho_paths),
        "rho_first_moments_bitwise_fixed": _equal_paths(initial_moments[0], moments[0], rho_paths),
        "rho_second_moments_bitwise_fixed": _equal_paths(initial_moments[1], moments[1], rho_paths),
        "clean_teacher_parameters_bitwise_fixed": _equal_paths(
            params_initial, params, teacher_paths
        ),
        "raw_rho_connectivity_nonzero": float(last_metrics["grad_rho_raw_norm"]) > 0,
        "applied_rho_gradient_zero": float(last_metrics["grad_rho_applied_norm"]) == 0,
        "finite_training_metrics": all(
            bool(jnp.all(jnp.isfinite(value))) for value in jax.tree_util.tree_leaves(last_metrics)
        ),
    }
    zero_summary = summarize(final_zero_records)
    one_summary = summarize(final_one_records)
    passed = (
        all(checks.values())
        and len(final_one_records) == 32
        and float(zero_summary["byte_accuracy"]["mean"]) >= 0.9
        and float(one_summary["byte_accuracy"]["mean"]) >= 0.9
        and float(zero_summary["byte_accuracy"]["mean"]) - majority >= 0.2
        and float(one_summary["byte_accuracy"]["mean"]) - majority >= 0.2
        and float(zero_summary["nonfinite_rate"]["mean"]) == 0.0
        and float(one_summary["nonfinite_rate"]["mean"]) == 0.0
    )
    _write(EVIDENCE / f"{task}_seed{training_seed}_lambda0_final.json", final_zero_records)
    _write(EVIDENCE / f"{task}_seed{training_seed}_lambda1_final_32_roots.json", final_one_records)
    result = {
        "task": task,
        "training_seed": training_seed,
        "step": final_step,
        "training_root_key": jax.random.key_data(training_root),
        "evaluation_root_base": eval_base,
        "train_seed": task_config["train_seed"],
        "validation_seed": task_config["valid_seed"],
        "train_examples": 65_536,
        "validation_examples": 256,
        "batch_size": BATCH_SIZE,
        "majority_byte_baseline": majority,
        "lambda_zero": zero_summary,
        "lambda_one": one_summary,
        "initialization": initialization,
        "deterministic_initial_tree_sha256": _tree_sha256(deterministic_initial),
        "working_checkpoint": str(work_path.relative_to(ROOT)),
        "working_checkpoint_sha256": _sha256(work_path),
        "wall_clock_seconds": sum(
            json.loads(line)["wall_clock_seconds"]
            for line in curve_path.read_text(encoding="utf-8").splitlines()[-1:]
        ),
        "checks": checks,
        "passed": passed,
    }
    _write(EVIDENCE / f"{task}_seed{training_seed}_summary.json", result)
    return result


def write_decision() -> None:
    primary_paths = [EVIDENCE / f"{task}_seed0_summary.json" for task in TASK_CONFIG]
    if not all(path.exists() for path in primary_paths):
        return
    primary = [json.loads(path.read_text(encoding="utf-8")) for path in primary_paths]
    repeats = []
    for seed in (1, 2):
        seed_paths = [EVIDENCE / f"{task}_seed{seed}_summary.json" for task in TASK_CONFIG]
        if all(path.exists() for path in seed_paths):
            repeats.extend(json.loads(path.read_text(encoding="utf-8")) for path in seed_paths)
    if all(result["passed"] for result in primary):
        decision = "D3_STOCHASTIC_SCRATCH_PASS"
    elif any(
        result["lambda_zero"]["nonfinite_rate"]["mean"] > 0
        or result["lambda_one"]["nonfinite_rate"]["mean"] > 0
        for result in primary
    ):
        decision = "D3_STOCHASTIC_SCRATCH_NUMERICAL_FAILURE"
    else:
        decision = "D3_STOCHASTIC_SCRATCH_UNRESOLVED"
    lines = [
        "# D3 — scratch generative trainability",
        "",
        f"Decision: **{decision}**.",
        "",
    ]
    for result in (*primary, *repeats):
        lines.append(
            f"- {result['task'].upper()} seed {result['training_seed']}: step {result['step']}, "
            f"lambda-zero `{result['lambda_zero']['byte_accuracy']['mean']:.6f}`, "
            f"lambda-one `{result['lambda_one']['byte_accuracy']['mean']:.6f}`."
        )
    (EVIDENCE / "D3_STOCHASTIC_SCRATCH.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(decision, flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("tasks", nargs="*")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=60_000)
    args = parser.parse_args()
    tasks = args.tasks or tuple(TASK_CONFIG)
    unknown = set(tasks) - set(TASK_CONFIG)
    if unknown:
        parser.error(f"unknown tasks: {sorted(unknown)}")
    for task in tasks:
        run_task(task, args.seed, args.max_steps)
    write_decision()


if __name__ == "__main__":
    main()
