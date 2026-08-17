"""Phase E.1 controlled computational-depth experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import subprocess
import time
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp

from adze_t.evaluation import (
    _loss_from_parts,
    _paired_example_statistics,
    phase_d_root,
)
from adze_t.backends.torx import TorxOperatorConfig, TorxOps, stable_occurrence_id
from adze_t.model import apply_model
from adze_t.objectives import adamw_init
from adze_t.phase_e_1_pointer import (
    POINTER_V0,
    audit_pointer_dataset,
    balanced_depth_indices,
    generate_pointer_dataset,
    pointer_example_hashes,
    pointer_intermediate_states,
)
from adze_t.phase_e_1_paths import checkpoint_path, evidence_path, resolve_run_state
from adze_t.training import make_fixed_structure_batch, stochastic_train_step
from run_phase_e import BATCH, configs, initialise


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "results" / "phase_e_1"
WORK = ROOT / "results" / "runs" / "phase_e_1"
DATA_SEEDS = {"train": 920, "validation": 921, "test": 922}
TRAIN_COUNT = 65_536
EVAL_COUNT = 4_096
CALIBRATION_TRAIN_COUNT = 16_384
CALIBRATION_PER_DEPTH = 64
CALIBRATION_COUNT = POINTER_V0.max_depth * CALIBRATION_PER_DEPTH
MC_ROOTS_FINAL = 32
CHECKPOINTS_E1 = (100, 250, 500, 1_000, 2_000, 5_000, 10_000, 20_000, 40_000, 60_000)
CHECKPOINTS_E1A = (100, 250, 500, 1_000, 2_000, 5_000)
OVERFIT_CHECKPOINTS = (1, 10, 25, 50, 100, 250, 500, 1_000, 2_000)
OVERFIT_CAPS = {"one": 500, "few": 1_000, "small": 2_000}


def serialise(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    value = jax.device_get(value)
    return float(value) if getattr(value, "ndim", 0) == 0 else value.tolist()


def write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=serialise) + "\n")


def append(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, sort_keys=True, default=serialise) + "\n")


def load(path: Path) -> Any:
    with path.open("rb") as stream:
        return jax.tree.map(jnp.asarray, pickle.load(stream))  # noqa: S301


def save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as stream:
        pickle.dump(jax.device_get(value), stream, protocol=pickle.HIGHEST_PROTOCOL)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def initialize_or_resume(state_path: Path, config: Any, init_seed: int) -> tuple[Any, Any, int]:
    """Resume only an exact stage/arm/seed state, otherwise initialize from scratch."""

    def load_existing(path: Path) -> tuple[Any, Any, int]:
        state = load(path)
        return state["params"], state["moments"], int(state["step"])

    def initialize_scratch() -> tuple[Any, Any, int]:
        params = initialise(config, init_seed)
        zero = adamw_init(params)
        return params, (zero, zero), 0

    return resolve_run_state(
        state_path, load_state=load_existing, initialize_state=initialize_scratch
    )


def dataset(split: str, count: int) -> tuple[jax.Array, jax.Array, jax.Array, dict[str, Any]]:
    prompt, target, depths, audit = generate_pointer_dataset(count, DATA_SEEDS[split])
    return prompt, target, depths, {"split": split, **audit}


def calibration_validation() -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Return the frozen balanced 64-example-per-depth validation subset."""
    prompt, target, depths, _ = dataset("validation", EVAL_COUNT)
    ids = balanced_depth_indices(depths, CALIBRATION_PER_DEPTH)
    return prompt[ids], target[ids], depths[ids], ids.astype(jnp.uint32)


def dataset_audit() -> None:
    payload: dict[str, Any] = {
        "git_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "torx_pin": "f1fc858ed950ecd41935d15c06d0ec7c5e0674ae",
        "task_version": POINTER_V0.name,
        "generator_version": "phase_e_1_pointer.py:POINTER_V0",
        "overlap_check_scope": "full prompt+target+depth hash sets",
        "spec": {
            "n_states": POINTER_V0.n_states,
            "max_depth": POINTER_V0.max_depth,
            "queries": POINTER_V0.queries,
            "prompt_bytes": POINTER_V0.prompt_bytes,
            "target_bytes": POINTER_V0.target_bytes,
            "chance_byte_accuracy": 0.1,
            "chance_exact_sequence_accuracy": 0.1**8,
        },
        "seeds": DATA_SEEDS,
    }
    split_hashes: dict[str, set[str]] = {}
    for split, count in (("train", TRAIN_COUNT), ("validation", EVAL_COUNT), ("test", EVAL_COUNT)):
        prompt, target, depths, audit = dataset(split, count)
        checks = audit_pointer_dataset(prompt, target, depths)
        hashes = pointer_example_hashes(prompt, target, depths)
        split_hashes[split] = hashes
        payload[split] = {
            **audit,
            "checks": checks,
            "depth_counts": jnp.bincount(depths, length=12),
            "example_hash_count": len(hashes),
            "duplicate_count": count - len(hashes),
        }
    intersections = {
        "train_validation": len(split_hashes["train"] & split_hashes["validation"]),
        "train_test": len(split_hashes["train"] & split_hashes["test"]),
        "validation_test": len(split_hashes["validation"] & split_hashes["test"]),
    }
    payload["split_intersection_counts"] = intersections
    payload["split_overlap_check"] = all(count == 0 for count in intersections.values())
    write(EVIDENCE / "pointer" / "dataset_audit_v2.json", payload)


def _lambda_zero_example_stats(
    params: Any,
    prompt: jax.Array,
    target: jax.Array,
    ids: jax.Array,
    root: jax.Array,
    config: Any,
    cycles: int,
) -> dict[str, jax.Array]:
    """Single-pass deterministic metrics for cheap calibration."""

    def one(sample_prompt: jax.Array, sample_target: jax.Array, example_id: jax.Array):
        ops = TorxOps.create(
            root,
            config=TorxOperatorConfig(operator_stochasticity=True, lambda_op=0.0),
            global_example_id=example_id,
        )
        output = apply_model(
            params,
            sample_prompt[None, :],
            jnp.ones_like(sample_prompt[None, :], bool),
            sample_target[None, :],
            jnp.ones_like(sample_target[None, :], bool),
            config=config,
            ops=ops,
            target_ops=ops,
            dit_cycles=cycles,
        )
        teacher = output["target"]["teacher"]
        mask = teacher.slot_mask
        predicted = jnp.argmax(output["byte_logits"], axis=-1)
        correct = (predicted == teacher.slot_bytes) & mask
        log_probs = jax.nn.log_softmax(output["byte_logits"], axis=-1)
        selected = jnp.take_along_axis(log_probs, teacher.slot_bytes[..., None], axis=-1)[..., 0]
        count = jnp.maximum(jnp.sum(mask), 1)
        return {
            "byte_accuracy": jnp.sum(correct) / count,
            "exact_accuracy": jnp.all(correct | ~mask),
            "byte_nll": -jnp.sum(jnp.where(mask, selected, 0.0)) / count,
            "logit_nonfinite_rate": jnp.mean(~jnp.isfinite(output["byte_logits"])),
        }

    return jax.vmap(one)(prompt, target, ids)


def calibration_depth_evaluation(
    params: Any,
    config: Any,
    prompt: jax.Array,
    target: jax.Array,
    depths: jax.Array,
    ids: jax.Array,
    *,
    cycles: tuple[int, ...],
) -> dict[str, Any]:
    """Evaluate only the predeclared lambda-zero calibration curves."""
    call = jax.jit(_lambda_zero_example_stats, static_argnames=("config", "cycles"))
    root = phase_d_root(9410, 0)
    values: dict[int, dict[str, jax.Array]] = {}
    for q in cycles:
        chunks = [
            call(
                params,
                prompt[start : start + BATCH],
                target[start : start + BATCH],
                ids[start : start + BATCH],
                root,
                config=config,
                cycles=q,
            )
            for start in range(0, len(prompt), BATCH)
        ]
        values[q] = {key: jnp.concatenate([chunk[key] for chunk in chunks]) for key in chunks[0]}
    rows = []
    for depth in range(1, POINTER_V0.max_depth + 1):
        mask = depths == depth
        row: dict[str, Any] = {"depth": depth, "count": int(jnp.sum(mask))}
        for q in cycles:
            for metric in (
                "byte_accuracy",
                "exact_accuracy",
                "byte_nll",
                "logit_nonfinite_rate",
            ):
                row[f"q{q}_{metric}"] = jnp.mean(values[q][metric][mask])
        rows.append(row)
    return {
        "lambda_op": 0.0,
        "examples": len(prompt),
        "per_depth_count": CALIBRATION_PER_DEPTH,
        "cycles": list(cycles),
        "per_depth": rows,
        "overall": {
            f"q{q}_{metric}": jnp.mean(values[q][metric])
            for q in cycles
            for metric in (
                "byte_accuracy",
                "exact_accuracy",
                "byte_nll",
                "logit_nonfinite_rate",
            )
        },
    }


def _example_stats(
    params: Any,
    prompt: jax.Array,
    target: jax.Array,
    ids: jax.Array,
    root: jax.Array,
    lambda_op: float,
    config: Any,
    cycles: int,
) -> dict[str, jax.Array]:
    stats = jax.vmap(
        lambda p, t, example_id: _paired_example_statistics(
            params,
            p[None, :],
            t[None, :],
            root,
            example_id,
            jnp.asarray(lambda_op, jnp.float32),
            config=config,
            dit_cycles=cycles,
        )
    )(prompt, target, ids)
    losses = jax.vmap(lambda n, d: _loss_from_parts(n, d, config))(
        stats["component_numerators"], stats["component_denominators"]
    )
    return {
        "loss": losses,
        "byte_accuracy": stats["byte_correct"] / jnp.maximum(stats["byte_total"], 1),
        "exact_accuracy": stats["exact_correct"] / jnp.maximum(stats["exact_total"], 1),
    }


def paired_depth_evaluation(
    params: Any,
    config: Any,
    prompt: jax.Array,
    target: jax.Array,
    depths: jax.Array,
    ids: jax.Array,
    *,
    lambda_op: float,
    roots: int,
) -> dict[str, Any]:
    """Collect per-example Q=0..3 values with paired roots and depth buckets."""
    values: dict[int, list[dict[str, jax.Array]]] = {q: [] for q in range(4)}
    call = jax.jit(_example_stats, static_argnames=("config", "cycles"))
    for root_index in range(roots):
        root = phase_d_root(9400, root_index)
        for q in range(4):
            chunks = [
                call(
                    params,
                    prompt[start : start + BATCH],
                    target[start : start + BATCH],
                    ids[start : start + BATCH],
                    root,
                    lambda_op,
                    config=config,
                    cycles=q,
                )
                for start in range(0, len(prompt), BATCH)
            ]
            values[q].append(
                {key: jnp.concatenate([chunk[key] for chunk in chunks]) for key in chunks[0]}
            )
    means = {
        q: {key: jnp.mean(jnp.stack([row[key] for row in rows]), axis=0) for key in rows[0]}
        for q, rows in values.items()
    }
    rows: list[dict[str, Any]] = []

    def bucket_mean(values: jax.Array, mask: jax.Array) -> jax.Array:
        return jnp.sum(jnp.where(mask, values, 0.0)) / jnp.maximum(jnp.sum(mask), 1)

    for depth in range(1, POINTER_V0.max_depth + 1):
        mask = depths == depth
        row: dict[str, Any] = {"depth": depth, "count": int(jnp.sum(mask))}
        for q in range(4):
            for key in ("loss", "byte_accuracy", "exact_accuracy"):
                row[f"q{q}_{key}"] = bucket_mean(means[q][key], mask)
        row["delta_31_byte"] = row["q3_byte_accuracy"] - row["q1_byte_accuracy"]
        row["delta_31_loss"] = bucket_mean(means[1]["loss"] - means[3]["loss"], mask)
        row["delta_21_loss"] = bucket_mean(means[1]["loss"] - means[2]["loss"], mask)
        row["delta_32_loss"] = bucket_mean(means[2]["loss"] - means[3]["loss"], mask)
        rows.append(row)
    benefits = {
        "loss_q1_minus_q3": means[1]["loss"] - means[3]["loss"],
        "loss_q1_minus_q2": means[1]["loss"] - means[2]["loss"],
        "loss_q2_minus_q3": means[2]["loss"] - means[3]["loss"],
        "correct_q3_minus_q1": means[3]["byte_accuracy"] - means[1]["byte_accuracy"],
    }
    depth_float = depths.astype(jnp.float32)
    trend = {key: jnp.polyfit(depth_float, value, 1)[0] for key, value in benefits.items()}
    return {
        "lambda_op": lambda_op,
        "roots": roots,
        "per_depth": rows,
        "paired_benefit_mean": {key: jnp.mean(value) for key, value in benefits.items()},
        "paired_benefit_by_example": benefits,
        "linear_trend_slope": trend,
    }


def _recorded_steps(path: Path) -> set[int]:
    if not path.exists():
        return set()
    return {int(json.loads(line)["step"]) for line in path.read_text().splitlines() if line.strip()}


def calibrate(
    name: str,
    *,
    init_seed: int,
    stochastic_training_seed: int,
    max_steps: int,
) -> dict[str, Any]:
    """Run the runtime-bounded E.1A calibration using existing checkpoints."""
    if max_steps > 5_000:
        raise ValueError("Phase E.1A calibration may not exceed 5,000 steps")
    config = configs()[name]
    state_path = checkpoint_path(
        WORK,
        benchmark="pointer",
        stage="calibration",
        arm=name,
        init_seed=init_seed,
        stochastic_training_seed=stochastic_training_seed,
    )
    short_name = "ref" if name == "E_REF" else "q1"
    curve_path = evidence_path(
        EVIDENCE,
        benchmark="pointer",
        stage="calibration",
        stem=f"calibration_e1a_{short_name}",
        init_seed=init_seed,
        stochastic_training_seed=stochastic_training_seed,
        suffix=".jsonl",
    )
    progress_path = state_path.with_suffix(".progress.json")
    train_prompt, train_target, _, _ = dataset("train", CALIBRATION_TRAIN_COUNT)
    valid_prompt, valid_target, valid_depths, valid_ids = calibration_validation()
    params, moments, start = initialize_or_resume(state_path, config, init_seed)
    if start > max_steps:
        raise ValueError(
            f"checkpoint step {start} is beyond requested calibration step {max_steps}"
        )
    update = jax.jit(stochastic_train_step, static_argnames=("config",))
    training_root = jax.random.fold_in(
        jax.random.PRNGKey(stochastic_training_seed),
        stable_occurrence_id(f"phase_e_1:pointer:{name}"),
    )
    cycles = (0, 1, 2, 3) if name == "E_REF" else (1,)
    recorded = _recorded_steps(curve_path)
    started = time.monotonic()

    def record(step: int, train_metrics: Any | None, resumed: bool) -> None:
        nonlocal recorded
        if step in recorded:
            return
        evaluation_started = time.monotonic()
        evaluation = calibration_depth_evaluation(
            params,
            config,
            valid_prompt,
            valid_target,
            valid_depths,
            valid_ids,
            cycles=cycles,
        )
        jax.block_until_ready(evaluation["overall"][f"q{cycles[-1]}_byte_nll"])
        append(
            curve_path,
            {
                "git_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                "stage": "PHASE_E_1A_RUNTIME_EFFICIENT_POINTER_CALIBRATION",
                "task_version": POINTER_V0.name,
                "config": name,
                "init_seed": init_seed,
                "stochastic_training_seed": stochastic_training_seed,
                "step": step,
                "resumed_existing_checkpoint": resumed,
                "training_subset": {
                    "selection": "first deterministic examples from POINTER_V0 seed 920",
                    "count": CALIBRATION_TRAIN_COUNT,
                    "reason": (
                        "maximum approved calibration subset preserves the usable pre-existing "
                        "P_REF step-500 checkpoint, which consumed the first 16,000 examples"
                    ),
                },
                "validation_subset": {
                    "selection": "first 64 examples in each depth bucket from seed 921",
                    "count": CALIBRATION_COUNT,
                    "global_example_ids": valid_ids,
                },
                "train": train_metrics,
                "lambda_zero": evaluation,
                "evaluation_seconds": time.monotonic() - evaluation_started,
                "elapsed_seconds_this_run": time.monotonic() - started,
            },
        )
        save(state_path, {"params": params, "moments": moments, "step": step})
        recorded.add(step)
        print(f"pointer calibration {name} step={step}", flush=True)

    if start in CHECKPOINTS_E1A:
        record(start, None, True)
    for step in range(start + 1, max_steps + 1):
        offset = ((step - 1) * BATCH) % CALIBRATION_TRAIN_COUNT
        batch = make_fixed_structure_batch(
            train_prompt[offset : offset + BATCH],
            train_target[offset : offset + BATCH],
            config=config,
        )
        params, moments, metrics = update(
            params, moments, step, batch, training_root, config=config
        )
        jax.block_until_ready(metrics["loss"])
        if step in CHECKPOINTS_E1A:
            record(step, metrics, False)
        elif step % 100 == 0:
            save(state_path, {"params": params, "moments": moments, "step": step})
            write(
                progress_path,
                {
                    "config": name,
                    "step": step,
                    "loss": metrics["loss"],
                    "elapsed_seconds_this_run": time.monotonic() - started,
                },
            )
    result = {
        "config": name,
        "optimizer_step": max_steps,
        "checkpoint": str(state_path.relative_to(ROOT)),
        "checkpoint_sha256": sha256(state_path),
        "calibration_curve": str(curve_path.relative_to(ROOT)),
        "wall_clock_seconds_this_run": time.monotonic() - started,
    }
    write(
        evidence_path(
            EVIDENCE,
            benchmark="pointer",
            stage="calibration",
            stem=f"calibration_e1a_{short_name}_summary",
            init_seed=init_seed,
            stochastic_training_seed=stochastic_training_seed,
            suffix=".json",
        ),
        result,
    )
    return result


def _overfit_examples(case: str) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Select deterministic fixed examples without changing POINTER_V0."""
    prompt, target, depths, _ = dataset("train", CALIBRATION_TRAIN_COUNT)
    if case == "one":
        selected = balanced_depth_indices(depths, 1)[5:6]
    elif case == "few":
        first_by_depth = balanced_depth_indices(depths, 1)
        selected = first_by_depth[jnp.asarray((0, 1, 3, 5, 7, 8, 9, 10))]
    elif case == "small":
        selected = balanced_depth_indices(depths, 24)
    else:
        raise ValueError(f"unknown overfit case: {case}")
    return prompt[selected], target[selected], depths[selected], selected.astype(jnp.uint32)


def _overfit_evaluation(
    params: Any,
    config: Any,
    prompt: jax.Array,
    target: jax.Array,
    ids: jax.Array,
) -> dict[str, Any]:
    call = jax.jit(_lambda_zero_example_stats, static_argnames=("config", "cycles"))
    root = phase_d_root(9420, 0)
    chunks = [
        call(
            params,
            prompt[start : start + BATCH],
            target[start : start + BATCH],
            ids[start : start + BATCH],
            root,
            config=config,
            cycles=3,
        )
        for start in range(0, len(prompt), BATCH)
    ]
    values = {key: jnp.concatenate([chunk[key] for chunk in chunks]) for key in chunks[0]}
    return {key: jnp.mean(value) for key, value in values.items()}


def overfit_diagnostic(
    case: str,
    *,
    init_seed: int,
    stochastic_training_seed: int,
) -> dict[str, Any]:
    """Test whether faithful P_REF can memorize a fixed tiny pointer corpus."""
    config = configs()["E_REF"]
    cap = OVERFIT_CAPS[case]
    state_path = checkpoint_path(
        WORK,
        benchmark="pointer",
        stage="overfit",
        arm=case,
        init_seed=init_seed,
        stochastic_training_seed=stochastic_training_seed,
    )
    curve_path = evidence_path(
        EVIDENCE,
        benchmark="pointer",
        stage="overfit",
        stem=case,
        init_seed=init_seed,
        stochastic_training_seed=stochastic_training_seed,
        suffix=".jsonl",
    )
    prompt, target, depths, ids = _overfit_examples(case)
    params, moments, start = initialize_or_resume(state_path, config, init_seed)
    update = jax.jit(stochastic_train_step, static_argnames=("config",))
    training_root = jax.random.fold_in(
        jax.random.PRNGKey(stochastic_training_seed),
        stable_occurrence_id(f"phase_e_1:pointer:overfit:{case}"),
    )
    recorded = _recorded_steps(curve_path)
    started = time.monotonic()
    prior_rows = (
        [json.loads(line) for line in curve_path.read_text().splitlines() if line.strip()]
        if curve_path.exists()
        else []
    )
    final_evaluation: dict[str, Any] | None = (
        prior_rows[-1]["lambda_zero_q3"] if prior_rows else None
    )
    memorized = bool(
        final_evaluation is not None
        and final_evaluation["byte_accuracy"] >= 0.95
        and final_evaluation["byte_nll"] <= 0.25
    )
    final_step = start
    stop = start if memorized else cap
    for step in range(start + 1, stop + 1):
        final_step = step
        indices = jnp.arange((step - 1) * BATCH, step * BATCH) % len(prompt)
        batch = make_fixed_structure_batch(prompt[indices], target[indices], config=config)
        params, moments, metrics = update(
            params, moments, step, batch, training_root, config=config
        )
        jax.block_until_ready(metrics["loss"])
        if step in OVERFIT_CHECKPOINTS and step not in recorded:
            evaluation = _overfit_evaluation(params, config, prompt, target, ids)
            jax.block_until_ready(evaluation["byte_nll"])
            append(
                curve_path,
                {
                    "git_sha": subprocess.check_output(
                        ["git", "rev-parse", "HEAD"], text=True
                    ).strip(),
                    "stage": "PHASE_E_1A_POINTER_OVERFIT_DIAGNOSTIC",
                    "task_version": POINTER_V0.name,
                    "case": case,
                    "examples": len(prompt),
                    "depth_counts": jnp.bincount(depths, length=12),
                    "selection_global_example_ids": ids,
                    "step": step,
                    "train": metrics,
                    "lambda_zero_q3": evaluation,
                    "elapsed_seconds_this_run": time.monotonic() - started,
                },
            )
            final_evaluation = evaluation
            recorded.add(step)
            memorized = bool(evaluation["byte_accuracy"] >= 0.95 and evaluation["byte_nll"] <= 0.25)
            save(state_path, {"params": params, "moments": moments, "step": step})
            print(f"pointer overfit {case} step={step} memorized={memorized}", flush=True)
            if memorized:
                break
        elif step % 100 == 0:
            save(state_path, {"params": params, "moments": moments, "step": step})
    result = {
        "case": case,
        "examples": len(prompt),
        "cap": cap,
        "optimizer_step": final_step,
        "memorized": memorized,
        "criterion": {"byte_accuracy_at_least": 0.95, "byte_nll_at_most": 0.25},
        "final_lambda_zero_q3": final_evaluation,
        "checkpoint": str(state_path.relative_to(ROOT)),
        "wall_clock_seconds_this_run": time.monotonic() - started,
    }
    write(
        evidence_path(
            EVIDENCE,
            benchmark="pointer",
            stage="overfit",
            stem=f"{case}_summary",
            init_seed=init_seed,
            stochastic_training_seed=stochastic_training_seed,
            suffix=".json",
        ),
        result,
    )
    return result


def refresh_calibration_evaluation(
    name: str,
    *,
    init_seed: int,
    stochastic_training_seed: int,
) -> dict[str, Any]:
    """Re-evaluate a completed 5k checkpoint without resuming training."""
    config = configs()[name]
    state_path = checkpoint_path(
        WORK,
        benchmark="pointer",
        stage="calibration",
        arm=name,
        init_seed=init_seed,
        stochastic_training_seed=stochastic_training_seed,
    )
    state = load(state_path)
    if int(state["step"]) != 5_000:
        raise ValueError(f"expected completed step-5000 checkpoint, got {state['step']}")
    prompt, target, depths, ids = calibration_validation()
    cycles = (0, 1, 2, 3) if name == "E_REF" else (1,)
    started = time.monotonic()
    evaluation = calibration_depth_evaluation(
        state["params"], config, prompt, target, depths, ids, cycles=cycles
    )
    jax.block_until_ready(evaluation["overall"][f"q{cycles[-1]}_byte_nll"])
    result = {
        "git_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "task_version": POINTER_V0.name,
        "config": name,
        "optimizer_step": 5_000,
        "init_seed": init_seed,
        "stochastic_training_seed": stochastic_training_seed,
        "checkpoint_sha256": sha256(state_path),
        "evaluation_seconds": time.monotonic() - started,
        "lambda_zero": evaluation,
    }
    short_name = "ref" if name == "E_REF" else "q1"
    write(
        evidence_path(
            EVIDENCE,
            benchmark="pointer",
            stage="calibration",
            stem=f"calibration_e1a_{short_name}_final_eval",
            init_seed=init_seed,
            stochastic_training_seed=stochastic_training_seed,
            suffix=".json",
        ),
        result,
    )
    return result


def train(
    name: str,
    *,
    init_seed: int,
    stochastic_training_seed: int,
    max_steps: int,
) -> dict[str, Any]:
    config = configs()[name]
    state_path = checkpoint_path(
        WORK,
        benchmark="pointer",
        stage="primary",
        arm=name,
        init_seed=init_seed,
        stochastic_training_seed=stochastic_training_seed,
    )
    short_name = "ref" if name == "E_REF" else "q1"
    curve_path = evidence_path(
        EVIDENCE,
        benchmark="pointer",
        stage="primary",
        stem=f"training_{short_name}",
        init_seed=init_seed,
        stochastic_training_seed=stochastic_training_seed,
        suffix=".jsonl",
    )
    train_prompt, train_target, _, _ = dataset("train", TRAIN_COUNT)
    validation_count = CALIBRATION_COUNT if max_steps <= 500 else EVAL_COUNT
    valid_prompt, valid_target, valid_depths, _ = dataset("validation", validation_count)
    params, moments, start = initialize_or_resume(state_path, config, init_seed)
    update = jax.jit(stochastic_train_step, static_argnames=("config",))
    training_root = jax.random.fold_in(
        jax.random.PRNGKey(stochastic_training_seed),
        stable_occurrence_id(f"phase_e_1:pointer:{name}"),
    )
    checkpoints = set(CHECKPOINTS_E1)
    started = time.monotonic()
    for step in range(start + 1, max_steps + 1):
        offset = ((step - 1) * BATCH) % TRAIN_COUNT
        batch = make_fixed_structure_batch(
            train_prompt[offset : offset + BATCH],
            train_target[offset : offset + BATCH],
            config=config,
        )
        params, moments, metrics = update(
            params, moments, step, batch, training_root, config=config
        )
        jax.block_until_ready(metrics["loss"])
        if step in checkpoints or step == max_steps:
            summary = paired_depth_evaluation(
                params,
                config,
                valid_prompt,
                valid_target,
                valid_depths,
                jnp.arange(validation_count, dtype=jnp.uint32),
                lambda_op=1.0,
                roots=1,
            )
            append(
                curve_path,
                {
                    "git_sha": subprocess.check_output(
                        ["git", "rev-parse", "HEAD"], text=True
                    ).strip(),
                    "config": name,
                    "init_seed": init_seed,
                    "stochastic_training_seed": stochastic_training_seed,
                    "step": step,
                    "train": metrics,
                    "calibration": summary,
                },
            )
            save(state_path, {"params": params, "moments": moments, "step": step})
            print(f"pointer {name} step={step} loss={float(metrics['loss']):.5f}", flush=True)
    final = {
        "config": name,
        "init_seed": init_seed,
        "stochastic_training_seed": stochastic_training_seed,
        "optimizer_step": max_steps,
        "checkpoint": str(state_path.relative_to(ROOT)),
        "checkpoint_sha256": sha256(state_path),
        "wall_clock_seconds": time.monotonic() - started,
    }
    write(
        evidence_path(
            EVIDENCE,
            benchmark="pointer",
            stage="primary",
            stem=f"training_{short_name}_final",
            init_seed=init_seed,
            stochastic_training_seed=stochastic_training_seed,
            suffix=".json",
        ),
        final,
    )
    return {"params": params, "config": config, "metadata": final}


def final_evaluation(run: dict[str, Any], *, init_seed: int, stochastic_training_seed: int) -> None:
    """Write paired depth/Q curves and localization diagnostics for one model."""
    prompt, target, depths, _ = dataset("test", EVAL_COUNT)
    ids = jnp.arange(EVAL_COUNT, dtype=jnp.uint32)
    config = run["config"]
    output: dict[str, Any] = {
        "git_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "config": "E_REF" if config.model.cycles_Q == 3 else "E_Q1",
        "init_seed": init_seed,
        "stochastic_training_seed": stochastic_training_seed,
        "task_version": POINTER_V0.name,
    }
    for lambda_op, roots, label in ((0.0, 1, "lambda0"), (1.0, MC_ROOTS_FINAL, "lambda1")):
        output[label] = paired_depth_evaluation(
            run["params"], config, prompt, target, depths, ids, lambda_op=lambda_op, roots=roots
        )

    def final_path(stem: str) -> Path:
        return evidence_path(
            EVIDENCE,
            benchmark="pointer",
            stage="primary",
            stem=stem,
            init_seed=init_seed,
            stochastic_training_seed=stochastic_training_seed,
            suffix=".json",
        )

    write(final_path(f"depth_metrics_{output['config'].lower()}"), output)

    if config.model.cycles_Q != 3:
        return
    localization: dict[str, Any] = {"config": "E_REF", "lambda0": {}, "lambda1": {}}
    for lambda_op, label in ((0.0, "lambda0"), (1.0, "lambda1")):
        root = phase_d_root(9700, 0)
        rows: list[dict[str, Any]] = []
        for start in range(0, EVAL_COUNT, BATCH):
            end = min(start + BATCH, EVAL_COUNT)
            ops = TorxOps.create(
                root,
                config=TorxOperatorConfig(operator_stochasticity=True, lambda_op=lambda_op),
                global_example_id=ids[start:end],
            )
            clean = TorxOps.create(
                root,
                config=TorxOperatorConfig(operator_stochasticity=True, lambda_op=0.0),
                global_example_id=ids[start:end],
            )
            out = apply_model(
                run["params"],
                prompt[start:end],
                jnp.ones_like(prompt[start:end], bool),
                target[start:end],
                jnp.ones_like(target[start:end], bool),
                config=config,
                ops=ops,
                target_ops=clean,
                capture_diagnostics=True,
            )
            trajectory = out["dit_aux"]["trajectory"]
            states = [out["packed_carrier"].mean(axis=(1, 2))]
            states.extend(trajectory[index].mean(axis=1) for index in range(trajectory.shape[0]))
            teacher = out["target"]["teacher"]
            correct = (
                jnp.argmax(out["byte_logits"], axis=-1) == teacher.slot_bytes
            ) & teacher.slot_mask
            for row in range(end - start):
                rows.append(
                    {
                        "depth": depths[start + row],
                        "byte_accuracy": jnp.mean(correct[row]),
                        "exact_accuracy": jnp.all(correct[row] | ~teacher.slot_mask[row]),
                        "activation_rms": [jnp.sqrt(jnp.mean(state[row] ** 2)) for state in states],
                        "update_rms": [
                            jnp.sqrt(jnp.mean((states[index + 1][row] - states[index][row]) ** 2))
                            for index in range(len(states) - 1)
                        ],
                    }
                )
        localization[label] = rows
    write(final_path("localization"), localization)
    intervention_result: dict[str, Any] = {}
    for lambda_op, label in ((0.0, "lambda0"), (1.0, "lambda1")):
        intervention_result[label] = {}
        interventions: tuple[tuple[str, dict[str, Any]], ...] = (
            ("normal", {}),
            ("suppress_q1_to_q2", {"suppress_cycle": 1}),
            ("suppress_q2_to_q3", {"suppress_cycle": 2}),
            ("shuffle_q1_to_q2", {"shuffle_cycle": 1}),
            ("stop_gradient_identity_q1_to_q2", {"stop_gradient_after_cycle": 0}),
        )
        for name, kwargs in interventions:
            correct_by_depth: list[list[jax.Array]] = [[] for _ in range(12)]
            for start in range(0, EVAL_COUNT, BATCH):
                end = min(start + BATCH, EVAL_COUNT)
                root = phase_d_root(9750, 0)
                ops = TorxOps.create(
                    root,
                    config=TorxOperatorConfig(operator_stochasticity=True, lambda_op=lambda_op),
                    global_example_id=ids[start:end],
                )
                clean = TorxOps.create(
                    root,
                    config=TorxOperatorConfig(operator_stochasticity=True, lambda_op=0.0),
                    global_example_id=ids[start:end],
                )
                out = apply_model(
                    run["params"],
                    prompt[start:end],
                    jnp.ones_like(prompt[start:end], bool),
                    target[start:end],
                    jnp.ones_like(target[start:end], bool),
                    config=config,
                    ops=ops,
                    target_ops=clean,
                    capture_diagnostics=False,
                    **kwargs,
                )
                teacher = out["target"]["teacher"]
                correct = (
                    jnp.argmax(out["byte_logits"], axis=-1) == teacher.slot_bytes
                ) & teacher.slot_mask
                for depth in range(1, 12):
                    mask = depths[start:end] == depth
                    correct_by_depth[depth].append(jnp.mean(correct, axis=(1, 2))[mask])
            intervention_result[label][name] = [
                {
                    "depth": depth,
                    "byte_accuracy": jnp.mean(jnp.concatenate(values)) if values else jnp.nan,
                }
                for depth, values in enumerate(correct_by_depth)
                if depth > 0
            ]
    write(final_path("interventions"), intervention_result)
    write(
        final_path("frozen_probes"),
        frozen_probes(run["params"], config, init_seed, stochastic_training_seed),
    )


def _probe_representations(
    params: Any,
    config: Any,
    prompt: jax.Array,
    target: jax.Array,
    ids: jax.Array,
    cycles: int,
) -> dict[str, jax.Array]:
    features: dict[str, list[jax.Array]] = {"x_pre": [], "h_hat": []}
    for index in range(cycles):
        features[f"x_q{index + 1}"] = []
    for start in range(0, len(prompt), BATCH):
        end = min(start + BATCH, len(prompt))
        root = phase_d_root(9800, 0)
        ops = TorxOps.create(
            root,
            config=TorxOperatorConfig(operator_stochasticity=True, lambda_op=0.0),
            global_example_id=ids[start:end],
        )
        out = apply_model(
            params,
            prompt[start:end],
            jnp.ones_like(prompt[start:end], bool),
            target[start:end],
            jnp.ones_like(target[start:end], bool),
            config=config,
            ops=ops,
            target_ops=ops,
            dit_cycles=cycles,
            capture_diagnostics=True,
        )
        x_pre = out["packed_carrier"].mean(axis=(1, 2))
        if x_pre.ndim != 2 or x_pre.shape[0] != end - start:
            raise ValueError(f"x_pre probe matrix must be [batch, features], got {x_pre.shape}")
        features["x_pre"].append(x_pre)
        for index in range(cycles):
            x_q = out["dit_aux"]["trajectory"][index].mean(axis=1)
            if x_q.ndim != 2 or x_q.shape[0] != end - start:
                raise ValueError(
                    f"x_q{index + 1} probe matrix must be [batch, features], got {x_q.shape}"
                )
            features[f"x_q{index + 1}"].append(x_q)
        h_hat = out["prediction"][0].mean(axis=1)
        if h_hat.ndim != 2 or h_hat.shape[0] != end - start:
            raise ValueError(f"h_hat probe matrix must be [batch, features], got {h_hat.shape}")
        features["h_hat"].append(h_hat)
    return {key: jnp.concatenate(value) for key, value in features.items()}


def _fit_probe(
    train_features: jax.Array,
    train_labels: jax.Array,
    eval_features: jax.Array,
    eval_labels: jax.Array,
    *,
    alpha: float = 1.0e-2,
) -> dict[str, Any]:
    design = jnp.concatenate(
        (train_features, jnp.ones((train_features.shape[0], 1), train_features.dtype)), axis=1
    )
    targets = jax.nn.one_hot(train_labels, POINTER_V0.n_states).reshape(train_features.shape[0], -1)
    gram = design.T @ design + alpha * jnp.eye(design.shape[1], dtype=design.dtype)
    weights = jnp.linalg.solve(gram, design.T @ targets)
    eval_design = jnp.concatenate(
        (eval_features, jnp.ones((eval_features.shape[0], 1), eval_features.dtype)), axis=1
    )
    predicted = (eval_design @ weights).reshape(
        eval_labels.shape[0], eval_labels.shape[1], POINTER_V0.n_states
    )
    predicted = jnp.argmax(predicted, axis=-1)
    correct = predicted == eval_labels
    return {
        "byte_accuracy": jnp.mean(correct),
        "exact_sequence_accuracy": jnp.mean(jnp.all(correct, axis=1)),
    }


def frozen_probes(
    params: Any, config: Any, init_seed: int, stochastic_training_seed: int
) -> dict[str, Any]:
    """Fit held-out linear probes without feeding probe gradients into Adze."""
    prompt, target, depths, _ = dataset("validation", 1_024)
    ids = jnp.arange(len(prompt), dtype=jnp.uint32)
    split = len(prompt) // 2
    states = pointer_intermediate_states(prompt)
    labels = {
        "x1": states[:, 0],
        "x2": states[:, 1],
        "final": target,
    }
    result: dict[str, Any] = {
        "init_seed": init_seed,
        "stochastic_training_seed": stochastic_training_seed,
        "held_out_examples": len(prompt) - split,
        "probes": {},
    }
    for cycles in range(4):
        representations = _probe_representations(params, config, prompt, target, ids, cycles)
        for representation, values in representations.items():
            result["probes"][f"q{cycles}_{representation}"] = {}
            for label_name, label_values in labels.items():
                valid = depths >= (2 if label_name == "x2" else 1)
                train_mask = valid.at[split:].set(False)
                test_mask = valid.at[:split].set(False)
                fitted = _fit_probe(
                    values[train_mask],
                    label_values[train_mask],
                    values[train_mask],
                    label_values[train_mask],
                )
                held_out = _fit_probe(
                    values[train_mask],
                    label_values[train_mask],
                    values[test_mask],
                    label_values[test_mask],
                )
                result["probes"][f"q{cycles}_{representation}"][label_name] = {
                    "train": fitted,
                    "held_out": held_out,
                    "valid_examples": int(jnp.sum(valid)),
                }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        choices=("audit", "calibration", "calibration-eval", "overfit", "primary"),
        default="audit",
    )
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--configs", nargs="*", default=["E_Q1", "E_REF"])
    parser.add_argument("--init-seed", type=int, default=0)
    parser.add_argument("--stochastic-training-seed", type=int, default=0)
    parser.add_argument("--overfit-cases", nargs="*", default=["one", "few", "small"])
    args = parser.parse_args()
    if args.stage == "audit":
        dataset_audit()
        return
    if args.stage == "overfit":
        unknown_cases = set(args.overfit_cases) - set(OVERFIT_CAPS)
        if unknown_cases:
            parser.error(f"unknown overfit cases: {sorted(unknown_cases)}")
        for case in args.overfit_cases:
            overfit_diagnostic(
                case,
                init_seed=args.init_seed,
                stochastic_training_seed=args.stochastic_training_seed,
            )
        return
    if args.stage == "calibration-eval":
        for name in args.configs:
            refresh_calibration_evaluation(
                name,
                init_seed=args.init_seed,
                stochastic_training_seed=args.stochastic_training_seed,
            )
        return
    steps = args.max_steps or (5_000 if args.stage == "calibration" else 40_000)
    unknown = set(args.configs) - {"E_REF", "E_Q1"}
    if unknown:
        parser.error(f"unknown configurations: {sorted(unknown)}")
    for name in args.configs:
        if args.stage == "calibration":
            calibrate(
                name,
                init_seed=args.init_seed,
                stochastic_training_seed=args.stochastic_training_seed,
                max_steps=steps,
            )
        else:
            result = train(
                name,
                init_seed=args.init_seed,
                stochastic_training_seed=args.stochastic_training_seed,
                max_steps=steps,
            )
            final_evaluation(
                result,
                init_seed=args.init_seed,
                stochastic_training_seed=args.stochastic_training_seed,
            )


if __name__ == "__main__":
    main()
