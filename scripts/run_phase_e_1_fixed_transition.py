"""Phase E.1B fixed-transition audit and runtime-bounded calibration."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import pickle
import subprocess
import time
from typing import Any

import jax
import jax.numpy as jnp

from adze_t.backends.torx import TorxOperatorConfig, TorxOps, stable_occurrence_id
from adze_t.model import apply_model
from adze_t.objectives import adamw_init
from adze_t.phase_e_1_fixed_transition import (
    FIXED_TRANSITION_V0,
    audit_fixed_transition_dataset,
    balanced_transition_indices,
    fixed_transition_example_hashes,
    generate_fixed_transition_dataset,
    transition_quality_audit,
)
from adze_t.phase_e_1_paths import checkpoint_path, evidence_path, resolve_run_state
from adze_t.training import make_fixed_structure_batch, stochastic_train_step
from run_phase_e import BATCH, configs, initialise


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "results" / "phase_e_1"
WORK = ROOT / "results" / "runs" / "phase_e_1"
DATA_SEEDS = {"train": 930, "validation": 931, "test": 932}
QUALITY_AUDIT_SEED = 933
TRAIN_COUNT = 65_536
EVAL_COUNT = 4_096
CALIBRATION_TRAIN_COUNT = 8_192
CALIBRATION_PER_DEPTH = 64
CALIBRATION_COUNT = len(FIXED_TRANSITION_V0.depths) * CALIBRATION_PER_DEPTH
CHECKPOINTS = (100, 250, 500, 1_000, 2_000, 5_000)
ARMS = {"T_REF": "E_REF", "T_Q1": "E_Q1"}


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


def save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as stream:
        pickle.dump(jax.device_get(value), stream, protocol=pickle.HIGHEST_PROTOCOL)


def load(path: Path) -> Any:
    with path.open("rb") as stream:
        return jax.tree.map(jnp.asarray, pickle.load(stream))  # noqa: S301


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def dataset(split: str, count: int) -> tuple[jax.Array, jax.Array, jax.Array, dict[str, Any]]:
    prompt, target, depths, audit = generate_fixed_transition_dataset(count, DATA_SEEDS[split])
    return prompt, target, depths, {"split": split, **audit}


def arm_config(arm: str) -> Any:
    return configs()[ARMS[arm]]


def initialize_or_resume(path: Path, config: Any, init_seed: int) -> tuple[Any, Any, int]:
    def load_existing(checkpoint: Path) -> tuple[Any, Any, int]:
        state = load(checkpoint)
        return state["params"], state["moments"], int(state["step"])

    def initialize_scratch() -> tuple[Any, Any, int]:
        params = initialise(config, init_seed)
        zero = adamw_init(params)
        return params, (zero, zero), 0

    return resolve_run_state(path, load_state=load_existing, initialize_state=initialize_scratch)


def calibration_validation() -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    prompt, target, depths, _ = dataset("validation", EVAL_COUNT)
    ids = balanced_transition_indices(depths, CALIBRATION_PER_DEPTH)
    return prompt[ids], target[ids], depths[ids], ids.astype(jnp.uint32)


def run_dataset_audit() -> dict[str, Any]:
    payload: dict[str, Any] = {
        "git_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "torx_pin": "f1fc858ed950ecd41935d15c06d0ec7c5e0674ae",
        "task_version": FIXED_TRANSITION_V0.name,
        "rule": FIXED_TRANSITION_V0.rule_name,
        "generator_version": "phase_e_1_fixed_transition.py:FIXED_TRANSITION_V0",
        "seeds": DATA_SEEDS,
        "quality_audit_seed": QUALITY_AUDIT_SEED,
        "sizes": {"train": TRAIN_COUNT, "validation": EVAL_COUNT, "test": EVAL_COUNT},
        "overlap_check_scope": "full prompt+target+depth hash sets",
        "spec": {
            "state_bits": FIXED_TRANSITION_V0.state_bits,
            "depths": FIXED_TRANSITION_V0.depths,
            "prompt_bytes": FIXED_TRANSITION_V0.prompt_bytes,
            "target_bytes": FIXED_TRANSITION_V0.target_bytes,
            "chance_bit_accuracy": 0.5,
            "chance_byte_accuracy": 1.0 / 256.0,
            "chance_exact_state_accuracy": 2.0**-64,
        },
    }
    split_hashes: dict[str, set[str]] = {}
    for split, count in (("train", TRAIN_COUNT), ("validation", EVAL_COUNT), ("test", EVAL_COUNT)):
        prompt, target, depths, audit = dataset(split, count)
        hashes = fixed_transition_example_hashes(prompt, target, depths)
        split_hashes[split] = hashes
        payload[split] = {
            **audit,
            "checks": audit_fixed_transition_dataset(prompt, target, depths),
            "depth_counts": {
                str(depth): int(jnp.sum(depths == depth)) for depth in FIXED_TRANSITION_V0.depths
            },
            "example_hash_count": len(hashes),
            "duplicate_count": count - len(hashes),
        }
    intersections = {
        "train_validation": len(split_hashes["train"] & split_hashes["validation"]),
        "train_test": len(split_hashes["train"] & split_hashes["test"]),
        "validation_test": len(split_hashes["validation"] & split_hashes["test"]),
    }
    payload["split_intersection_counts"] = intersections
    quality_initial = jax.random.randint(
        jax.random.PRNGKey(QUALITY_AUDIT_SEED), (65_536, 8), 0, 256
    )
    payload["transition_quality"] = transition_quality_audit(quality_initial)
    payload["passed"] = bool(
        all(count == 0 for count in intersections.values())
        and all(payload[split]["duplicate_count"] == 0 for split in DATA_SEEDS)
        and all(payload[split]["checks"]["oracle_matches_target"] for split in DATA_SEEDS)
        and all(payload["transition_quality"]["quality_gate"].values())
    )
    output = EVIDENCE / "fixed_transition" / "dataset_audit.json"
    write(output, payload)
    if not payload["passed"]:
        raise RuntimeError("FIXED_STATE_TRANSITION_V0 failed its model-independent task audit")
    return payload


def _example_metrics(
    params: Any,
    prompt: jax.Array,
    target: jax.Array,
    ids: jax.Array,
    root: jax.Array,
    config: Any,
    cycles: int,
) -> dict[str, jax.Array]:
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
        predicted = jnp.argmax(output["byte_logits"], axis=-1).astype(jnp.int32)
        correct = (predicted == teacher.slot_bytes) & mask
        shifts = jnp.arange(8, dtype=jnp.int32)
        predicted_bits = (predicted[..., None] >> shifts) & 1
        target_bits = (teacher.slot_bytes[..., None] >> shifts) & 1
        bit_correct = (predicted_bits == target_bits) & mask[..., None]
        log_probs = jax.nn.log_softmax(output["byte_logits"], axis=-1)
        selected = jnp.take_along_axis(log_probs, teacher.slot_bytes[..., None], axis=-1)[..., 0]
        byte_count = jnp.maximum(jnp.sum(mask), 1)
        bit_count = jnp.maximum(8 * jnp.sum(mask), 1)
        return {
            "byte_accuracy": jnp.sum(correct) / byte_count,
            "bit_accuracy": jnp.sum(bit_correct) / bit_count,
            "exact_accuracy": jnp.all(correct | ~mask),
            "byte_nll": -jnp.sum(jnp.where(mask, selected, 0.0)) / byte_count,
            "logit_nonfinite_rate": jnp.mean(~jnp.isfinite(output["byte_logits"])),
        }

    return jax.vmap(one)(prompt, target, ids)


def calibration_evaluation(
    params: Any,
    config: Any,
    prompt: jax.Array,
    target: jax.Array,
    depths: jax.Array,
    ids: jax.Array,
    *,
    cycles: tuple[int, ...],
) -> dict[str, Any]:
    call = jax.jit(_example_metrics, static_argnames=("config", "cycles"))
    root = jax.random.PRNGKey(9440)
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
    metrics = tuple(values[cycles[0]])
    rows = []
    for depth in FIXED_TRANSITION_V0.depths:
        mask = depths == depth
        row: dict[str, Any] = {"depth": depth, "count": int(jnp.sum(mask))}
        for q in cycles:
            for metric in metrics:
                row[f"q{q}_{metric}"] = jnp.mean(values[q][metric][mask])
        rows.append(row)
    return {
        "lambda_op": 0.0,
        "examples": len(prompt),
        "cycles": cycles,
        "per_depth": rows,
        "overall": {
            f"q{q}_{metric}": jnp.mean(values[q][metric]) for q in cycles for metric in metrics
        },
    }


def _recorded_steps(path: Path) -> set[int]:
    if not path.exists():
        return set()
    return {int(json.loads(line)["step"]) for line in path.read_text().splitlines() if line.strip()}


def calibrate(
    arm: str,
    *,
    init_seed: int,
    stochastic_training_seed: int,
    max_steps: int,
) -> dict[str, Any]:
    if max_steps > 5_000:
        raise ValueError("fixed-transition calibration may not exceed 5000 steps")
    config = arm_config(arm)
    state_path = checkpoint_path(
        WORK,
        benchmark="fixed_transition",
        stage="calibration",
        arm=arm,
        init_seed=init_seed,
        stochastic_training_seed=stochastic_training_seed,
    )
    short = "ref" if arm == "T_REF" else "q1"
    curve_path = evidence_path(
        EVIDENCE,
        benchmark="fixed_transition",
        stage="calibration",
        stem=f"training_{short}",
        init_seed=init_seed,
        stochastic_training_seed=stochastic_training_seed,
        suffix=".jsonl",
    )
    summary_path = evidence_path(
        EVIDENCE,
        benchmark="fixed_transition",
        stage="calibration",
        stem=f"training_{short}_summary",
        init_seed=init_seed,
        stochastic_training_seed=stochastic_training_seed,
        suffix=".json",
    )
    progress_path = state_path.with_suffix(".progress.json")
    train_prompt, train_target, _, _ = dataset("train", CALIBRATION_TRAIN_COUNT)
    valid_prompt, valid_target, valid_depths, valid_ids = calibration_validation()
    params, moments, start = initialize_or_resume(state_path, config, init_seed)
    if start > max_steps:
        raise ValueError(f"checkpoint step {start} exceeds requested step {max_steps}")
    update = jax.jit(stochastic_train_step, static_argnames=("config",))
    training_root = jax.random.fold_in(
        jax.random.PRNGKey(stochastic_training_seed),
        stable_occurrence_id(f"phase_e_1:fixed_transition:{arm}"),
    )
    cycles = (0, 1, 2, 3) if arm == "T_REF" else (1,)
    recorded = _recorded_steps(curve_path)
    started = time.monotonic()

    def record(step: int, train_metrics: Any | None, resumed: bool) -> None:
        if step in recorded:
            return
        evaluation_started = time.monotonic()
        evaluation = calibration_evaluation(
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
                "stage": "PHASE_E_1B_FIXED_TRANSITION_CALIBRATION",
                "task_version": FIXED_TRANSITION_V0.name,
                "rule": FIXED_TRANSITION_V0.rule_name,
                "arm": arm,
                "init_seed": init_seed,
                "stochastic_training_seed": stochastic_training_seed,
                "step": step,
                "resumed_existing_checkpoint": resumed,
                "training_examples": CALIBRATION_TRAIN_COUNT,
                "validation_examples": CALIBRATION_COUNT,
                "train": train_metrics,
                "lambda_zero": evaluation,
                "evaluation_seconds": time.monotonic() - evaluation_started,
                "elapsed_seconds_this_run": time.monotonic() - started,
            },
        )
        save(state_path, {"params": params, "moments": moments, "step": step})
        recorded.add(step)
        print(f"fixed-transition calibration {arm} step={step}", flush=True)

    if start in CHECKPOINTS:
        record(start, None, True)
    for step in range(start + 1, max_steps + 1):
        indices = jnp.arange((step - 1) * BATCH, step * BATCH) % CALIBRATION_TRAIN_COUNT
        batch = make_fixed_structure_batch(
            train_prompt[indices], train_target[indices], config=config
        )
        params, moments, metrics = update(
            params, moments, step, batch, training_root, config=config
        )
        jax.block_until_ready(metrics["loss"])
        if step in CHECKPOINTS:
            record(step, metrics, False)
        elif step % 100 == 0:
            save(state_path, {"params": params, "moments": moments, "step": step})
            write(
                progress_path,
                {
                    "arm": arm,
                    "step": step,
                    "loss": metrics["loss"],
                    "elapsed_seconds_this_run": time.monotonic() - started,
                },
            )
    result = {
        "arm": arm,
        "optimizer_step": max_steps,
        "init_seed": init_seed,
        "stochastic_training_seed": stochastic_training_seed,
        "checkpoint": str(state_path.relative_to(ROOT)),
        "checkpoint_sha256": sha256(state_path),
        "curve": str(curve_path.relative_to(ROOT)),
        "wall_clock_seconds_this_run": time.monotonic() - started,
    }
    write(summary_path, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("audit", "calibration"), default="audit")
    parser.add_argument("--arms", nargs="*", default=["T_Q1", "T_REF"])
    parser.add_argument("--max-steps", type=int, default=5_000)
    parser.add_argument("--init-seed", type=int, default=0)
    parser.add_argument("--stochastic-training-seed", type=int, default=0)
    args = parser.parse_args()
    if args.stage == "audit":
        run_dataset_audit()
        return
    unknown = set(args.arms) - set(ARMS)
    if unknown:
        parser.error(f"unknown arms: {sorted(unknown)}")
    for arm in args.arms:
        calibrate(
            arm,
            init_seed=args.init_seed,
            stochastic_training_seed=args.stochastic_training_seed,
            max_steps=args.max_steps,
        )


if __name__ == "__main__":
    main()
