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
    generate_pointer_dataset,
    pointer_intermediate_states,
)
from adze_t.training import make_fixed_structure_batch, stochastic_train_step
from run_phase_e import BATCH, configs, initialise


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "results" / "phase_e_1"
WORK = ROOT / "results" / "runs" / "phase_e_1"
DATA_SEEDS = {"train": 920, "validation": 921, "test": 922}
TRAIN_COUNT = 65_536
EVAL_COUNT = 4_096
CALIBRATION_COUNT = 256
MC_ROOTS_FINAL = 32
CHECKPOINTS_E1 = (100, 250, 500, 1_000, 2_000, 5_000, 10_000, 20_000, 40_000, 60_000)


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


def dataset(split: str, count: int) -> tuple[jax.Array, jax.Array, jax.Array, dict[str, Any]]:
    prompt, target, depths, audit = generate_pointer_dataset(count, DATA_SEEDS[split])
    return prompt, target, depths, {"split": split, **audit}


def dataset_audit() -> None:
    payload: dict[str, Any] = {
        "git_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "torx_pin": "f1fc858ed950ecd41935d15c06d0ec7c5e0674ae",
        "task_version": POINTER_V0.name,
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
    for split, count in (("train", TRAIN_COUNT), ("validation", EVAL_COUNT), ("test", EVAL_COUNT)):
        prompt, target, depths, audit = dataset(split, count)
        checks = audit_pointer_dataset(prompt, target, depths)
        payload[split] = {
            **audit,
            "checks": checks,
            "depth_counts": jnp.bincount(depths, length=12),
        }
    train_prompt, _, _, _ = dataset("train", 256)
    valid_prompt, _, _, _ = dataset("validation", 256)
    payload["split_overlap_check"] = bool(
        not jnp.any(jnp.all(train_prompt[:, None, :] == valid_prompt[None, :, :], axis=-1))
    )
    write(EVIDENCE / "pointer" / "dataset_audit.json", payload)


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


def train(
    name: str,
    *,
    init_seed: int,
    stochastic_training_seed: int,
    max_steps: int,
) -> dict[str, Any]:
    config = configs()[name]
    state_path = WORK / f"pointer/{name}/init{init_seed}_stoch{stochastic_training_seed}.pkl"
    curve_path = EVIDENCE / "pointer" / f"training_{'ref' if name == 'E_REF' else 'q1'}.jsonl"
    train_prompt, train_target, _, _ = dataset("train", TRAIN_COUNT)
    validation_count = CALIBRATION_COUNT if max_steps <= 500 else EVAL_COUNT
    valid_prompt, valid_target, valid_depths, _ = dataset("validation", validation_count)
    if state_path.exists():
        state = load(state_path)
        params, moments, start = state["params"], state["moments"], int(state["step"])
    else:
        params = initialise(config, init_seed)
        zero = adamw_init(params)
        moments, start = (zero, zero), 0
        curve_path.unlink(missing_ok=True)
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
    write(EVIDENCE / "pointer" / f"training_{name.lower()}_final.json", final)
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
    write(EVIDENCE / "pointer" / f"depth_metrics_{output['config'].lower()}.json", output)

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
    write(EVIDENCE / "pointer" / "localization.json", localization)
    intervention_result: dict[str, Any] = {}
    for lambda_op, label in ((0.0, "lambda0"), (1.0, "lambda1")):
        intervention_result[label] = {}
        interventions: tuple[tuple[str, dict[str, Any]], ...] = (
            ("normal", {}),
            ("suppress_q1_to_q2", {"suppress_cycle": 1}),
            ("suppress_q2_to_q3", {"suppress_cycle": 2}),
            ("shuffle_q1_to_q2", {"shuffle_cycle": 1}),
            ("stop_gradient_identity_q1_to_q2", {"suppress_cycle": 1}),
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
    write(EVIDENCE / "pointer" / "interventions.json", intervention_result)
    write(
        EVIDENCE / "pointer" / "frozen_probes.json",
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
        features["x_pre"].append(out["packed_carrier"].mean(axis=(1, 2)))
        for index in range(cycles):
            features[f"x_q{index + 1}"].append(out["dit_aux"]["trajectory"][index].mean(axis=2))
        features["h_hat"].append(out["prediction"][0].mean(axis=1))
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
    parser.add_argument("--stage", choices=("audit", "calibration", "primary"), default="audit")
    parser.add_argument("--max-steps", type=int, default=20_000)
    parser.add_argument("--configs", nargs="*", default=["E_REF", "E_Q1"])
    parser.add_argument("--init-seed", type=int, default=0)
    parser.add_argument("--stochastic-training-seed", type=int, default=0)
    args = parser.parse_args()
    if args.stage == "audit":
        dataset_audit()
        return
    steps = 500 if args.stage == "calibration" else args.max_steps
    unknown = set(args.configs) - {"E_REF", "E_Q1"}
    if unknown:
        parser.error(f"unknown configurations: {sorted(unknown)}")
    for name in args.configs:
        result = train(
            name,
            init_seed=args.init_seed,
            stochastic_training_seed=args.stochastic_training_seed,
            max_steps=steps,
        )
        if args.stage == "primary":
            final_evaluation(
                result,
                init_seed=args.init_seed,
                stochastic_training_seed=args.stochastic_training_seed,
            )


if __name__ == "__main__":
    main()
