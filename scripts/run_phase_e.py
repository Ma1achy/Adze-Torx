"""Phase E faithful Q-recurrence controls and causal diagnostics.

The script is intentionally resumable.  Evidence is versioned under
``results/phase_e`` while potentially large working state remains ignored.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import gc
import hashlib
import json
from pathlib import Path
import pickle
import time
from typing import Any

import jax
import jax.numpy as jnp

from adze_t.backends.mapping import deterministic_to_torx
from adze_t.backends.torx import TorxOperatorConfig, TorxOps, stable_occurrence_id
from adze_t.config import REFERENCE_SMALL_V0, ReferenceConfig
from adze_t.evaluation import (
    aggregate_root_chunks,
    paired_chunk_statistics,
    phase_d_root,
    student_t_summary,
)
from adze_t.model import apply_model, init_model_params
from adze_t.objectives import adamw_init
from adze_t.training import _CODEC_ENCODER_NAMES, make_fixed_structure_batch, stochastic_train_step
from run_phase_b import dataset


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "results" / "phase_e"
WORK = ROOT / "results" / "runs" / "phase_e"
CODEC = ROOT / "results" / "phase_b" / "checkpoints" / "target_codec_b1.pkl"
CHECKPOINTS = (100, 250, 500, 1_000, 2_000, 5_000, 10_000, 15_000, 20_000)
BATCH = 32
TASKS = {"copy": (820, 821, 8100), "reverse": (830, 831, 8200)}


def configs() -> dict[str, ReferenceConfig]:
    reference = REFERENCE_SMALL_V0
    return {
        "E_REF": reference,
        "E_Q1": replace(reference, model=replace(reference.model, cycles_Q=1)),
        "E_UNSHARED12": replace(
            reference, model=replace(reference.model, physical_blocks_L=12, cycles_Q=1)
        ),
        "E_REF_NODEPTHCOND": replace(
            reference, model=replace(reference.model, effective_depth_conditioning=False)
        ),
    }


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


def count_params(tree: Any) -> int:
    return sum(leaf.size for leaf in jax.tree_util.tree_leaves(tree))


def initialise(config: ReferenceConfig, init_seed: int) -> Any:
    """Overlay only the accepted frozen codec on deterministic scratch means."""
    codec = load(CODEC)
    fresh = init_model_params(jax.random.PRNGKey(init_seed), config)
    encoder = dict(fresh["encoder"])
    for name in _CODEC_ENCODER_NAMES:
        encoder[name] = codec["encoder"][name]
    deterministic = {**fresh, "encoder": encoder, "decoder": codec["decoder"]}
    params, _ = deterministic_to_torx(deterministic)
    return params


def audit(init_seed: int = 0) -> dict[str, Any]:
    configured = configs()
    counts = {
        name: count_params(init_model_params(jax.random.PRNGKey(init_seed), config))
        for name, config in configured.items()
    }
    reference_count = counts["E_REF"]
    rows = []
    for name, config in configured.items():
        model = config.model
        rows.append(
            {
                "config": name,
                "physical_blocks": model.physical_blocks_L,
                "cycles_Q": model.cycles_Q,
                "effective_applications": model.physical_blocks_L * model.cycles_Q,
                "weight_sharing": model.cycles_Q > 1,
                "effective_depth_conditioning": model.effective_depth_conditioning,
                "parameter_count": counts[name],
                "parameter_mismatch_fraction": abs(counts[name] - reference_count)
                / reference_count,
            }
        )
    ref_q1_equal = counts["E_REF"] == counts["E_Q1"]
    unshared_distinct = configured["E_UNSHARED12"].model.physical_blocks_L == 12
    no_depth_toggle = not configured["E_REF_NODEPTHCOND"].model.effective_depth_conditioning
    # A lambda-zero smoke checks all operator routes with each static configuration.
    prompt, target = dataset("copy", BATCH, 820)
    finite = {}
    for name, config in configured.items():
        params = initialise(config, init_seed)
        output = apply_model(
            params,
            prompt,
            jnp.ones_like(prompt, bool),
            target,
            jnp.ones_like(target, bool),
            config=config,
            ops=TorxOps.create(jax.random.PRNGKey(91), config=TorxOperatorConfig(lambda_op=0.0)),
            target_ops=TorxOps.create(
                jax.random.PRNGKey(91), config=TorxOperatorConfig(lambda_op=0.0)
            ),
        )
        finite[name] = bool(all(jnp.all(jnp.isfinite(x)) for x in jax.tree.leaves(output)))
        del output, params
        gc.collect()
    result = {
        "git_sha": __import__("subprocess")
        .check_output(["git", "rev-parse", "HEAD"], text=True)
        .strip(),
        "torx_pin": "f1fc858ed950ecd41935d15c06d0ec7c5e0674ae",
        "init_seed": init_seed,
        "aliases": {"E_REF_DEPTHCOND": "E_REF", "E_PARAMMATCH_Q1": "E_Q1"},
        "rows": rows,
        "checks": {
            "ref_q1_exact_parameter_match": ref_q1_equal,
            "ref_q1_parameter_mismatch_zero": counts["E_REF"] == counts["E_Q1"],
            "unshared12_has_twelve_distinct_block_entries": unshared_distinct,
            "depth_conditioning_toggle_available": no_depth_toggle,
            "lambda0_finite": finite,
        },
    }
    result["passed"] = (
        ref_q1_equal and unshared_distinct and no_depth_toggle and all(finite.values())
    )
    write(EVIDENCE / "config_audit.json", result)
    if not result["passed"]:
        raise RuntimeError("E0 control audit failed; Phase E must stop")
    return result


def evaluate(
    params: Any,
    config: ReferenceConfig,
    prompt: jax.Array,
    target: jax.Array,
    base: int,
    lambda_op: float,
    roots: int,
    *,
    dit_cycles: int | None = None,
    depth_code_override: str = "correct",
    suppress_cycle: int | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    call = jax.jit(
        paired_chunk_statistics,
        static_argnames=("config", "dit_cycles", "depth_code_override", "suppress_cycle"),
    )
    records = []
    for index in range(roots):
        chunks = [
            call(
                params,
                prompt[start : start + BATCH],
                target[start : start + BATCH],
                phase_d_root(base, index),
                jnp.arange(start, min(start + BATCH, len(prompt)), dtype=jnp.uint32),
                jnp.asarray(lambda_op, jnp.float32),
                config=config,
                dit_cycles=dit_cycles,
                depth_code_override=depth_code_override,
                suppress_cycle=suppress_cycle,
            )
            for start in range(0, len(prompt), BATCH)
        ]
        aggregate = aggregate_root_chunks(chunks, [BATCH] * len(chunks), config=config)
        jax.block_until_ready(aggregate["loss"])
        records.append({key: serialise(value) for key, value in aggregate.items()})
    summary = {
        key: student_t_summary([record[key] for record in records])
        for key in ("loss", "byte_accuracy", "exact_sequence_accuracy", "nonfinite_rate")
    }
    return summary, records


def cycle_diagnostics(
    params: Any, config: ReferenceConfig, prompt: jax.Array, target: jax.Array
) -> dict[str, Any]:
    """Small representative diagnostic batch; primary metrics retain 256 examples/MC."""
    rows = {}
    for lambda_op in (0.0, 1.0):
        root = jax.random.PRNGKey(930 + int(lambda_op))
        noisy = TorxOps.create(root, config=TorxOperatorConfig(lambda_op=lambda_op))
        clean = TorxOps.create(root, config=TorxOperatorConfig(lambda_op=0.0))
        out = apply_model(
            params,
            prompt[:BATCH],
            jnp.ones_like(prompt[:BATCH], bool),
            target[:BATCH],
            jnp.ones_like(target[:BATCH], bool),
            config=config,
            ops=noisy,
            target_ops=clean,
            capture_diagnostics=True,
        )
        trajectory = out["dit_aux"]["trajectory"]
        initial = out["packed_carrier"].reshape(trajectory.shape[1:])
        states = [initial, *[trajectory[index] for index in range(trajectory.shape[0])]]
        updates = []
        for before, after in zip(states, states[1:]):
            delta = after - before
            updates.append(
                {
                    "update_rms": jnp.sqrt(jnp.mean(delta**2)),
                    "relative_update": jnp.sqrt(jnp.mean(delta**2))
                    / jnp.maximum(jnp.sqrt(jnp.mean(before**2)), 1e-8),
                    "cosine": jnp.sum(before * after)
                    / jnp.maximum(jnp.linalg.norm(before) * jnp.linalg.norm(after), 1e-8),
                    "activation_rms": jnp.sqrt(jnp.mean(after**2)),
                }
            )
        rows[f"lambda_{int(lambda_op)}"] = {
            "updates": updates,
            "attention_output_rms": out["dit_aux"]["cycle_attention_rms"],
            "ffn_output_rms": out["dit_aux"]["cycle_ffn_rms"],
            "attention_gate_mean": out["dit_aux"]["attention_gate_mean"],
            "ffn_gate_mean": out["dit_aux"]["ffn_gate_mean"],
        }
    rows["jvp_local_contraction"] = {"status": "deferred", "reason": "secondary diagnostic"}
    rows["linear_probes"] = {"status": "deferred", "reason": "secondary diagnostic"}
    return rows


def run(
    task: str,
    name: str,
    *,
    init_seed: int,
    stochastic_training_seed: int,
    max_steps: int,
    progress_every: int = 1_000,
) -> dict[str, Any]:
    config = configs()[name]
    run_name = f"{task}/{name}/init{init_seed}_stoch{stochastic_training_seed}"
    state_path = WORK / f"{run_name}.pkl"
    progress_path = WORK / f"{run_name}.progress.json"
    curve_path = EVIDENCE / task / f"{name}_init{init_seed}_stoch{stochastic_training_seed}.jsonl"
    params_initial = initialise(config, init_seed)
    if state_path.exists():
        state = load(state_path)
        params, moments, start = state["params"], state["moments"], int(state["step"])
    else:
        params = params_initial
        zero = adamw_init(params)
        moments, start = (zero, zero), 0
        curve_path.unlink(missing_ok=True)
    train_seed, valid_seed, root_base = TASKS[task]
    train_prompt, train_target = dataset(task, 65_536, train_seed)
    valid_prompt, valid_target = dataset(task, 256, valid_seed)
    update = jax.jit(stochastic_train_step, static_argnames=("config",))
    training_root = jax.random.fold_in(
        jax.random.PRNGKey(stochastic_training_seed), stable_occurrence_id(f"phase_e:{task}:{name}")
    )
    compile_seconds = None
    elapsed_start = time.monotonic()
    steady = []
    final = None
    for step in range(start + 1, max_steps + 1):
        offset = ((step - 1) * BATCH) % len(train_prompt)
        batch = make_fixed_structure_batch(
            train_prompt[offset : offset + BATCH],
            train_target[offset : offset + BATCH],
            config=config,
        )
        began = time.monotonic()
        params, moments, metrics = update(
            params, moments, step, batch, training_root, config=config
        )
        jax.block_until_ready(metrics["loss"])
        duration = time.monotonic() - began
        if compile_seconds is None:
            compile_seconds = duration
        else:
            steady.append(duration)
        if step % progress_every == 0 and step not in CHECKPOINTS:
            write(
                progress_path,
                {
                    "config": name,
                    "task": task,
                    "optimizer_step": step,
                    "loss": metrics["loss"],
                    "grad_norm": metrics["grad_optimizer_reported_norm"],
                    "steady_step_seconds": sum(steady[-20:]) / max(len(steady[-20:]), 1),
                },
            )
            save(state_path, {"params": params, "moments": moments, "step": step})
            print(
                f"{task} {name} progress step={step} loss={float(metrics['loss']):.5f}", flush=True
            )
        if step not in CHECKPOINTS and step != max_steps:
            continue
        zero_summary, _ = evaluate(params, config, valid_prompt, valid_target, root_base, 0.0, 1)
        one_summary, _ = evaluate(params, config, valid_prompt, valid_target, root_base, 1.0, 1)
        row = {
            "git_sha": __import__("subprocess")
            .check_output(["git", "rev-parse", "HEAD"], text=True)
            .strip(),
            "config": name,
            "task": task,
            "init_seed": init_seed,
            "stochastic_training_seed": stochastic_training_seed,
            "parameter_count": count_params(params),
            "effective_applications": config.model.physical_blocks_L * config.model.cycles_Q,
            "optimizer_step": step,
            "cumulative_block_applications": step
            * BATCH
            * config.model.physical_blocks_L
            * config.model.cycles_Q,
            "lambda": 1.0,
            "sigma": 1e-3,
            "train": metrics,
            "lambda_zero": zero_summary,
            "lambda_one": one_summary,
            "compile_seconds": compile_seconds,
            "steady_step_seconds": sum(steady[-20:]) / max(len(steady[-20:]), 1),
            "examples_per_second": BATCH / max(sum(steady[-20:]) / max(len(steady[-20:]), 1), 1e-9),
        }
        append(curve_path, row)
        save(state_path, {"params": params, "moments": moments, "step": step})
        final = row
        print(
            f"{task} {name} step={step} lambda1={one_summary['byte_accuracy']['mean']:.5f}",
            flush=True,
        )
    if final is None:
        final = {
            "config": name,
            "task": task,
            "init_seed": init_seed,
            "stochastic_training_seed": stochastic_training_seed,
            "parameter_count": count_params(params),
            "effective_applications": config.model.physical_blocks_L * config.model.cycles_Q,
            "optimizer_step": start,
            "cumulative_block_applications": start
            * BATCH
            * config.model.physical_blocks_L
            * config.model.cycles_Q,
            "lambda": 1.0,
            "sigma": 1e-3,
            "resumed_completed_checkpoint": True,
        }
    zero, zero_roots = evaluate(params, config, valid_prompt, valid_target, root_base, 0.0, 1)
    one, one_roots = evaluate(params, config, valid_prompt, valid_target, root_base, 1.0, 32)
    final.update(
        {
            "lambda_zero_final": zero,
            "lambda_one_final_32root": one,
            "lambda_zero_roots": zero_roots,
            "lambda_one_roots": one_roots,
            "wall_clock_seconds": time.monotonic() - elapsed_start,
            "checkpoint": str(state_path.relative_to(ROOT)),
            "checkpoint_sha256": sha256(state_path),
        }
    )
    write(
        EVIDENCE / task / f"{name}_init{init_seed}_stoch{stochastic_training_seed}_summary.json",
        final,
    )
    return final


def diagnostics(task: str, *, init_seed: int = 0, stochastic_training_seed: int = 0) -> None:
    path = WORK / f"{task}/E_REF/init{init_seed}_stoch{stochastic_training_seed}.pkl"
    state = load(path)
    config = configs()["E_REF"]
    _, valid_seed, base = TASKS[task]
    prompt, target = dataset(task, 256, valid_seed)
    result: dict[str, Any] = {
        "task": task,
        "checkpoint": str(path.relative_to(ROOT)),
        "checkpoint_sha256": sha256(path),
    }
    result["cycle_metrics"] = cycle_diagnostics(state["params"], config, prompt, target)
    result["truncation"] = {
        f"Q_exec_{q}": {
            f"lambda_{int(lam)}": evaluate(
                state["params"], config, prompt, target, base, lam, 32 if lam else 1, dit_cycles=q
            )[0]
            for lam in (0.0, 1.0)
        }
        for q in (1, 2, 3)
    }
    result["interventions"] = {
        "identity": evaluate(state["params"], config, prompt, target, base, 1.0, 32)[0],
        "suppress_cycle_1_delta": evaluate(
            state["params"], config, prompt, target, base, 1.0, 32, suppress_cycle=1
        )[0],
        "shuffled_state": {
            "status": "deferred",
            "reason": "requires a separate cross-example intervention path",
        },
    }
    result["depth_code"] = {
        code: evaluate(
            state["params"], config, prompt, target, base, 1.0, 32, depth_code_override=code
        )[0]
        for code in ("correct", "all_q0", "reversed")
    }
    write(EVIDENCE / "diagnostics" / f"{task}.json", result)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage", choices=("e0", "e1", "primary", "diagnostics", "all"), default="all"
    )
    parser.add_argument("--tasks", nargs="*", default=list(TASKS))
    parser.add_argument("--configs", nargs="*", default=list(configs()))
    parser.add_argument("--max-steps", type=int, default=20_000)
    parser.add_argument("--progress-every", type=int, default=1_000)
    parser.add_argument("--init-seed", type=int, default=0)
    parser.add_argument("--stochastic-training-seed", type=int, default=0)
    args = parser.parse_args()
    unknown = set(args.configs) - set(configs())
    if unknown:
        parser.error(f"unknown Phase E configurations: {sorted(unknown)}")
    if args.stage in ("e0", "all"):
        audit(args.init_seed)
    if args.stage in ("e1", "all"):
        for name in args.configs:
            run(
                "copy",
                name,
                init_seed=args.init_seed,
                stochastic_training_seed=args.stochastic_training_seed,
                max_steps=500,
                progress_every=args.progress_every,
            )
    if args.stage in ("primary", "all"):
        for task in args.tasks:
            for name in args.configs:
                run(
                    task,
                    name,
                    init_seed=args.init_seed,
                    stochastic_training_seed=args.stochastic_training_seed,
                    max_steps=args.max_steps,
                    progress_every=args.progress_every,
                )
    if args.stage in ("diagnostics", "all"):
        for task in args.tasks:
            diagnostics(
                task,
                init_seed=args.init_seed,
                stochastic_training_seed=args.stochastic_training_seed,
            )


if __name__ == "__main__":
    main()
