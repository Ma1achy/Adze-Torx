"""Regenerate Phase-D final MC evidence with stable per-example identities.

This performs evaluation only.  It never calls an optimizer or writes a working
checkpoint; original Phase-D evidence is deliberately left untouched.
"""

from __future__ import annotations

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
    all_d3_runs_pass,
    paired_chunk_statistics,
    phase_d_root,
    phase_d_stage_names,
    student_t_summary,
)

from run_phase_b import dataset


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results" / "phase_d_1"
PHASE_D = ROOT / "results" / "phase_d"
WORK = ROOT / "results" / "runs" / "phase_d"
CHECKPOINTS = ROOT / "results" / "phase_b" / "checkpoints"
D1 = {"copy": (821, 4100), "reverse": (831, 4200)}
D2 = {"copy": (821, 6100), "reverse": (831, 6200)}
D3 = {"copy": (821, 7100), "reverse": (831, 7200)}


def _json(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    value = jax.device_get(value)
    return float(value) if getattr(value, "ndim", 0) == 0 else value.tolist()


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=_json) + "\n")


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _load(path: Path) -> Any:
    with path.open("rb") as stream:
        return jax.tree.map(jnp.asarray, pickle.load(stream))  # noqa: S301


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _record(result: dict[str, Any], index: int) -> dict[str, Any]:
    return {
        "root_index": index,
        **{
            name: result[name]
            for name in ("loss", "byte_accuracy", "exact_sequence_accuracy", "nonfinite_rate")
        },
        "stages": {
            name: {"signal_rms": signal, "perturbation_rms": perturbation}
            for name, signal, perturbation in zip(
                phase_d_stage_names(), result["signal_rms"], result["perturbation_rms"], strict=True
            )
        },
    }


def _summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        name: student_t_summary([float(record[name]) for record in records])
        for name in ("loss", "byte_accuracy", "exact_sequence_accuracy", "nonfinite_rate")
    }


def evaluate(
    params: Any, prompt: jax.Array, target: jax.Array, root_base: int, roots: int, lambda_op: float
) -> list[dict[str, Any]]:
    call = jax.jit(paired_chunk_statistics, static_argnames=("config",))
    records = []
    for root_index in range(roots):
        chunks = []
        for start in range(0, prompt.shape[0], 32):
            end = min(start + 32, prompt.shape[0])
            chunks.append(
                call(
                    params,
                    prompt[start:end],
                    target[start:end],
                    phase_d_root(root_base, root_index),
                    jnp.arange(start, end, dtype=jnp.uint32),
                    jnp.asarray(lambda_op, jnp.float32),
                    config=REFERENCE_SMALL_V0,
                )
            )
        result = aggregate_root_chunks(chunks, [32] * len(chunks), config=REFERENCE_SMALL_V0)
        jax.block_until_ready(result["loss"])
        records.append(_record(result, root_index))
        print(
            f"lambda={lambda_op:g} root={root_index + 1}/{roots} "
            f"byte={float(result['byte_accuracy']):.6f}",
            flush=True,
        )
    return records


def d1() -> list[dict[str, Any]]:
    completed = []
    for task, (valid_seed, root_base) in D1.items():
        params, _ = deterministic_to_torx(_load(CHECKPOINTS / f"{task}.pkl"))
        prompt, target = dataset(task, 256, valid_seed)
        old = json.loads((PHASE_D / "d1" / f"{task}.json").read_text())
        results = []
        for lambda_op, count in ((0.0, 1), (0.1, 16), (0.25, 16), (0.5, 16), (1.0, 32)):
            records = evaluate(params, prompt, target, root_base, count, lambda_op)
            results.append(
                {"lambda_op": lambda_op, "root_count": count, **_summary(records), "roots": records}
            )
        old_by_lambda = {item["lambda_op"]: item for item in old["lambda_results"]}
        comparison = [
            {
                "lambda_op": item["lambda_op"],
                "old_byte_accuracy_mean": old_by_lambda[item["lambda_op"]]["byte_accuracy"]["mean"],
                "corrected_byte_accuracy_mean": item["byte_accuracy"]["mean"],
                "delta_byte_accuracy": item["byte_accuracy"]["mean"]
                - old_by_lambda[item["lambda_op"]]["byte_accuracy"]["mean"],
                "old_loss_mean": old_by_lambda[item["lambda_op"]]["loss"]["mean"],
                "corrected_loss_mean": item["loss"]["mean"],
            }
            for item in results
        ]
        value = {
            "task": task,
            "validation_seed": valid_seed,
            "base_root_seed": root_base,
            "paired_nested_roots": True,
            "corrected_per_example_identity": True,
            "old_phase_d": old,
            "corrected": results,
            "old_vs_corrected": comparison,
        }
        _write(OUTPUT / "d1" / f"{task}.json", value)
        completed.append(value)
    return completed


def d2() -> list[dict[str, Any]]:
    completed = []
    for task, (valid_seed, root_base) in D2.items():
        path = WORK / "d2" / f"{task}.pkl"
        if not path.exists():
            continue
        state = _load(path)
        prompt, target = dataset(task, 256, valid_seed)
        zero = evaluate(state["params"], prompt, target, root_base, 1, 0.0)
        one = evaluate(state["params"], prompt, target, root_base, 32, 1.0)
        value = {
            "task": task,
            "checkpoint": str(path.relative_to(ROOT)),
            "checkpoint_sha256": _hash(path),
            "step": int(state["step"]),
            "lambda_zero": _summary(zero),
            "lambda_one": _summary(one),
            "lambda_one_roots": one,
        }
        _write(OUTPUT / "d2" / f"{task}_final_mc.json", value)
        completed.append(value)
    return completed


def d3() -> list[dict[str, Any]]:
    completed = []
    for task, (valid_seed, root_base) in D3.items():
        prompt, target = dataset(task, 256, valid_seed)
        for seed in range(3):
            path = WORK / "d3" / f"{task}_seed{seed}.pkl"
            if not path.exists():
                continue
            state = _load(path)
            zero = evaluate(state["params"], prompt, target, root_base + seed, 1, 0.0)
            one = evaluate(state["params"], prompt, target, root_base + seed, 32, 1.0)
            value = {
                "task": task,
                "stochastic_training_seed": seed,
                "checkpoint": str(path.relative_to(ROOT)),
                "checkpoint_sha256": _hash(path),
                "step": int(state["step"]),
                "lambda_zero": _summary(zero),
                "lambda_one": _summary(one),
                "lambda_one_roots": one,
            }
            value["passed"] = bool(
                value["lambda_zero"]["byte_accuracy"]["mean"] >= 0.9
                and value["lambda_one"]["byte_accuracy"]["mean"] >= 0.9
                and value["lambda_one"]["nonfinite_rate"]["mean"] == 0
            )
            _write(OUTPUT / "d3" / f"{task}_seed{seed}_final_mc.json", value)
            completed.append(value)
    return completed


def _read_records(directory: Path) -> list[dict[str, Any]]:
    return [json.loads(path.read_text()) for path in sorted(directory.glob("*.json"))]


def finalize() -> bool:
    """Write the D.1 decision from already regenerated, versioned evidence."""
    d1_results = _read_records(OUTPUT / "d1")
    d2_results = _read_records(OUTPUT / "d2")
    d3_results = _read_records(OUTPUT / "d3")
    d0 = json.loads((OUTPUT / "d0" / "primitive_moments_gradients.json").read_text())
    passed = (
        bool(d0["passed"])
        and len(d1_results) == 2
        and len(d2_results) == 2
        and len(d3_results) == 6
        and all_d3_runs_pass(d3_results)
    )
    manifest = {
        "base_phase_d_commit": "0e06d4d",
        "evaluator_fix": "root + global_example_id + static scope/module + existing recurrence coordinates",
        "d0": d0,
        "d1": d1_results,
        "d2": d2_results,
        "d3": d3_results,
        "d3_all_authoritative_runs_pass": all_d3_runs_pass(d3_results),
        "passed": passed,
    }
    _write(OUTPUT / "reevaluation_manifest.json", manifest)
    decision = "PHASE_D_1_PASS" if passed else "PHASE_D_1_BLOCKED"
    _write_text(
        OUTPUT / "DECISION.md",
        "# Phase D.1 decision\n\n"
        f"Decision: **{decision}**.\n\n"
        "The evaluator now folds a dedicated integer global-example identity into every "
        "operator occurrence key. Original Phase-D evidence is preserved; this directory "
        "contains corrected evaluation-only evidence.\n\n"
        f"D3 all included stochastic-training seeds pass: `{all_d3_runs_pass(d3_results)}`.\n\n"
        "**TORX_STOCHASTIC_TRAINABILITY_PASS is reconfirmed under corrected MC evaluation "
        "semantics.**\n"
        if passed
        else "Corrected D.1 evidence did not satisfy all gates.\n",
    )
    lines = [
        "# Phase D.1 — corrected MC evaluation and D0 gradient hardening",
        "",
        f"Decision: **{decision}**.",
        "",
        "## Evaluation correction",
        "",
        "Every validation sample now receives a stable integer `global_example_id`, separate "
        "from static scope/module identities. Evaluation runs each logical sample as a B=1 "
        "execution under `vmap`, then aggregates objective numerators and denominators using "
        "the original loss normalization. Training was not changed: its factor calls already "
        "sample batch-shaped epsilon tensors under a fresh optimizer-step key.",
        "",
        "## Corrected final evidence",
        "",
    ]
    for result in [*d2_results, *d3_results]:
        one = result["lambda_one"]["byte_accuracy"]
        lines.append(
            f"- {result['task'].upper()} stochastic-training seed "
            f"{result.get('stochastic_training_seed', 'D2')}: lambda-zero "
            f"`{result['lambda_zero']['byte_accuracy']['mean']:.6f}`, lambda-one "
            f"`{one['mean']:.6f}` (SD `{one['sample_sd']:.6f}`, 95% CI "
            f"`[{one['ci95'][0]:.6f}, {one['ci95'][1]:.6f}]`)."
        )
    lines.extend(
        [
            "",
            "D0 fixed-key parity and MC expected-gradient evidence cover affine, categorical-logit, "
            "embedding, and depthwise-convolution factors. New records correctly call D3 trials "
            "independent stochastic-training seeds; their non-codec initialization is fixed at seed 700.",
        ]
    )
    _write_text(OUTPUT / "RESULTS.md", "\n".join(lines) + "\n")
    return passed


def main() -> None:
    d1_results = d1()
    d2_results = d2()
    d3_results = d3()
    del d1_results, d2_results, d3_results
    print("PHASE_D_1_PASS" if finalize() else "PHASE_D_1_BLOCKED")


if __name__ == "__main__":
    main()
