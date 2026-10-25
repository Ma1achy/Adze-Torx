"""Phase-G engineering gates and bounded generated-structure smoke study."""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import pickle
import shutil
import subprocess
import sys
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from adze_t.backends.torx import TorxOperatorConfig, TorxOps
from adze_t.config import REFERENCE_SMALL_V0
from adze_t.model import apply_target_codec
from adze_t.objectives import adamw_init
from adze_t.phase_g import (
    evaluate_committed_replay,
    generated_structure_truth,
    generated_train_step,
    generated_two_pass_objective,
    neutral_structure,
)
from adze_t.phase_g_oracle import (
    descendant_fixture,
    enumerated_bridge_gradient,
    exact_value_and_gradient,
    fixture_parameters,
    monte_carlo_conformance,
)
from adze_t.state import CommittedStructure
from adze_t.teacher import canonical_teacher_structure


ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT / "results/runs/phase_f/f1/denoise_v1/one_step/init0_stoch0.pkl"
CODEC = ROOT / "results/phase_b/checkpoints/target_codec_b1.pkl"
EVIDENCE = ROOT / "results/phase_g"
WORK = ROOT / "results/runs/phase_g"
EXPECTED_PARENT_SHA256 = "520d3723d3a3187aa08ff3f467f434d43b579cdc4b0ce667727dfad96e83cf57"
EXPECTED_CODEC_SHA256 = "55f057f78f795f1585d7aac28fa7ff37846d4c1098507ede375e86b21c4f8ff6"
TORX_PIN = "f1fc858ed950ecd41935d15c06d0ec7c5e0674ae"
TRAIN_COUNT, VALIDATION_COUNT = 1024, 256
TRAIN_SEED, VALIDATION_SEED, STOCHASTIC_SEED = 950, 951, 952
CHECKPOINTS = (0, 10, 25, 50, 100)
ARMS = ("teacher", "argmax", "sample")
PROTOCOL = "PHASE_G_GENERATED_STRUCTURE_SMOKE_V0"
ROOT_COORDINATES = {"nu": 0, "operator": 1, "diffusion": 2, "structure": 3}


def import_artifacts(source_root: Path) -> dict[str, Any]:
    """Copy immutable prerequisites into this worktree without sharing run state."""
    relative_paths = (
        (Path("results/runs/phase_f/f1/denoise_v1/one_step/init0_stoch0.pkl"), PARENT),
        (Path("results/phase_b/checkpoints/target_codec_b1.pkl"), CODEC),
    )
    expected = (EXPECTED_PARENT_SHA256, EXPECTED_CODEC_SHA256)
    copied = []
    for (relative, destination), expected_sha in zip(relative_paths, expected, strict=True):
        source = source_root / relative
        source_sha = sha256(source) if source.is_file() else None
        if source_sha != expected_sha:
            raise RuntimeError(f"Phase-G prerequisite source mismatch: {source}")
        if destination.exists():
            if sha256(destination) != expected_sha:
                raise RuntimeError(
                    f"refusing to overwrite mismatched local artifact: {destination}"
                )
        else:
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_suffix(destination.suffix + ".importing")
            shutil.copy2(source, temporary)
            if sha256(temporary) != expected_sha:
                raise RuntimeError(f"copied Phase-G prerequisite failed hash: {temporary}")
            temporary.replace(destination)
        copy_sha = sha256(destination)
        copied.append(
            {
                "source": str(source.resolve()),
                "source_sha256": source_sha,
                "destination": str(destination.resolve()),
                "copy_sha256": copy_sha,
                "identical": source_sha == copy_sha == expected_sha,
            }
        )
    payload = {
        "status": "G_LOCAL_ARTIFACT_IMPORT_PASS",
        "implementation_git_sha": implementation_sha(),
        "artifacts": copied,
    }
    write_json(EVIDENCE / "artifact_import.json", payload)
    return payload


def config():
    return replace(
        REFERENCE_SMALL_V0,
        training=replace(REFERENCE_SMALL_V0.training, proposal_weight=0.0),
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def git_sha() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def implementation_sha() -> str:
    """Require runtime gates to use a committed, immutable implementation tree."""
    status = subprocess.check_output(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"], text=True
    )
    disallowed = []
    for line in status.splitlines():
        path = line[3:].split(" -> ")[-1]
        evidence_only = path.startswith("results/phase_g/") and Path(path).suffix in {
            ".json",
            ".jsonl",
        }
        if not evidence_only:
            disallowed.append(line)
    if disallowed:
        raise RuntimeError(
            "Phase-G runtime requires an immutable committed implementation; "
            f"disallowed worktree changes: {disallowed}"
        )
    return git_sha()


def load_pickle(path: Path) -> Any:
    with path.open("rb") as stream:
        value = pickle.load(stream)  # noqa: S301
    return jax.tree.map(
        lambda leaf: jnp.asarray(leaf) if isinstance(leaf, np.ndarray) else leaf,
        value,
    )


def save_pickle(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        pickle.dump(jax.device_get(value), stream, protocol=pickle.HIGHEST_PROTOCOL)
    temporary.replace(path)


def serialise(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    value = np.asarray(jax.device_get(value))
    if value.ndim == 0:
        return bool(value) if np.issubdtype(value.dtype, np.bool_) else float(value)
    return value.tolist()


def tree_all_finite(value: Any) -> bool:
    return all(bool(jnp.all(jnp.isfinite(leaf))) for leaf in jax.tree.leaves(value))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=serialise) + "\n")


def append_training_record(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    identity = (
        value["checkpoint"]["arm"],
        value["checkpoint"]["updates"],
        value["checkpoint"]["protocol"],
    )
    if path.exists():
        for line in path.read_text().splitlines():
            existing = json.loads(line)
            existing_identity = (
                existing["checkpoint"]["arm"],
                existing["checkpoint"]["updates"],
                existing["checkpoint"]["protocol"],
            )
            if existing_identity == identity:
                return
    with path.open("a") as stream:
        stream.write(json.dumps(value, sort_keys=True, default=serialise) + "\n")


def protocol_manifest() -> dict[str, Any]:
    cfg = config()
    return {
        "protocol": PROTOCOL,
        "implementation_git_sha": implementation_sha(),
        "parent_sha256": EXPECTED_PARENT_SHA256,
        "codec_sha256": EXPECTED_CODEC_SHA256,
        "torx_pin": TORX_PIN,
        "architecture": {"C": 32, "L_max": 4, "M_max": 32, "K": 8, "L": 4, "Q": 3, "R": 0},
        "optimizer": {
            "name": "AdamW",
            "learning_rate": 3.0e-4,
            "weight_decay": 0.01,
            "clip_norm": 1.0,
            "batch_size": 32,
            "fresh_moments": True,
            "initial_step": 0,
        },
        "data": {
            "domain": "iid bytes 1..32",
            "lengths": [8, 64],
            "counts": {"train": TRAIN_COUNT, "validation": VALIDATION_COUNT},
            "global_id_ranges": {
                "train": [0, TRAIN_COUNT - 1],
                "validation": [TRAIN_COUNT, TRAIN_COUNT + VALIDATION_COUNT - 1],
            },
            "seeds": [TRAIN_SEED, VALIDATION_SEED, STOCHASTIC_SEED],
            "stochastic_root_coordinates": ROOT_COORDINATES,
            "nu": [0.025, 0.25],
        },
        "structure_task": {
            "version": "PHASE_G_IDENTIFIABLE_CONTENT_CUTS_V0",
            "mandatory_cut_period": 8,
            "content_cut": "active carrier first clean codec slot byte in 1..16",
            "extent": "clean codec extent",
            "commit_ownership": (
                "mandatory K edges and terminal fixed; all optional nonmandatory edges "
                "and all extent sites owned"
            ),
            "predictor_observation": "F1-corrupted clean teacher carrier",
            "noisy_identifiability_nu": [0.025, 0.10, 0.25],
        },
        "interfaces": {
            "generated_structure": "PHASE_G_GENERATED_STRUCTURE_INTERFACE_V1",
            "training": "PHASE_G_TRAINING_PROTOCOL_V0",
            "score_estimator_owner": "local_score_bridge for sample; none otherwise",
        },
        "arms": list(ARMS),
        "checkpoints": list(CHECKPOINTS),
        "lambda_op": 0.0,
        "temperature_sample": 1.0,
        "two_pass": "predictor and committed replay share one corrupted S1 input; condition index zero",
        "loss": {
            "common": "boundary CE + extent CE + 0.5 predictor h-MSE",
            "descendant": "0.5 replay h-MSE + global compacted length-aware byte loss",
        },
        "config_batch_size": cfg.training.batch_size,
    }


def audit() -> dict[str, Any]:
    if sha256(PARENT) != EXPECTED_PARENT_SHA256:
        raise RuntimeError("Phase G F1 parent checkpoint hash mismatch")
    if sha256(CODEC) != EXPECTED_CODEC_SHA256:
        raise RuntimeError("Phase G codec checkpoint hash mismatch")
    parent = load_pickle(PARENT)
    if set(parent) != {"params", "moments", "step"} or int(parent["step"]) != 5000:
        raise RuntimeError("Phase G requires exact accepted F1 step-5000 state")
    payload = {
        "status": "G_PROVENANCE_PASS",
        "manifest": protocol_manifest(),
        "parent_optimizer_step_ignored_for_fresh_g_optimizer": int(parent["step"]),
    }
    write_json(EVIDENCE / "provenance.json", payload)
    return payload


def dataset(count: int, seed: int, *, global_id_offset: int = 0) -> dict[str, jax.Array]:
    cfg = config()
    key = jax.random.key(seed)
    length_key, byte_key = jax.random.split(key)
    lengths = jax.random.randint(length_key, (count,), 8, 65)
    target = jax.random.randint(byte_key, (count, 64), 1, 33, dtype=jnp.int32)
    mask = jnp.arange(64)[None, :] < lengths[:, None]
    target = jnp.where(mask, target, 0)
    teacher = canonical_teacher_structure(target, mask, cfg)
    truth = generated_structure_truth(teacher.slot_bytes, teacher.length, K=cfg.packing.K)
    if truth.length is None:
        raise AssertionError("Phase-G task truth requires explicit lengths")
    return {
        "prompt": jnp.full((count, 1), 0xF1, jnp.int32),
        "prompt_mask": jnp.ones((count, 1), bool),
        "target": target,
        "target_mask": mask,
        "global_example_id": jnp.arange(
            global_id_offset, global_id_offset + count, dtype=jnp.uint32
        ),
        "structure_boundary": truth.c_b,
        "structure_length": truth.length,
    }


def stochastic_root(namespace: str) -> jax.Array:
    return jax.random.fold_in(jax.random.key(STOCHASTIC_SEED), ROOT_COORDINATES[namespace])


def with_nu(batch: dict[str, jax.Array], occurrence: int) -> dict[str, jax.Array]:
    nu = jax.random.uniform(
        jax.random.fold_in(stochastic_root("nu"), occurrence),
        (batch["target"].shape[0],),
        minval=0.025,
        maxval=0.25,
    )
    return {**batch, "nu": nu}


def take_batch(data: dict[str, jax.Array], indices: jax.Array) -> dict[str, jax.Array]:
    return jax.tree.map(lambda value: value[indices], data)


def _noisy_structure_identifiability(params: Any) -> list[dict[str, Any]]:
    """Use the accepted F1 denoise/decoder as a witness over G's corruption range."""
    cfg = config()
    data = dataset(VALIDATION_COUNT, VALIDATION_SEED, global_id_offset=TRAIN_COUNT)
    teacher = canonical_teacher_structure(data["target"], data["target_mask"], cfg)
    truth = generated_structure_truth(teacher.slot_bytes, teacher.length, K=cfg.packing.K)
    neutral = neutral_structure(VALIDATION_COUNT, cfg)
    carrier = jnp.arange(cfg.carrier.C)[None, :]
    mandatory = (carrier + 1) % cfg.packing.K == 0
    eligible = truth.activity & ~mandatory
    rows = []
    for nu in (0.025, 0.10, 0.25):
        batch = {**data, "nu": jnp.full((VALIDATION_COUNT,), nu)}
        _, replay = evaluate_committed_replay(
            params,
            batch,
            neutral,
            stochastic_root("operator"),
            stochastic_root("diffusion"),
            0,
            config=cfg,
        )
        decoded_slots = jnp.argmax(replay.byte_logits, axis=-1).astype(jnp.int32)
        decoded_length = jnp.argmax(replay.l_logits, axis=-1).astype(jnp.int32)
        witness = generated_structure_truth(decoded_slots, decoded_length, K=cfg.packing.K)
        witness_correct = (witness.c_b == truth.c_b) & eligible
        direct = jnp.argmax(replay.b_logits, axis=-1).astype(jnp.int32)
        direct_correct = (direct == truth.c_b) & eligible
        count = int(jnp.sum(eligible))
        successes = int(jnp.sum(witness_correct))
        accuracy = successes / max(count, 1)
        z = 1.959963984540054
        denominator = 1.0 + z * z / max(count, 1)
        centre = accuracy + z * z / (2 * max(count, 1))
        radius = z * np.sqrt(
            accuracy * (1 - accuracy) / max(count, 1) + z * z / (4 * max(count, 1) ** 2)
        )
        rows.append(
            {
                "nu": nu,
                "eligible_content_cuts": count,
                "decoder_witness_content_accuracy": accuracy,
                "decoder_witness_wilson_95_lower": (centre - radius) / denominator,
                "untrained_direct_boundary_accuracy_diagnostic": float(
                    jnp.sum(direct_correct) / jnp.maximum(jnp.sum(eligible), 1)
                ),
                "finite": bool(
                    jnp.all(jnp.isfinite(replay.h_hat)) & jnp.all(jnp.isfinite(replay.byte_logits))
                ),
            }
        )
    return rows


def run_codec_gate() -> dict[str, Any]:
    audit()
    state = load_pickle(PARENT)
    cfg = config()
    ops = TorxOps.create(
        stochastic_root("operator"),
        config=TorxOperatorConfig(operator_stochasticity=True, lambda_op=0.0),
        optimizer_step=0,
    )
    rows = []
    for length in range(8, 65):
        target = jax.random.randint(
            jax.random.fold_in(jax.random.key(VALIDATION_SEED), length), (1, 64), 1, 33
        )
        mask = jnp.arange(64)[None, :] < length
        target = jnp.where(mask, target, 0)
        output = apply_target_codec(state["params"], target, mask, config=cfg, ops=ops)
        teacher = output["target"]["teacher"]
        predicted = jnp.argmax(output["codec_logits"], axis=-1)
        byte_accuracy = jnp.sum((predicted == teacher.slot_bytes) & teacher.slot_mask) / jnp.sum(
            teacher.slot_mask
        )
        ell = jnp.argmax(output["target"]["l_logits"], axis=-1)
        task_truth = generated_structure_truth(teacher.slot_bytes, teacher.length, K=cfg.packing.K)
        decoded_truth = generated_structure_truth(predicted, ell, K=cfg.packing.K)
        signature = jnp.mean(
            jnp.concatenate(
                (
                    decoded_truth.c_b[:, :-1] == task_truth.c_b[:, :-1],
                    decoded_truth.length == task_truth.length,
                ),
                axis=1,
            )
        )
        rows.append(
            {
                "length": length,
                "accuracy": float(byte_accuracy),
                "signature_accuracy": float(signature),
                "capacity_overflow": bool(teacher.capacity_overflow[0]),
                "nonfinite": not bool(jnp.all(jnp.isfinite(output["codec_logits"]))),
            }
        )
    aggregate = float(np.mean([row["accuracy"] for row in rows]))
    signature = float(np.mean([row["signature_accuracy"] for row in rows]))
    noisy_identifiability = _noisy_structure_identifiability(state["params"])
    passed = (
        aggregate >= 0.99
        and min(row["accuracy"] for row in rows) >= 0.98
        and signature >= 0.99
        and not any(row["capacity_overflow"] or row["nonfinite"] for row in rows)
        and all(
            row["finite"]
            and row["eligible_content_cuts"] > 0
            and row["decoder_witness_wilson_95_lower"] > 0.5
            for row in noisy_identifiability
        )
    )
    payload = {
        "status": "G_CODEC_TASK_COMPATIBILITY_PASS"
        if passed
        else "G_CODEC_TASK_COMPATIBILITY_BLOCKED",
        "protocol": protocol_manifest(),
        "aggregate_accuracy": aggregate,
        "minimum_length_accuracy": min(row["accuracy"] for row in rows),
        "signature_accuracy": signature,
        "noisy_identifiability": noisy_identifiability,
        "noisy_identifiability_rule": (
            "accepted F1 denoise/decoder witness on the same corrupted carrier; "
            "for content-dependent cuts, every nu Wilson-95 lower bound must exceed chance"
        ),
        "rows": rows,
    }
    write_json(EVIDENCE / "codec_compatibility.json", payload)
    if not passed:
        raise RuntimeError("G_CODEC_TASK_COMPATIBILITY_BLOCKED")
    return payload


def run_oracle_gate() -> dict[str, Any]:
    """Run the frozen exact and sixteen-root production-loss score oracle."""
    audit()
    rows = []
    passed = True
    descendant_names = (
        "inactive",
        "reactivation",
        "zero",
        "missing",
        "extra",
        "direct_dependencies",
        "baseline",
    )
    for parameter_name in ("uniform", "asymmetric", "peaked"):
        parameters = fixture_parameters(parameter_name)
        for descendant_name in descendant_names:
            descendant = descendant_fixture(descendant_name)
            _, exact = exact_value_and_gradient(parameters, descendant)
            bridge = enumerated_bridge_gradient(parameters, descendant)
            exact_ok = jax.tree_util.tree_all(
                jax.tree.map(
                    lambda left, right: jnp.allclose(left, right, rtol=1.0e-6, atol=2.0e-6),
                    exact,
                    bridge,
                )
            )
            mc = monte_carlo_conformance(parameters, descendant, root_seed=8100)
            accepted = bool(exact_ok) and bool(mc["accepted"])
            passed &= accepted
            rows.append(
                {
                    "parameter_fixture": parameter_name,
                    "descendant_fixture": descendant_name,
                    "exact_bridge_accepted": bool(exact_ok),
                    "monte_carlo_accepted": bool(mc["accepted"]),
                    "statistics": mc,
                }
            )
    payload = {
        "status": "G_GRADIENT_ORACLE_PASS" if passed else "G_GRADIENT_ORACLE_FAIL",
        "protocol": protocol_manifest(),
        "root_seed": 8100,
        "roots": 16,
        "samples_per_root": 4096,
        "componentwise_limit_root_se": 4.0,
        "zero_se_absolute_tolerance": 1.0e-6,
        "rows": rows,
    }
    write_json(EVIDENCE / "gradient_oracle.json", payload)
    if not passed:
        raise RuntimeError("G sampled-gradient oracle failed; sampled smoke arm remains disabled")
    return payload


def run_engineering_preflight() -> dict[str, Any]:
    """Run durable correctness gates required before any smoke arm."""
    audit()
    commands = (
        (
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "tests/test_phase_g_structure.py",
            "tests/test_phase_g_integration.py",
            "-m",
            "not slow",
        ),
        (sys.executable, "scripts/check_public_boundaries.py"),
        (
            "ruff",
            "check",
            "src/adze_t/generated_structure.py",
            "src/adze_t/phase_g.py",
            "src/adze_t/phase_g_oracle.py",
            "scripts/run_phase_g.py",
            "tests/test_phase_g_structure.py",
            "tests/test_phase_g_integration.py",
            "tests/test_phase_g_oracle.py",
        ),
    )
    records = []
    passed = True
    for command in commands:
        completed = subprocess.run(
            command,
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        passed &= completed.returncode == 0
        records.append(
            {
                "command": list(command),
                "returncode": completed.returncode,
                "stdout": completed.stdout,
                "stderr": completed.stderr,
            }
        )
        if completed.returncode != 0:
            break
    payload = {
        "status": "G_ENGINEERING_PREFLIGHT_PASS" if passed else "G_ENGINEERING_PREFLIGHT_BLOCKED",
        "protocol": protocol_manifest(),
        "covered_gates": [
            "eager_and_jit_structure_contract",
            "fail_closed_routing_loss_and_optimizer",
            "sanitized_forward_target_leakage",
            "packing_identity_zero_extent_and_reactivation",
            "finite_full_model_gradient_and_optimizer_update",
            "frozen_teacher_and_rho",
            "public_dependency_boundaries",
            "focused_static_lint",
        ],
        "commands": records,
    }
    write_json(EVIDENCE / "engineering_preflight.json", payload)
    if not passed:
        raise RuntimeError("G_ENGINEERING_PREFLIGHT_BLOCKED")
    return payload


def _bootstrap_interval(values: np.ndarray, seed: int) -> tuple[float, float]:
    indices = np.asarray(
        jax.random.randint(jax.random.key(seed), (2000, len(values)), 0, len(values))
    )
    means = np.mean(values[indices], axis=1)
    lower, upper = np.percentile(means, [2.5, 97.5])
    return float(lower), float(upper)


def _control_structures(data: dict[str, jax.Array]) -> dict[str, CommittedStructure]:
    cfg = config()
    correct_b, correct_l = data["structure_boundary"], data["structure_length"]
    carrier = jnp.arange(cfg.carrier.C)[None, :]
    mandatory = (carrier + 1) % cfg.packing.K == 0
    extra = (correct_b > 0) & ~mandatory
    if cfg.carrier.C % cfg.packing.K:
        raise ValueError("shifted-cut control requires complete fixed K blocks")
    extra_blocks = extra.reshape(correct_b.shape[0], cfg.carrier.C // cfg.packing.K, cfg.packing.K)
    shifted_extra = jnp.concatenate(
        (
            jnp.roll(extra_blocks[:, :, :-1], 1, axis=2),
            jnp.zeros_like(extra_blocks[:, :, -1:]),
        ),
        axis=2,
    ).reshape(extra.shape)
    shifted = (mandatory | shifted_extra).astype(jnp.int32).at[:, -1].set(1)
    permuted = jnp.roll(correct_l, 1, axis=1)
    neutral = neutral_structure(correct_b.shape[0], cfg)
    return {
        "correct": CommittedStructure(correct_b, correct_l > 0, correct_l),
        "shifted_cuts": CommittedStructure(shifted, correct_l > 0, correct_l),
        "permuted_extents": CommittedStructure(correct_b, permuted > 0, permuted),
        "joint": CommittedStructure(shifted, permuted > 0, permuted),
        "fixed_neutral": neutral,
    }


def run_preflight(checkpoint: Path) -> dict[str, Any]:
    state = load_pickle(checkpoint)
    if state.get("protocol") != protocol_manifest() or state.get("arm") != "teacher":
        raise RuntimeError(f"stale or mixed Phase-G teacher checkpoint: {checkpoint}")
    data = with_nu(dataset(VALIDATION_COUNT, VALIDATION_SEED, global_id_offset=TRAIN_COUNT), 0)
    structures = _control_structures(data)
    results = {}
    for name, structure in structures.items():
        metrics, replay = evaluate_committed_replay(
            state["params"],
            data,
            structure,
            stochastic_root("operator"),
            stochastic_root("diffusion"),
            0,
            config=config(),
        )
        if not bool(replay.executed):
            raise RuntimeError(f"preflight structure failed routing: {name}")
        results[name] = {
            "accuracy": np.asarray(metrics.accuracy_per_example),
            "loss": np.asarray(metrics.loss_per_example),
        }
    correct = results["correct"]
    comparisons = {}
    passed = True
    for index, name in enumerate(("shifted_cuts", "permuted_extents", "joint", "fixed_neutral")):
        control = results[name]
        accuracy_delta = correct["accuracy"] - control["accuracy"]
        relative_loss = (control["loss"] - correct["loss"]) / np.maximum(control["loss"], 1e-12)
        accuracy_ci = _bootstrap_interval(accuracy_delta, 9600 + index)
        loss_ci = _bootstrap_interval(relative_loss, 9700 + index)
        accepted = (
            float(np.mean(accuracy_delta)) >= 0.01
            and float(np.mean(relative_loss)) >= 0.02
            and accuracy_ci[0] > 0
            and loss_ci[0] > 0
        )
        passed &= accepted
        comparisons[name] = {
            "accuracy_delta": float(np.mean(accuracy_delta)),
            "accuracy_bootstrap_95": accuracy_ci,
            "relative_loss_reduction": float(np.mean(relative_loss)),
            "relative_loss_bootstrap_95": loss_ci,
            "accepted": accepted,
        }
    payload = {
        "status": ("G_INTEGRATION_PREFLIGHT_PASS" if passed else "G_INTEGRATION_PREFLIGHT_BLOCKED"),
        "checkpoint": {
            "path": str(checkpoint),
            "sha256": sha256(checkpoint),
            "updates": int(state["updates"]),
            "optimizer_step": int(state["optimizer_step"]),
        },
        "protocol": state["protocol"],
        "comparisons": comparisons,
    }
    write_json(EVIDENCE / "integration_preflight.json", payload)
    if not passed:
        raise RuntimeError("teacher structure-necessity preflight failed")
    return payload


def _initial_state(arm: str) -> dict[str, Any]:
    parent = load_pickle(PARENT)
    return {
        "protocol": protocol_manifest(),
        "arm": arm,
        "params": parent["params"],
        "moments": (adamw_init(parent["params"]), adamw_init(parent["params"])),
        "optimizer_step": 0,
        "updates": 0,
        "new_leaf_initialization_seed": 703,
        "new_leaves": [],
    }


def _checkpoint_record(state: dict[str, Any], path: Path) -> dict[str, Any]:
    return {
        "arm": state["arm"],
        "updates": int(state["updates"]),
        "optimizer_step": int(state["optimizer_step"]),
        "path": str(path),
        "sha256": sha256(path),
        "protocol": state["protocol"],
    }


def _validation_metrics(params: Any, arm: str, updates: int) -> dict[str, Any]:
    """Evaluate deterministic teacher and stopped-argmax commits on the fixed split."""
    validation = with_nu(
        dataset(VALIDATION_COUNT, VALIDATION_SEED, global_id_offset=TRAIN_COUNT), 0
    )
    rows = {}
    for evaluation_arm in ("teacher", "argmax"):
        loss, metrics = generated_two_pass_objective(
            params,
            validation,
            stochastic_root("operator"),
            stochastic_root("diffusion"),
            stochastic_root("structure"),
            updates,
            arm=evaluation_arm,
            config=config(),
        )
        rows[evaluation_arm] = {"loss": loss, **metrics}
    return rows


def _same_saved_state(left: dict[str, Any], right: dict[str, Any]) -> bool:
    if (
        left.get("protocol") != right.get("protocol")
        or left.get("arm") != right.get("arm")
        or int(left.get("updates", -1)) != int(right.get("updates", -2))
        or int(left.get("optimizer_step", -1)) != int(right.get("optimizer_step", -2))
    ):
        return False
    return bool(
        jax.tree_util.tree_all(
            jax.tree.map(
                jnp.array_equal,
                (left["params"], left["moments"]),
                (right["params"], right["moments"]),
            )
        )
    )


def _persist_checkpoint(state: dict[str, Any], metrics: dict[str, Any] | None) -> Path:
    update = int(state["updates"])
    checkpoint = WORK / state["arm"] / f"update_{update:04d}.pkl"
    if checkpoint.exists():
        existing = load_pickle(checkpoint)
        if not _same_saved_state(existing, state):
            raise RuntimeError(f"stale or mixed Phase-G checkpoint: {checkpoint}")
    else:
        save_pickle(checkpoint, state)
    record = {
        "checkpoint": _checkpoint_record(state, checkpoint),
        "training_metrics": metrics,
        "validation": _validation_metrics(state["params"], state["arm"], update),
    }
    write_json(WORK / state["arm"] / f"update_{update:04d}.json", record)
    return checkpoint


def train_arm(arm: str) -> Path:
    if arm not in ARMS:
        raise ValueError(arm)
    state_path = WORK / arm / "state.pkl"
    fresh = not state_path.exists()
    state = load_pickle(state_path) if not fresh else _initial_state(arm)
    if state["protocol"] != protocol_manifest() or state["arm"] != arm:
        raise RuntimeError(f"stale/mixed Phase G state: {state_path}")
    if fresh:
        checkpoint = _persist_checkpoint(state, None)
        append_training_record(
            EVIDENCE / "training.jsonl",
            {"checkpoint": _checkpoint_record(state, checkpoint), "training_metrics": None},
        )
        save_pickle(state_path, state)
    train = dataset(TRAIN_COUNT, TRAIN_SEED)
    update = jax.jit(
        generated_train_step,
        static_argnames=("arm", "temperature", "config"),
    )
    while state["updates"] < 100:
        update_index = state["updates"]
        permutation = jax.random.permutation(
            jax.random.fold_in(jax.random.key(TRAIN_SEED), update_index // 32), TRAIN_COUNT
        )
        start = (update_index * 32) % TRAIN_COUNT
        indices = permutation[start : start + 32]
        batch = take_batch(train, indices)
        batch = with_nu(batch, update_index)
        params, moments, optimizer_step, metrics = update(
            state["params"],
            state["moments"],
            state["optimizer_step"],
            batch,
            stochastic_root("operator"),
            stochastic_root("diffusion"),
            stochastic_root("structure"),
            arm=arm,
            temperature=1.0 if arm == "sample" else None,
            config=config(),
        )
        if not bool(metrics["executed"]):
            raise RuntimeError(f"Phase G {arm} produced invalid generated routing")
        if not tree_all_finite((params, moments)):
            raise RuntimeError(f"Phase G {arm} produced nonfinite optimizer state")
        state = {
            **state,
            "params": params,
            "moments": moments,
            "optimizer_step": int(optimizer_step),
            "updates": update_index + 1,
        }
        if state["updates"] in CHECKPOINTS:
            checkpoint = _persist_checkpoint(state, metrics)
            append_training_record(
                EVIDENCE / "training.jsonl",
                {
                    "checkpoint": _checkpoint_record(state, checkpoint),
                    "training_metrics": metrics,
                },
            )
        save_pickle(state_path, state)
    return state_path


def run_smoke() -> dict[str, Any]:
    audit()
    codec = EVIDENCE / "codec_compatibility.json"
    oracle = EVIDENCE / "gradient_oracle.json"
    engineering = EVIDENCE / "engineering_preflight.json"
    required = (
        (codec, "G_CODEC_TASK_COMPATIBILITY_PASS"),
        (oracle, "G_GRADIENT_ORACLE_PASS"),
        (engineering, "G_ENGINEERING_PREFLIGHT_PASS"),
    )
    for path, status in required:
        if not path.exists():
            raise RuntimeError(f"missing durable Phase-G gate: {path}")
        evidence = json.loads(path.read_text())
        if evidence.get("status") != status or evidence.get("protocol") != protocol_manifest():
            raise RuntimeError(f"stale, mixed, or failed Phase-G gate: {path}")
    teacher = train_arm("teacher")
    run_preflight(teacher)
    argmax = train_arm("argmax")
    sample = train_arm("sample")
    arm_records = {}
    for arm, path in (("teacher", teacher), ("argmax", argmax), ("sample", sample)):
        state = load_pickle(path)
        arm_records[arm] = {
            "state": str(path),
            "state_sha256": sha256(path),
            "updates": int(state["updates"]),
            "optimizer_step": int(state["optimizer_step"]),
            "protocol": state["protocol"],
        }
    payload = {
        "status": "G_SMOKE_COMPLETE_NO_SCIENTIFIC_LABEL",
        "protocol": protocol_manifest(),
        "arms": arm_records,
    }
    write_json(EVIDENCE / "smoke_results.json", payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        required=True,
        choices=(
            "audit",
            "import-artifacts",
            "engineering-preflight",
            "oracle",
            "codec",
            "preflight",
            "smoke",
        ),
    )
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--artifact-source-root", type=Path)
    args = parser.parse_args()
    if args.stage == "import-artifacts":
        if args.artifact_source_root is None:
            raise ValueError("--artifact-source-root is required for artifact import")
        import_artifacts(args.artifact_source_root)
    elif args.stage == "audit":
        audit()
    elif args.stage == "engineering-preflight":
        run_engineering_preflight()
    elif args.stage == "oracle":
        run_oracle_gate()
    elif args.stage == "codec":
        run_codec_gate()
    elif args.stage == "preflight":
        if args.checkpoint is None:
            raise ValueError("--checkpoint is required for preflight")
        run_preflight(args.checkpoint)
    else:
        run_smoke()


if __name__ == "__main__":
    main()
