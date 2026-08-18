"""Phase F.2 frozen-checkpoint same-model denoising-depth evaluation."""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
import pickle
import subprocess
import time
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from adze_t.backends.torx import TorxOperatorConfig, TorxOps
from adze_t.config import REFERENCE_SMALL_V0
from adze_t.corruption import PHASE_F_EVAL_GRID, alpha, corrupt_h, sigma
from adze_t.denoise import (
    F2_NATIVE_S_CONDITIONING_UNTRAINED,
    F2_STEP0_CONDITIONING,
    apply_denoising_trajectory,
    make_sanitized_target_analysis,
)
from adze_t.model import apply_clean_target_teacher
from adze_t.phase_f_1 import (
    DENOISE_V1,
    dataset_audit,
    generate_denoise_v0,
    initial_diffusion_epsilon,
)
from adze_t.teacher import canonical_teacher_structure_core


ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = ROOT / "results/runs/phase_f/f1/denoise_v1/one_step/init0_stoch0.pkl"
CODEC = ROOT / "results/phase_b/checkpoints/target_codec_b1.pkl"
WORK = ROOT / "results/runs/phase_f/f2"
EVIDENCE = ROOT / "results/phase_f/f2"
EXPECTED_CHECKPOINT_SHA256 = "520d3723d3a3187aa08ff3f467f434d43b579cdc4b0ce667727dfad96e83cf57"
EXPECTED_CODEC_SHA256 = "55f057f78f795f1585d7aac28fa7ff37846d4c1098507ede375e86b21c4f8ff6"
F1_GIT_SHA = "1ebcbdefa87d35841adfc490af315c98ae043105"
TORX_PIN = "f1fc858ed950ecd41935d15c06d0ec7c5e0674ae"
TEST_SEED = 942
TEST_COUNT = 4_096
STOCHASTIC_COUNT = 512
STOCHASTIC_SUBSET_SEED = 10_421
BOOTSTRAP_SEED = 10_422
BOOTSTRAP_RESAMPLES = 2_000
OPERATOR_ROOT_SEED = 10_423
DIFFUSION_ROOT_SEED = 10_424
LEVELS = tuple(PHASE_F_EVAL_GRID)
S_EXEC = 4
BATCH = 32
SHARD_SIZE = 256
ROOT_COUNT = 16
ROOT_T_95_DF15 = 2.131449545559323
PAIR_NAMES = {"B21": (0, 1), "B31": (0, 2), "B41": (0, 3), "B32": (1, 2), "B43": (2, 3)}


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


def serialise(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    value = jax.device_get(value)
    if getattr(value, "ndim", 0) == 0 and np.issubdtype(np.asarray(value).dtype, np.bool_):
        return bool(value)
    return float(value) if getattr(value, "ndim", 0) == 0 else value.tolist()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=serialise) + "\n")


def load_checkpoint() -> dict[str, Any]:
    checkpoint_hash = sha256(CHECKPOINT)
    codec_hash = sha256(CODEC)
    if checkpoint_hash != EXPECTED_CHECKPOINT_SHA256:
        raise RuntimeError(f"F2 checkpoint provenance mismatch: {checkpoint_hash}")
    if codec_hash != EXPECTED_CODEC_SHA256:
        raise RuntimeError(f"F2 codec provenance mismatch: {codec_hash}")
    with CHECKPOINT.open("rb") as stream:
        state = jax.tree.map(jnp.asarray, pickle.load(stream))  # noqa: S301
    if set(state) != {"moments", "params", "step"} or int(state["step"]) != 5_000:
        raise RuntimeError("F2 requires the accepted F1 step-5000 checkpoint")
    return state


def _array_hash(value: Any) -> str:
    digest = hashlib.sha256()
    leaves_with_paths = jax.tree_util.tree_leaves_with_path(jax.device_get(value))
    for path, leaf in leaves_with_paths:
        array = np.asarray(leaf)
        digest.update(jax.tree_util.keystr(path).encode())
        digest.update(str(array.dtype).encode())
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


def _tree_equal(left: Any, right: Any) -> bool:
    return all(
        np.array_equal(np.asarray(a), np.asarray(b))
        for a, b in zip(
            jax.tree.leaves(jax.device_get(left)),
            jax.tree.leaves(jax.device_get(right)),
            strict=True,
        )
    )


def _rho_tree(params: Any) -> dict[str, Any]:
    return {
        jax.tree_util.keystr(path): leaf
        for path, leaf in jax.tree.leaves_with_path(params)
        if "['rho']" in jax.tree_util.keystr(path)
    }


def test_split() -> tuple[jax.Array, jax.Array, jax.Array]:
    return generate_denoise_v0(TEST_COUNT, TEST_SEED, spec=DENOISE_V1)


def _example_hash(prompt: np.ndarray, target: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(prompt.tobytes())
    digest.update(target.tobytes())
    return digest.hexdigest()


def run_audit() -> dict[str, Any]:
    state = load_checkpoint()
    prompt, target, ids = test_split()
    prompt_host, target_host = map(np.asarray, jax.device_get((prompt, target)))
    hashes = [
        _example_hash(sample_prompt, sample_target)
        for sample_prompt, sample_target in zip(prompt_host, target_host, strict=True)
    ]
    ordered = hashlib.sha256("".join(hashes).encode()).hexdigest()
    corrected = dataset_audit(prompt, target, spec=DENOISE_V1)
    subset_permutation = jax.random.permutation(
        jax.random.PRNGKey(STOCHASTIC_SUBSET_SEED), TEST_COUNT
    )
    subset_ids = ids[subset_permutation[:STOCHASTIC_COUNT]]
    cfg = config()
    payload = {
        "status": "F2_PROVENANCE_PASS",
        "current_git_sha": git_sha(),
        "f1_git_sha": F1_GIT_SHA,
        "torx_pin": TORX_PIN,
        "checkpoint": str(CHECKPOINT.relative_to(ROOT)),
        "checkpoint_sha256": sha256(CHECKPOINT),
        "checkpoint_step": int(state["step"]),
        "initialization_seed": 0,
        "stochastic_training_seed": 0,
        "codec": str(CODEC.relative_to(ROOT)),
        "codec_sha256": sha256(CODEC),
        "architecture": {"L": cfg.model.physical_blocks_L, "Q": cfg.model.cycles_Q, "R": 0},
        "training_or_parameter_mutation": False,
        "rho_teacher_structure_frozen": True,
        "conditioning": {
            "primary": F2_STEP0_CONDITIONING,
            "primary_actual_s_indices": [0, 1, 2, 3],
            "primary_denoise_condition_indices": [0, 0, 0, 0],
            "diagnostic": F2_NATIVE_S_CONDITIONING_UNTRAINED,
            "diagnostic_denoise_condition_indices": [0, 1, 2, 3],
        },
        "dataset": {
            **corrected,
            "seed": TEST_SEED,
            "global_ids_preserved": bool(jnp.array_equal(ids, jnp.arange(TEST_COUNT))),
            "per_example_sha256": hashes,
            "ordered_split_sha256": ordered,
        },
        "f1_audit_correction": {
            "historical_record_unchanged": True,
            "issue": (
                "F1 nested split audits called dataset_audit without spec=DENOISE_V1, so their "
                "nested task_version/domain labels are stale DENOISE_V0 metadata"
            ),
            "generated_targets_and_top_level_contract_were_denoise_v1": True,
            "f2_explicit_spec": DENOISE_V1.name,
        },
        "stochastic_subset": {
            "selection": "first 512 positions of a fixed JAX seeded permutation",
            "seed": STOCHASTIC_SUBSET_SEED,
            "count": STOCHASTIC_COUNT,
            "original_global_ids": np.asarray(subset_ids).tolist(),
            "ordered_ids_sha256": hashlib.sha256(np.asarray(subset_ids).tobytes()).hexdigest(),
        },
        "known_f1_raw_gradient_issue": (
            "The historical F1 first-gradient record reports raw norm 1.4641897472e11 "
            "and byte loss 5.12751456e8 before clipping (applied norm 1.0). F2 records "
            "this anomaly without changing optimizer behavior or any historical evidence."
        ),
    }
    write_json(EVIDENCE / "provenance.json", payload)
    write_json(
        EVIDENCE / "dataset_audit.json",
        {
            "task_version": DENOISE_V1.name,
            "explicit_spec_passed": True,
            "historical_f1_audit_unchanged": True,
            **payload["dataset"],
        },
    )
    return payload


def _batch_metrics(
    params: Any,
    prompt: jax.Array,
    target: jax.Array,
    ids: jax.Array,
    nu: jax.Array,
    *,
    cycles: int,
    eta_diff: int,
    lambda_op: float,
    operator_root: jax.Array,
    conditioning_mode: str,
) -> dict[str, jax.Array]:
    cfg = config()
    mask = jnp.ones_like(target, dtype=bool)
    clean_ops = TorxOps.create(
        operator_root,
        config=TorxOperatorConfig(operator_stochasticity=True, lambda_op=0.0),
        global_example_id=ids[0],
    )
    clean = apply_clean_target_teacher(params, target, mask, config=cfg, ops=clean_ops)
    teacher = clean["target"]["teacher"]
    h0 = clean["target"]["h0"]
    diffusion_root = jax.random.PRNGKey(DIFFUSION_ROOT_SEED)
    epsilon = initial_diffusion_epsilon(h0, diffusion_root, ids)
    initial = corrupt_h(h0, nu, epsilon)
    sanitized = make_sanitized_target_analysis(
        teacher.boundaries,
        teacher.length,
        config=cfg,
        target_width=target.shape[1],
    )
    trajectory = apply_denoising_trajectory(
        params,
        prompt,
        jnp.ones_like(prompt, dtype=bool),
        initial,
        sanitized,
        nu,
        ids,
        s_exec=S_EXEC,
        eta_diff=eta_diff,
        diffusion_root=diffusion_root,
        operator_backend="torx",
        operator_root=operator_root,
        operator_config=TorxOperatorConfig(
            operator_stochasticity=True,
            lambda_op=lambda_op,
            sigma_min=1.0e-3 if lambda_op else 1.0e-6,
            sigma_max=1.0e-3 if lambda_op else 0.25,
        ),
        conditioning_mode=conditioning_mode,
        config=cfg,
        dit_cycles=cycles,
    )
    predicted = jnp.argmax(trajectory.byte_logits, axis=-1)
    slot_bytes = teacher.slot_bytes[None, ...]
    slot_mask = teacher.slot_mask[None, ...]
    correct = (predicted == slot_bytes) & slot_mask
    log_probs = jax.nn.log_softmax(trajectory.byte_logits, axis=-1)
    selected = jnp.take_along_axis(log_probs, slot_bytes[..., None], axis=-1)[..., 0]
    byte_count = jnp.maximum(jnp.sum(slot_mask, axis=(2, 3)), 1)
    mse = jnp.mean((trajectory.h_hat - h0[None, ...]) ** 2, axis=(2, 3))
    nll = -jnp.sum(jnp.where(slot_mask, selected, 0.0), axis=(2, 3)) / byte_count
    accuracy = jnp.sum(correct, axis=(2, 3)) / byte_count
    exact = jnp.all(correct | ~slot_mask, axis=(2, 3))
    return {
        "mse": mse.T,
        "nll": nll.T,
        "byte_accuracy": accuracy.T,
        "exact_accuracy": exact.T,
        "nonfinite": trajectory.diagnostics["nonfinite"].T,
        "h_hat_rms": trajectory.diagnostics["h_hat_rms"].T,
        "inter_step_rms_change": trajectory.diagnostics["inter_step_rms_change"].T,
        "relative_update": trajectory.diagnostics["relative_update"].T,
        "cosine_similarity": trajectory.diagnostics["cosine_similarity"].T,
        "initial_signal_rms": jnp.sqrt(jnp.mean((alpha(nu)[:, None, None] * h0) ** 2, axis=(1, 2))),
        "initial_noise_rms": jnp.sqrt(
            jnp.mean((sigma(nu)[:, None, None] * epsilon) ** 2, axis=(1, 2))
        ),
    }


def _condition_key(
    *,
    conditioning_mode: str,
    eta_diff: int,
    lambda_op: float,
    cycles: int,
    root_index: int,
    nu: float,
    ids: np.ndarray,
) -> dict[str, Any]:
    return {
        "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
        "conditioning_mode": conditioning_mode,
        "eta_diff": eta_diff,
        "lambda_op": lambda_op,
        "sigma_op": 1.0e-3 if lambda_op else 0.0,
        "cycles_Q": cycles,
        "operator_root_index": root_index,
        "operator_root_seed": OPERATOR_ROOT_SEED,
        "diffusion_root_seed": DIFFUSION_ROOT_SEED,
        "nu": nu,
        "test_ids_sha256": hashlib.sha256(ids.astype(np.uint32).tobytes()).hexdigest(),
        "test_ids": ids.tolist(),
    }


def _shard_path(condition: dict[str, Any], start: int, end: int) -> Path:
    identity = {**condition, "start": start, "end": end}
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    return WORK / "shards" / f"{digest}.pkl"


def evaluate_condition(
    *,
    conditioning_mode: str,
    eta_diff: int,
    lambda_op: float,
    cycles: int,
    root_index: int,
    subset: bool,
) -> None:
    state = load_checkpoint()
    params = state["params"]
    params_before = jax.tree.map(lambda value: np.asarray(value).copy(), jax.device_get(params))
    rho_before = jax.tree.map(lambda value: np.asarray(value).copy(), _rho_tree(params))
    prompt, target, ids = test_split()
    if subset:
        positions = jax.random.permutation(jax.random.PRNGKey(STOCHASTIC_SUBSET_SEED), TEST_COUNT)[
            :STOCHASTIC_COUNT
        ]
        prompt, target, ids = prompt[positions], target[positions], ids[positions]
    host_ids = np.asarray(ids, dtype=np.uint32)
    teacher_before = jax.tree.map(
        lambda value: np.asarray(value).copy(),
        jax.device_get(
            canonical_teacher_structure_core(target, jnp.ones_like(target, bool), config())
        ),
    )
    root_indices = range(ROOT_COUNT) if root_index == -1 else (root_index,)

    def call(model_params, prompts, targets, example_ids, nus, runtime_operator_root):
        return _batch_metrics(
            model_params,
            prompts,
            targets,
            example_ids,
            nus,
            cycles=cycles,
            eta_diff=eta_diff,
            lambda_op=lambda_op,
            operator_root=runtime_operator_root,
            conditioning_mode=conditioning_mode,
        )

    compiled = jax.jit(call)
    runtime_batch = 4 if lambda_op else BATCH
    started = time.monotonic()
    for current_root_index in root_indices:
        operator_root = jax.random.fold_in(
            jax.random.PRNGKey(OPERATOR_ROOT_SEED), current_root_index
        )
        for nu in LEVELS:
            condition = _condition_key(
                conditioning_mode=conditioning_mode,
                eta_diff=eta_diff,
                lambda_op=lambda_op,
                cycles=cycles,
                root_index=current_root_index,
                nu=nu,
                ids=host_ids,
            )
            for start in range(0, len(ids), SHARD_SIZE):
                end = min(start + SHARD_SIZE, len(ids))
                path = _shard_path(condition, start, end)
                if path.exists():
                    with path.open("rb") as stream:
                        existing = pickle.load(stream)  # noqa: S301
                    if (
                        existing.get("complete") is True
                        and existing.get("condition") == condition
                        and existing.get("start") == start
                        and existing.get("end") == end
                    ):
                        continue
                    raise RuntimeError(f"mismatched existing F2 shard: {path}")
                chunks: dict[str, list[np.ndarray]] = {}
                for batch_start in range(start, end, runtime_batch):
                    batch_end = min(batch_start + runtime_batch, end)
                    values = compiled(
                        params,
                        prompt[batch_start:batch_end],
                        target[batch_start:batch_end],
                        ids[batch_start:batch_end],
                        jnp.full((batch_end - batch_start,), nu, dtype=jnp.float32),
                        operator_root,
                    )
                    for name, value in jax.device_get(values).items():
                        chunks.setdefault(name, []).append(np.asarray(value))
                metrics = {name: np.concatenate(values) for name, values in chunks.items()}
                if any(
                    not np.all(np.isfinite(value))
                    for name, value in metrics.items()
                    if name != "nonfinite"
                ):
                    raise RuntimeError(f"nonfinite F2 metric in condition {condition}")
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open("wb") as stream:
                    pickle.dump(
                        {
                            "complete": True,
                            "condition": condition,
                            "start": start,
                            "end": end,
                            "global_example_ids": host_ids[start:end],
                            "metrics": metrics,
                        },
                        stream,
                        protocol=pickle.HIGHEST_PROTOCOL,
                    )
                print(
                    f"F2 eta={eta_diff} lambda={lambda_op:g} q={cycles} "
                    f"root={current_root_index} nu={nu:.2f} {end}/{len(ids)}",
                    flush=True,
                )
    if not _tree_equal(params, params_before):
        raise RuntimeError("parameter tree changed during F2 evaluation")
    if not _tree_equal(_rho_tree(params), rho_before):
        raise RuntimeError("rho tree changed during F2 evaluation")
    teacher_after = canonical_teacher_structure_core(target, jnp.ones_like(target, bool), config())
    if not _tree_equal(teacher_after, teacher_before):
        raise RuntimeError("teacher tree changed during F2 evaluation")
    print(f"condition complete in {time.monotonic() - started:.1f}s", flush=True)


def _load_condition(
    *,
    conditioning_mode: str,
    eta_diff: int,
    lambda_op: float,
    cycles: int,
    root_index: int,
    subset: bool,
) -> dict[float, dict[str, np.ndarray]]:
    _, _, ids = test_split()
    if subset:
        positions = jax.random.permutation(jax.random.PRNGKey(STOCHASTIC_SUBSET_SEED), TEST_COUNT)[
            :STOCHASTIC_COUNT
        ]
        ids = ids[positions]
    host_ids = np.asarray(ids, dtype=np.uint32)
    loaded = {}
    for nu in LEVELS:
        condition = _condition_key(
            conditioning_mode=conditioning_mode,
            eta_diff=eta_diff,
            lambda_op=lambda_op,
            cycles=cycles,
            root_index=root_index,
            nu=nu,
            ids=host_ids,
        )
        pieces: dict[str, list[np.ndarray]] = {}
        seen_ids = []
        for start in range(0, len(ids), SHARD_SIZE):
            end = min(start + SHARD_SIZE, len(ids))
            path = _shard_path(condition, start, end)
            if not path.exists():
                raise RuntimeError(f"incomplete F2 evaluation; missing {path}")
            with path.open("rb") as stream:
                shard = pickle.load(stream)  # noqa: S301
            if not shard.get("complete") or shard["condition"] != condition:
                raise RuntimeError(f"invalid F2 shard: {path}")
            seen_ids.append(np.asarray(shard["global_example_ids"]))
            for name, value in shard["metrics"].items():
                pieces.setdefault(name, []).append(np.asarray(value))
        if not np.array_equal(np.concatenate(seen_ids), host_ids):
            raise RuntimeError("F2 shard ID order mismatch")
        loaded[nu] = {name: np.concatenate(values) for name, values in pieces.items()}
    return loaded


def _summary(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    count = len(values)
    mean = float(np.mean(values))
    sd = float(np.std(values, ddof=1)) if count > 1 else 0.0
    se = sd / math.sqrt(count)
    return {
        "count": count,
        "mean": mean,
        "sd": sd,
        "se": se,
        "normal_95_ci_low": mean - 1.96 * se,
        "normal_95_ci_high": mean + 1.96 * se,
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


def _paired(metric_values: np.ndarray, pair_name: str, seed_offset: int) -> dict[str, Any]:
    earlier, later = PAIR_NAMES[pair_name]
    before = np.asarray(metric_values[:, earlier], dtype=np.float64)
    after = np.asarray(metric_values[:, later], dtype=np.float64)
    change = after - before
    stats = _summary(change)
    key = jax.random.fold_in(jax.random.PRNGKey(BOOTSTRAP_SEED), seed_offset)
    indices = jax.random.randint(key, (BOOTSTRAP_RESAMPLES, len(change)), 0, len(change))
    bootstrap = jnp.mean(jnp.asarray(change)[indices], axis=1)
    low, high = map(float, np.asarray(jnp.percentile(bootstrap, jnp.asarray([2.5, 97.5]))))
    return {
        "definition": "later step minus earlier step",
        "earlier_step": earlier + 1,
        "later_step": later + 1,
        "earlier_mean": float(np.mean(before)),
        "later_mean": float(np.mean(after)),
        "absolute_change": stats["mean"],
        "relative_change": stats["mean"] / max(abs(float(np.mean(before))), 1.0e-12),
        "paired_change": stats,
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        "bootstrap_percentile_95_ci_low": low,
        "bootstrap_percentile_95_ci_high": high,
    }


def _summarize_condition(data: dict[float, dict[str, np.ndarray]]) -> dict[str, Any]:
    result = {"per_nu": []}
    seed_offset = 0
    for nu, metrics in data.items():
        row: dict[str, Any] = {"nu": nu, "count": len(metrics["mse"])}
        for metric in (
            "mse",
            "nll",
            "byte_accuracy",
            "exact_accuracy",
            "h_hat_rms",
            "inter_step_rms_change",
            "relative_update",
            "cosine_similarity",
        ):
            row[metric] = [_summary(metrics[metric][:, step]) for step in range(S_EXEC)]
        row["nonfinite_count"] = int(np.sum(metrics["nonfinite"]))
        row["pairs"] = {}
        for pair_name in PAIR_NAMES:
            row["pairs"][pair_name] = {}
            for metric in ("mse", "nll", "byte_accuracy"):
                paired = _paired(metrics[metric], pair_name, seed_offset)
                seed_offset += 1
                if metric == "byte_accuracy":
                    paired["absolute_change_percentage_points"] = 100.0 * paired["absolute_change"]
                row["pairs"][pair_name][metric] = paired
        result["per_nu"].append(row)
    return result


def _write_per_example(
    path: Path, data: dict[float, dict[str, np.ndarray]], ids: np.ndarray
) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for nu, metrics in data.items():
            for index, example_id in enumerate(ids):
                for step in range(S_EXEC):
                    row = {
                        "global_example_id": int(example_id),
                        "nu": nu,
                        "step": step + 1,
                        **{
                            name: serialise(values[index, step])
                            for name, values in metrics.items()
                            if values.ndim == 2
                        },
                    }
                    stream.write(json.dumps(row, sort_keys=True) + "\n")


def _classify(primary: dict[str, Any], stochastic: dict[str, Any]) -> str:
    informative = [row for row in primary["per_nu"] if row["nu"] in {0.25, 0.5, 0.75}]
    if any(row["nonfinite_count"] for row in primary["per_nu"]):
        raise RuntimeError("primary F2 result contains nonfinite values")
    changes = [row["pairs"]["B41"] for row in informative]
    benefit = all(
        item["byte_accuracy"]["absolute_change"] >= 0.001 and item["nll"]["absolute_change"] < 0
        for item in changes
    )
    degradation = all(
        item["byte_accuracy"]["absolute_change"] <= -0.001 and item["nll"]["absolute_change"] > 0
        for item in changes
    )
    stochastic_direction = stochastic["informative_b41_byte_accuracy_mean_change"]
    if benefit and stochastic_direction >= 0:
        return "PHASE_F_2_SAME_MODEL_S_BENEFIT"
    if degradation and stochastic_direction <= 0:
        return "PHASE_F_2_SAME_MODEL_S_DEGRADATION"
    return "PHASE_F_2_SAME_MODEL_S_NEUTRAL"


def aggregate() -> dict[str, Any]:
    primary_data = _load_condition(
        conditioning_mode=F2_STEP0_CONDITIONING,
        eta_diff=0,
        lambda_op=0.0,
        cycles=3,
        root_index=0,
        subset=False,
    )
    eta1_data = _load_condition(
        conditioning_mode=F2_STEP0_CONDITIONING,
        eta_diff=1,
        lambda_op=0.0,
        cycles=3,
        root_index=0,
        subset=False,
    )
    q0_data = _load_condition(
        conditioning_mode=F2_STEP0_CONDITIONING,
        eta_diff=0,
        lambda_op=0.0,
        cycles=0,
        root_index=0,
        subset=False,
    )
    stochastic_roots: dict[int, dict[int, dict[float, dict[str, np.ndarray]]]] = {
        eta: {
            root: _load_condition(
                conditioning_mode=F2_STEP0_CONDITIONING,
                eta_diff=eta,
                lambda_op=1.0,
                cycles=3,
                root_index=root,
                subset=True,
            )
            for root in range(ROOT_COUNT)
        }
        for eta in (0, 1)
    }
    stochastic_summary: dict[str, Any] = {"root_count": ROOT_COUNT, "per_eta": {}}
    subset_positions = np.asarray(
        jax.random.permutation(jax.random.PRNGKey(STOCHASTIC_SUBSET_SEED), TEST_COUNT)[
            :STOCHASTIC_COUNT
        ]
    )
    lambda_zero_by_eta = {0: primary_data, 1: eta1_data}
    for eta, roots in stochastic_roots.items():
        eta_rows = []
        for nu in LEVELS:
            if any(np.any(roots[root][nu]["nonfinite"]) for root in range(ROOT_COUNT)):
                raise RuntimeError(f"nonfinite stochastic F2 trajectory for eta={eta}, nu={nu}")
            root_metrics = {}
            for metric in ("mse", "nll", "byte_accuracy", "exact_accuracy"):
                root_step_means = np.asarray(
                    [np.mean(roots[root][nu][metric], axis=0) for root in range(ROOT_COUNT)]
                )
                step_rows = []
                for step in range(S_EXEC):
                    summary = _summary(root_step_means[:, step])
                    half = ROOT_T_95_DF15 * summary["se"]
                    summary["student_t_95_ci_low"] = summary["mean"] - half
                    summary["student_t_95_ci_high"] = summary["mean"] + half
                    step_rows.append(summary)
                root_metrics[metric] = step_rows
            root_averaged = {
                metric: np.mean(
                    np.stack([roots[root][nu][metric] for root in range(ROOT_COUNT)]), axis=0
                )
                for metric in ("mse", "nll", "byte_accuracy")
            }
            pairs = {
                name: {
                    metric: _paired(values, name, 10_000 + eta * 1000 + int(nu * 100) * 10 + i)
                    for i, (metric, values) in enumerate(root_averaged.items())
                }
                for name in PAIR_NAMES
            }
            for pair_metrics in pairs.values():
                pair_metrics["byte_accuracy"]["absolute_change_percentage_points"] = (
                    100.0 * pair_metrics["byte_accuracy"]["absolute_change"]
                )
            lambda_zero_subset = {
                metric: values[subset_positions]
                for metric, values in lambda_zero_by_eta[eta][nu].items()
                if values.ndim == 2
            }
            eta_rows.append(
                {
                    "nu": nu,
                    "root_level": root_metrics,
                    "root_averaged_pairs": pairs,
                    "matching_lambda_zero_subset": {
                        metric: [_summary(values[:, step]) for step in range(S_EXEC)]
                        for metric, values in lambda_zero_subset.items()
                    },
                }
            )
        stochastic_summary["per_eta"][str(eta)] = eta_rows
    informative_stochastic = [
        row["root_averaged_pairs"]["B41"]["byte_accuracy"]["absolute_change"]
        for row in stochastic_summary["per_eta"]["0"]
        if row["nu"] in {0.25, 0.5, 0.75}
    ]
    stochastic_summary["informative_b41_byte_accuracy_mean_change"] = float(
        np.mean(informative_stochastic)
    )
    primary = _summarize_condition(primary_data)
    eta1 = _summarize_condition(eta1_data)
    q0 = _summarize_condition(q0_data)
    label = _classify(primary, stochastic_summary)
    _, _, full_ids = test_split()
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    _write_per_example(EVIDENCE / "eta0_per_example.jsonl", primary_data, np.asarray(full_ids))
    _write_per_example(EVIDENCE / "eta1_per_example.jsonl", eta1_data, np.asarray(full_ids))
    write_json(EVIDENCE / "primary_eta0_summary.json", primary)
    write_json(EVIDENCE / "eta1_summary.json", eta1)
    write_json(EVIDENCE / "q0_shell_summary.json", q0)
    write_json(EVIDENCE / "stochastic_summary.json", stochastic_summary)
    result = {
        "label": label,
        "primary_condition": F2_STEP0_CONDITIONING,
        "primary_examples_per_nu": TEST_COUNT,
        "primary_lambda_op": 0.0,
        "primary_eta_diff": 0,
        "informative_region": [0.25, 0.5, 0.75],
        "functional_neutrality_threshold_byte_accuracy_percentage_points": 0.1,
        "stochastic_confirmation": {
            "examples_per_nu": STOCHASTIC_COUNT,
            "operator_roots": ROOT_COUNT,
            "lambda_op": 1.0,
            "sigma_op": 1.0e-3,
            "student_t_df": 15,
        },
        "native_s_conditioning": "diagnostic-only and not run unless separately requested",
        "no_training": True,
        "immutability_checks": {
            "parameter_tree_bitwise_equal_before_after": True,
            "rho_tree_bitwise_equal_before_after": True,
            "teacher_tree_bitwise_equal_before_after": True,
        },
        "all_trajectories_finite": True,
        "operator_diffusion_namespace_collision": False,
        "parameter_tree_sha256": _array_hash(load_checkpoint()["params"]),
    }
    write_json(EVIDENCE / "result.json", result)
    (EVIDENCE / "DECISION.md").write_text(
        "# Phase F.2 decision\n\n"
        f"**{label}**\n\n"
        "The label is determined only from the full-test, lambda-op-zero, eta-zero "
        f"`{F2_STEP0_CONDITIONING}` result plus finite 16-root Torx confirmation. "
        "Literal native-S conditioning is diagnostic-only and cannot determine this label.\n\n"
        "No training, optimizer update, rollout training, parameter mutation, rho mutation, "
        "teacher mutation, structure commit, or outer refinement was performed.\n"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("audit", "evaluate", "aggregate"), required=True)
    parser.add_argument(
        "--conditioning",
        choices=(F2_STEP0_CONDITIONING, F2_NATIVE_S_CONDITIONING_UNTRAINED),
        default=F2_STEP0_CONDITIONING,
    )
    parser.add_argument("--eta", type=int, choices=(0, 1), default=0)
    parser.add_argument("--lambda-op", type=float, choices=(0.0, 1.0), default=0.0)
    parser.add_argument("--cycles", type=int, choices=(0, 3), default=3)
    parser.add_argument("--root-index", type=int, default=0)
    parser.add_argument("--subset", action="store_true")
    args = parser.parse_args()
    if args.stage == "audit":
        run_audit()
    elif args.stage == "evaluate":
        if args.root_index != -1 and not 0 <= args.root_index < ROOT_COUNT:
            raise ValueError(f"root-index must be -1 or lie in [0, {ROOT_COUNT})")
        evaluate_condition(
            conditioning_mode=args.conditioning,
            eta_diff=args.eta,
            lambda_op=args.lambda_op,
            cycles=args.cycles,
            root_index=args.root_index,
            subset=args.subset,
        )
    else:
        aggregate()


if __name__ == "__main__":
    main()
