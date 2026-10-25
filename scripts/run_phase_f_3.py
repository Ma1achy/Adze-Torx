"""Phase F.3 self-generated S=2 rollout training and evaluation."""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import pickle
import subprocess
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from adze_t.backends.torx import TorxOperatorConfig, TorxOps
from adze_t.config import REFERENCE_SMALL_V0
from adze_t.corruption import PHASE_F_EVAL_GRID, corrupt_h
from adze_t.denoise import (
    F2_STEP0_CONDITIONING,
    F3_NATIVE_S_CONDITIONING_TRAINED,
    apply_denoising_trajectory,
    make_sanitized_target_analysis,
)
from adze_t.model import apply_clean_target_teacher
from adze_t.phase_f_1 import (
    DENOISE_V1,
    dataset_audit,
    denoise_example_hashes,
    generate_denoise_v0,
    initial_diffusion_epsilon,
)
from adze_t.phase_f_3 import (
    INFORMATIVE_LEVELS,
    PHASE_F_3_BLOCKED,
    ROOT_T_95_DF15,
    bucket_effect,
    classify,
    effect_pattern,
    is_pareto_best_5k,
    paired_summary,
    require_exact_protocol,
    should_extend_to_10k,
    step0_forgetting,
    summary,
)
from adze_t.training import (
    _CODEC_ENCODER_NAMES,
    _gradient_metrics,
    _masked_gradients,
    _norm,
    _path_norm,
    s2_rollout_objective,
    stochastic_model_update_mask,
    stochastic_s2_rollout_train_step,
)


ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT / "results/runs/phase_f/f1/denoise_v1/one_step/init0_stoch0.pkl"
CODEC = ROOT / "results/phase_b/checkpoints/target_codec_b1.pkl"
WORK = ROOT / "results/runs/phase_f/f3/s2/init0_stoch0"
EVIDENCE = ROOT / "results/phase_f/f3"
EXPECTED_PARENT_SHA256 = "520d3723d3a3187aa08ff3f467f434d43b579cdc4b0ce667727dfad96e83cf57"
EXPECTED_CODEC_SHA256 = "55f057f78f795f1585d7aac28fa7ff37846d4c1098507ede375e86b21c4f8ff6"
EXPECTED_BASE_GIT_SHA = "d518cfff24366b0e44298a81ee1e7936c8e5aab4"
TORX_PIN = "f1fc858ed950ecd41935d15c06d0ec7c5e0674ae"
F1_RAW_GRADIENT_NORM = 1.4641897472e11
DATA_SEEDS = {"train": 940, "validation": 941, "test": 942}
TRAIN_COUNT = 16_384
VALIDATION_COUNT = 512
TEST_COUNT = 4_096
STOCHASTIC_SUBSET_SEED = 10_421
BOOTSTRAP_SEED = 10_422
OPERATOR_ROOT_SEED = 10_423
TEST_DIFFUSION_ROOT_SEED = 10_424
TRAIN_OPERATOR_ROOT = jax.random.PRNGKey(10_400)
TRAIN_DIFFUSION_ROOT = jax.random.PRNGKey(10_401)
VALIDATION_DIFFUSION_ROOT = jax.random.PRNGKey(10_402)
NU_ROOT = jax.random.PRNGKey(10_403)
LEVELS = tuple(PHASE_F_EVAL_GRID)
CHECKPOINTS = (100, 250, 500, 1_000, 2_000, 5_000, 10_000)
BATCH = 32
ROOT_COUNT = 16
PARENT_OPTIMIZER_STEP = 5_000
EVALUATION_PROTOCOL = "PHASE_F_3_EVALUATION_PROTOCOL_V0"


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
    if getattr(value, "ndim", 0) == 0:
        return bool(value) if np.issubdtype(np.asarray(value).dtype, np.bool_) else float(value)
    return value.tolist()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=serialise) + "\n")


def append_jsonl(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, sort_keys=True, default=serialise) + "\n")


def load_pickle(path: Path) -> Any:
    with path.open("rb") as stream:
        return pickle.load(stream)  # noqa: S301


def save_pickle(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as stream:
        pickle.dump(jax.device_get(value), stream, protocol=pickle.HIGHEST_PROTOCOL)


def array_hash(value: Any) -> str:
    digest = hashlib.sha256()
    for path, leaf in jax.tree_util.tree_leaves_with_path(jax.device_get(value)):
        array = np.asarray(leaf)
        digest.update(jax.tree_util.keystr(path).encode())
        digest.update(str(array.dtype).encode())
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


def json_hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), default=serialise).encode()
    return hashlib.sha256(encoded).hexdigest()


def ids_hash(ids: Any) -> str:
    return hashlib.sha256(np.asarray(ids, dtype=np.uint32).tobytes()).hexdigest()


def implementation_source_hash() -> str:
    digest = hashlib.sha256()
    for path in (
        ROOT / "src/adze_t/denoise.py",
        ROOT / "src/adze_t/training.py",
        ROOT / "src/adze_t/phase_f_3.py",
        ROOT / "scripts/run_phase_f_3.py",
    ):
        digest.update(str(path.relative_to(ROOT)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def checkpoint_manifest(path: Path, state: dict[str, Any]) -> dict[str, Any]:
    f3_updates = int(state.get("f3_rollout_updates", 0))
    absolute_step = int(state["step"])
    if absolute_step != PARENT_OPTIMIZER_STEP + f3_updates:
        raise RuntimeError(f"invalid checkpoint step coordinates: {path}")
    if f3_updates and state.get("parent_sha256") != EXPECTED_PARENT_SHA256:
        raise RuntimeError(f"invalid F3 parent provenance in checkpoint: {path}")
    return {
        "path": str(path.relative_to(ROOT)),
        "file_sha256": sha256(path),
        "parameter_tree_sha256": array_hash(state["params"]),
        "moment_tree_sha256": array_hash(state["moments"]),
        "f3_rollout_updates": f3_updates,
        "optimizer_absolute_step": absolute_step,
        "parent_sha256": EXPECTED_PARENT_SHA256,
    }


def evaluation_protocol(
    checkpoint: dict[str, Any],
    ids: Any,
    *,
    split_name: str,
    conditioning_mode: str,
    eta_diff: int,
    lambda_op: float,
    diffusion_root_seed: int,
    operator_root_seed: int,
    operator_root_index: int | None,
    baseline_checkpoint: dict[str, Any] | None = None,
) -> dict[str, Any]:
    protocol = {
        "contract": EVALUATION_PROTOCOL,
        "implementation_git_sha": git_sha(),
        "implementation_source_sha256": implementation_source_hash(),
        "torx_pin": TORX_PIN,
        "checkpoint": checkpoint,
        "dataset": {
            "split": split_name,
            "seed": DATA_SEEDS[split_name],
            "count": len(ids),
            "ordered_global_ids_sha256": ids_hash(ids),
        },
        "trajectory": {
            "s_exec": 2,
            "conditioning_mode": conditioning_mode,
            "actual_s_indices": [0, 1],
            "eta_diff": eta_diff,
            "cycles_Q": 3,
            "physical_blocks_L": 4,
            "outer_refinement_R": 0,
            "levels": list(LEVELS),
        },
        "operator": {
            "lambda_op": lambda_op,
            "sigma_op": 1.0e-3 if lambda_op else 0.0,
            "root_seed": operator_root_seed,
            "root_index": operator_root_index,
        },
        "diffusion_root_seed": diffusion_root_seed,
    }
    if baseline_checkpoint is not None:
        protocol["baseline_checkpoint"] = baseline_checkpoint
    return {**protocol, "protocol_sha256": json_hash(protocol)}


def split(name: str, count: int):
    return generate_denoise_v0(count, DATA_SEEDS[name], spec=DENOISE_V1)


def _ordered_example_hashes(prompt: jax.Array, target: jax.Array) -> list[str]:
    prompt_host, target_host = map(np.asarray, jax.device_get((prompt, target)))
    hashes = []
    for sample_prompt, sample_target in zip(prompt_host, target_host, strict=True):
        digest = hashlib.sha256()
        digest.update(sample_prompt.tobytes())
        digest.update(sample_target.tobytes())
        hashes.append(digest.hexdigest())
    return hashes


def load_parent() -> dict[str, Any]:
    if sha256(PARENT) != EXPECTED_PARENT_SHA256:
        raise RuntimeError("F3 parent checkpoint provenance mismatch")
    if sha256(CODEC) != EXPECTED_CODEC_SHA256:
        raise RuntimeError("F3 codec provenance mismatch")
    state = load_pickle(PARENT)
    if set(state) != {"moments", "params", "step"} or int(state["step"]) != 5_000:
        raise RuntimeError("F3 requires the accepted F1 optimizer state at absolute step 5000")
    return state


def run_audit() -> dict[str, Any]:
    state = load_parent()
    datasets = {}
    split_hash_sets = {}
    for name, count in (("train", TRAIN_COUNT), ("validation", TEST_COUNT), ("test", TEST_COUNT)):
        prompt, target, ids = split(name, count)
        hashes = denoise_example_hashes(prompt, target)
        ordered_hashes = _ordered_example_hashes(prompt, target)
        split_hash_sets[name] = hashes
        datasets[name] = {
            **dataset_audit(prompt, target, spec=DENOISE_V1),
            "count": count,
            "global_ids_preserved": bool(jnp.array_equal(ids, jnp.arange(count))),
            "per_example_sha256": ordered_hashes,
            "ordered_split_sha256": hashlib.sha256("".join(ordered_hashes).encode()).hexdigest(),
        }
    cfg = config()
    payload = {
        "status": "F3_PROVENANCE_PASS",
        "base_git_sha": EXPECTED_BASE_GIT_SHA,
        "implementation_git_sha": git_sha(),
        "implementation_source_sha256": implementation_source_hash(),
        "torx_pin": TORX_PIN,
        "parent": str(PARENT.relative_to(ROOT)),
        "parent_sha256": sha256(PARENT),
        "parent_optimizer_absolute_step": int(state["step"]),
        "parent_parameter_tree_sha256": array_hash(state["params"]),
        "parent_moment_tree_sha256": array_hash(state["moments"]),
        "f_s2_parent": "F_ONE_STEP_STEP5000",
        "codec_sha256": sha256(CODEC),
        "architecture": {"L": cfg.model.physical_blocks_L, "Q": cfg.model.cycles_Q, "R": 0},
        "trajectory": {
            "s_train": 2,
            "eta_diff": 0,
            "actual_s_indices": [0, 1],
            "denoise_condition_indices": [0, 1],
            "step1_input": "alpha(0.75 * nu0) * h_hat_0",
            "pathwise_gradient": True,
            "teacher_forcing": False,
        },
        "optimizer": {
            "name": "AdamW",
            "learning_rate": cfg.training.learning_rate,
            "weight_decay": cfg.training.weight_decay,
            "grad_clip": cfg.training.grad_clip_norm,
            "batch_size": BATCH,
            "continue_moments": True,
        },
        "datasets": datasets,
        "split_intersections": {
            "train_validation": len(split_hash_sets["train"] & split_hash_sets["validation"]),
            "train_test": len(split_hash_sets["train"] & split_hash_sets["test"]),
            "validation_test": len(split_hash_sets["validation"] & split_hash_sets["test"]),
        },
        "frozen": {
            "rho": True,
            "target_teacher": True,
            "structure": True,
            "proposal_weight": 0.0,
        },
    }
    write_json(EVIDENCE / "provenance.json", payload)
    return payload


def training_batch(prompt, target, ids, absolute_step: int):
    indices = jnp.arange((absolute_step - 1) * BATCH, absolute_step * BATCH) % TRAIN_COUNT
    nu = jax.random.uniform(
        jax.random.fold_in(NU_ROOT, absolute_step),
        (BATCH,),
        minval=0.025,
        maxval=0.9,
    )
    return {
        "prompt": prompt[indices],
        "prompt_mask": jnp.ones_like(prompt[indices], dtype=bool),
        "target": target[indices],
        "target_mask": jnp.ones_like(target[indices], dtype=bool),
        "nu": nu,
        "global_example_id": ids[indices],
        "diffusion_occurrence": jnp.asarray(absolute_step, dtype=jnp.uint32),
    }


def _tree_dot(left: Any, right: Any) -> jax.Array:
    products = [jnp.sum(a * b) for a, b in zip(jax.tree.leaves(left), jax.tree.leaves(right))]
    return jnp.sum(jnp.stack(products))


def _tree_cosine(left: Any, right: Any) -> jax.Array:
    return _tree_dot(left, right) / jnp.maximum(_norm(left) * _norm(right), 1.0e-30)


def _select_paths(tree: Any, predicate: Any) -> Any:
    return jax.tree_util.tree_map_with_path(
        lambda path, leaf: jnp.where(predicate(jax.tree_util.keystr(path)), leaf, 0), tree
    )


def _gradient_report(grads: Any) -> dict[str, Any]:
    mask = stochastic_model_update_mask(grads)
    permitted = _masked_gradients(grads, mask)
    direct_names = ("a_log", "d_skip", "delta_bias", "layer_scale")
    return {
        "raw_global_norm": _norm(grads),
        "permitted_global_norm": _norm(permitted),
        "clipped_norm": jnp.minimum(_norm(permitted), config().training.grad_clip_norm),
        "frontend_context_norm": _path_norm(
            grads,
            lambda path: (
                "['encoder']['byte_embed']" in path
                or "['encoder']['frontend']" in path
                or "['encoder']['context" in path
            ),
        ),
        "carrier_input_transform_norm": _path_norm(
            grads,
            lambda path: any(
                f"['{name}']" in path for name in ("carrier_in", "carrier_out", "carrier_prior")
            ),
        ),
        "dit_norm": _path_norm(grads, lambda path: "['dit']" in path),
        "conditioning_norm": _gradient_metrics(grads)["grad_conditioning"],
        "h_head_norm": _path_norm(grads, lambda path: "['h_head']" in path),
        "decoder_norm": _path_norm(grads, lambda path: "['decoder']" in path),
        "direct_ssm_norm": _path_norm(
            grads, lambda path: any(f"['{name}']" in path for name in direct_names)
        ),
        "rho_raw_norm": _path_norm(grads, lambda path: "['rho']" in path),
        "rho_applied_norm": 0.0,
    }


def run_first_gradient() -> dict[str, Any]:
    state = load_parent()
    parent_manifest = checkpoint_manifest(PARENT, state)
    prompt, target, ids = split("train", TRAIN_COUNT)
    batch = training_batch(prompt, target, ids, 5_001)

    def component_objective(params, component, *, detach_between_steps=False):
        _, auxiliary = s2_rollout_objective(
            params,
            batch,
            TRAIN_OPERATOR_ROOT,
            TRAIN_DIFFUSION_ROOT,
            5_001,
            config=config(),
            detach_between_steps=detach_between_steps,
        )
        return auxiliary[component]["loss"]

    g0 = jax.grad(lambda params: component_objective(params, "step0"))(state["params"])
    g1 = jax.grad(lambda params: component_objective(params, "step1"))(state["params"])
    g1_detached = jax.grad(
        lambda params: component_objective(params, "step1", detach_between_steps=True)
    )(state["params"])
    cross_boundary = jax.tree.map(lambda connected, detached: connected - detached, g1, g1_detached)
    cross_boundary_norm = _norm(cross_boundary)
    combined_loss, g = jax.value_and_grad(
        lambda params: s2_rollout_objective(
            params,
            batch,
            TRAIN_OPERATOR_ROOT,
            TRAIN_DIFFUSION_ROOT,
            5_001,
            config=config(),
        )[0]
    )(state["params"])
    expected = jax.tree.map(lambda a, b: 0.5 * (a + b), g0, g1)
    difference = jax.tree.map(lambda actual, wanted: actual - wanted, g, expected)
    groups = {
        "global": lambda _: True,
        "dit": lambda path: "['dit']" in path,
        "h_head": lambda path: "['h_head']" in path,
        "decoder": lambda path: "['decoder']" in path,
    }
    cosines = {}
    for name, predicate in groups.items():
        left0 = _select_paths(g0, predicate)
        left1 = _select_paths(g1, predicate)
        combined = _select_paths(g, predicate)
        cosines[name] = {
            "cosine_g0_g1": _tree_cosine(left0, left1),
            "cosine_g0_g": _tree_cosine(left0, combined),
            "cosine_g1_g": _tree_cosine(left1, combined),
        }
    reports = {
        "step0": _gradient_report(g0),
        "step1": _gradient_report(g1),
        "combined": _gradient_report(g),
    }
    combined_norm = float(reports["combined"]["permitted_global_norm"])
    denominator = 0.5 * (
        float(reports["step0"]["permitted_global_norm"])
        + float(reports["step1"]["permitted_global_norm"])
    )
    cancellation_ratio = combined_norm / max(denominator, 1.0e-30)
    finite = all(
        bool(jnp.all(jnp.isfinite(leaf)))
        for leaf in jax.tree.leaves((g0, g1, g1_detached, cross_boundary, g))
    )
    pathological = (
        (not finite)
        or (combined_norm >= 100.0 * F1_RAW_GRADIENT_NORM and cancellation_ratio <= 1.0e-3)
        or float(cross_boundary_norm) <= 0.0
    )

    def toy_step1_loss(step0_weight):
        h_hat_0 = step0_weight * jnp.asarray(1.25)
        h_1_input = jnp.asarray(0.7) * h_hat_0
        h_hat_1 = jax.lax.stop_gradient(jnp.asarray(1.1)) * h_1_input
        return (h_hat_1 - jnp.asarray(0.2)) ** 2

    toy_cross_boundary_gradient = jax.grad(toy_step1_loss)(jnp.asarray(0.9))
    payload = {
        "status": "F3_GRADIENT_PATHOLOGY" if pathological else "F3_FIRST_GRADIENT_PASS",
        "checkpoint": parent_manifest,
        "implementation_source_sha256": implementation_source_hash(),
        "optimizer_absolute_step": 5_001,
        "f3_rollout_updates": 1,
        "combined_loss": combined_loss,
        "gradients": reports,
        "full_model_cross_boundary": {
            "connected_step1_gradient_norm": _norm(g1),
            "detached_step1_gradient_norm": _norm(g1_detached),
            "connected_minus_detached_norm": cross_boundary_norm,
            "identical_operator_and_diffusion_keys": True,
            "passed": bool(jnp.isfinite(cross_boundary_norm) & (cross_boundary_norm > 0)),
        },
        "cosines": cosines,
        "combined_equals_half_sum_error_norm": _norm(difference),
        "cancellation_ratio": cancellation_ratio,
        "f1_raw_gradient_reference": F1_RAW_GRADIENT_NORM,
        "finite": finite,
        "pathology": pathological,
        "toy_cross_boundary_oracle": {
            "formula": "d/dw0 (stop_gradient(w1) * alpha1 * w0 * x - target)^2",
            "gradient": toy_cross_boundary_gradient,
            "nonzero": bool(jnp.abs(toy_cross_boundary_gradient) > 0),
        },
    }
    write_json(EVIDENCE / "first_gradient.json", payload)
    write_json(
        EVIDENCE / "gradient_conflict.json",
        {"cosines": cosines, "cancellation_ratio": cancellation_ratio},
    )
    if pathological:
        raise RuntimeError("F3_GRADIENT_PATHOLOGY")
    return payload


def _tree_is_finite(tree: Any) -> bool:
    return all(bool(jnp.all(jnp.isfinite(leaf))) for leaf in jax.tree.leaves(tree))


def _prospective_update_audit(
    before_params: Any,
    after_params: Any,
    before_moments: Any,
    after_moments: Any,
) -> dict[str, Any]:
    mask = stochastic_model_update_mask(before_params)
    before_paths = {
        jax.tree_util.keystr(path): np.asarray(value)
        for path, value in jax.tree.leaves_with_path(jax.device_get(before_params))
    }
    after_paths = {
        jax.tree_util.keystr(path): np.asarray(value)
        for path, value in jax.tree.leaves_with_path(jax.device_get(after_params))
    }
    mask_paths = {
        jax.tree_util.keystr(path): bool(np.asarray(value))
        for path, value in jax.tree.leaves_with_path(jax.device_get(mask))
    }
    before_moment_paths = {
        jax.tree_util.keystr(path): np.asarray(value)
        for path, value in jax.tree.leaves_with_path(jax.device_get(before_moments))
    }
    after_moment_paths = {
        jax.tree_util.keystr(path): np.asarray(value)
        for path, value in jax.tree.leaves_with_path(jax.device_get(after_moments))
    }
    changed_enabled_params = []
    changed_disabled_params = []
    changed_enabled_moments = []
    changed_disabled_moments = []
    for path, before in before_paths.items():
        enabled = mask_paths[path]
        changed = not np.array_equal(before, after_paths[path])
        if changed:
            (changed_enabled_params if enabled else changed_disabled_params).append(path)
        for moment_index in (0, 1):
            moment_path = f"[{moment_index}]{path}"
            moment_changed = not np.array_equal(
                before_moment_paths[moment_path], after_moment_paths[moment_path]
            )
            if moment_changed:
                (changed_enabled_moments if enabled else changed_disabled_moments).append(
                    moment_path
                )
    passed = bool(
        changed_enabled_params
        and changed_enabled_moments
        and not changed_disabled_params
        and not changed_disabled_moments
        and _tree_is_finite(after_params)
        and _tree_is_finite(after_moments)
    )
    return {
        "passed": passed,
        "all_parameters_finite": _tree_is_finite(after_params),
        "all_moments_finite": _tree_is_finite(after_moments),
        "changed_enabled_parameter_count": len(changed_enabled_params),
        "changed_enabled_moment_count": len(changed_enabled_moments),
        "changed_disabled_parameters": changed_disabled_params,
        "changed_disabled_moments": changed_disabled_moments,
        "changed_enabled_parameter_examples": changed_enabled_params[:20],
    }


def _require_gradient_gate(parent_manifest: dict[str, Any]) -> dict[str, Any]:
    path = EVIDENCE / "first_gradient.json"
    if not path.exists():
        raise RuntimeError("run the F3 first-gradient gate before an optimizer update")
    evidence = json.loads(path.read_text())
    if (
        evidence.get("checkpoint") != parent_manifest
        or evidence.get("implementation_source_sha256") != implementation_source_hash()
        or evidence.get("pathology")
    ):
        raise RuntimeError("stale, mixed, or failed F3 first-gradient evidence")
    if not evidence.get("full_model_cross_boundary", {}).get("passed", False):
        raise RuntimeError("F3 full-model cross-boundary gradient gate did not pass")
    return evidence


def run_trial_update() -> dict[str, Any]:
    state = load_parent()
    parent_manifest = checkpoint_manifest(PARENT, state)
    _require_gradient_gate(parent_manifest)
    prompt, target, ids = split("train", TRAIN_COUNT)
    batch = training_batch(prompt, target, ids, 5_001)
    update = jax.jit(stochastic_s2_rollout_train_step, static_argnames=("config",))
    before_frozen = _frozen_hashes(state["params"], state["moments"])
    updated_params, updated_moments, metrics = update(
        state["params"],
        state["moments"],
        5_001,
        batch,
        TRAIN_OPERATOR_ROOT,
        TRAIN_DIFFUSION_ROOT,
        config=config(),
    )
    jax.block_until_ready(metrics["loss"])
    metrics_finite = _tree_is_finite(metrics)
    changes = _prospective_update_audit(
        state["params"], updated_params, state["moments"], updated_moments
    )
    frozen_unchanged = before_frozen == _frozen_hashes(updated_params, updated_moments)
    passed = bool(metrics_finite and changes["passed"] and frozen_unchanged)
    payload = {
        "status": "F3_TRIAL_UPDATE_PASS" if passed else "F3_TRIAL_UPDATE_FAIL",
        "checkpoint": parent_manifest,
        "implementation_source_sha256": implementation_source_hash(),
        "optimizer_absolute_step": 5_001,
        "f3_rollout_updates": 1,
        "metrics": metrics,
        "metrics_finite": metrics_finite,
        "update_audit": changes,
        "frozen_teacher_rho_params_and_moments_bitwise_unchanged": frozen_unchanged,
        "persisted_optimizer_state": False,
        "passed": passed,
    }
    write_json(EVIDENCE / "trial_update.json", payload)
    if not passed:
        raise RuntimeError("F3 prospective full-model optimizer update failed")
    return payload


def _require_trial_update(parent_manifest: dict[str, Any]) -> dict[str, Any]:
    path = EVIDENCE / "trial_update.json"
    if not path.exists():
        raise RuntimeError("run the F3 prospective optimizer update before smoke training")
    evidence = json.loads(path.read_text())
    if (
        evidence.get("checkpoint") != parent_manifest
        or evidence.get("implementation_source_sha256") != implementation_source_hash()
        or not evidence.get("passed", False)
    ):
        raise RuntimeError("stale, mixed, or failed F3 prospective optimizer update evidence")
    return evidence


def run_runtime_smoke() -> dict[str, Any]:
    state = load_parent()
    parent_manifest = checkpoint_manifest(PARENT, state)
    _require_gradient_gate(parent_manifest)
    _require_trial_update(parent_manifest)
    prompt, target, ids = split("train", TRAIN_COUNT)
    update = jax.jit(stochastic_s2_rollout_train_step, static_argnames=("config",))
    params, moments = state["params"], state["moments"]
    expected_frozen = _frozen_hashes(params, moments)
    metric_rows = []
    for f3_updates in range(1, 11):
        absolute_step = PARENT_OPTIMIZER_STEP + f3_updates
        batch = training_batch(prompt, target, ids, absolute_step)
        params, moments, metrics = update(
            params,
            moments,
            absolute_step,
            batch,
            TRAIN_OPERATOR_ROOT,
            TRAIN_DIFFUSION_ROOT,
            config=config(),
        )
        jax.block_until_ready(metrics["loss"])
        if not _tree_is_finite((params, moments, metrics)):
            raise RuntimeError(f"nonfinite F3 runtime smoke at update {f3_updates}")
        if _frozen_hashes(params, moments) != expected_frozen:
            raise RuntimeError(f"frozen F3 state changed during smoke update {f3_updates}")
        metric_rows.append(
            {
                "f3_rollout_updates": f3_updates,
                "optimizer_absolute_step": absolute_step,
                "loss": metrics["loss"],
                "step0_loss": metrics["step0_loss"],
                "step1_loss": metrics["step1_loss"],
                "grad_raw_norm": metrics["grad_raw_norm"],
                "grad_clipped_applied_norm": metrics["grad_clipped_applied_norm"],
            }
        )
    changes = _prospective_update_audit(state["params"], params, state["moments"], moments)
    passed = bool(changes["passed"] and _tree_is_finite((params, moments)))
    payload = {
        "status": "F3_RUNTIME_SMOKE_PASS" if passed else "F3_RUNTIME_SMOKE_FAIL",
        "checkpoint": parent_manifest,
        "implementation_source_sha256": implementation_source_hash(),
        "updates": 10,
        "metrics": metric_rows,
        "update_audit": changes,
        "persisted_optimizer_state": False,
        "passed": passed,
    }
    write_json(EVIDENCE / "runtime_smoke.json", payload)
    if not passed:
        raise RuntimeError("F3 10-update runtime smoke failed")
    return payload


def _batch_metrics(
    params: Any,
    prompt: jax.Array,
    target: jax.Array,
    ids: jax.Array,
    nu: jax.Array,
    *,
    eta_diff: int,
    lambda_op: float,
    operator_root: jax.Array,
    diffusion_root: jax.Array,
    conditioning_mode: str,
    include_latents: bool,
) -> dict[str, jax.Array]:
    cfg = config()
    target_mask = jnp.ones_like(target, dtype=bool)
    clean_ops = TorxOps.create(operator_root, config=TorxOperatorConfig(lambda_op=0.0))
    clean = apply_clean_target_teacher(params, target, target_mask, config=cfg, ops=clean_ops)
    teacher = clean["target"]["teacher"]
    h0 = clean["target"]["h0"]
    epsilon = initial_diffusion_epsilon(h0, diffusion_root, ids)
    initial = corrupt_h(h0, nu, epsilon)
    sanitized = make_sanitized_target_analysis(
        teacher.boundaries, teacher.length, config=cfg, target_width=target.shape[1]
    )
    trajectory = apply_denoising_trajectory(
        params,
        prompt,
        jnp.ones_like(prompt, dtype=bool),
        initial,
        sanitized,
        nu,
        ids,
        s_exec=2,
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
        dit_cycles=3,
    )
    predicted = jnp.argmax(trajectory.byte_logits, axis=-1)
    slot_bytes = teacher.slot_bytes[None, ...]
    slot_mask = teacher.slot_mask[None, ...]
    correct = (predicted == slot_bytes) & slot_mask
    log_probs = jax.nn.log_softmax(trajectory.byte_logits, axis=-1)
    selected = jnp.take_along_axis(log_probs, slot_bytes[..., None], axis=-1)[..., 0]
    byte_count = jnp.maximum(jnp.sum(slot_mask, axis=(2, 3)), 1)
    values = {
        "mse": jnp.mean((trajectory.h_hat - h0[None, ...]) ** 2, axis=(2, 3)).T,
        "nll": (-jnp.sum(jnp.where(slot_mask, selected, 0.0), axis=(2, 3)) / byte_count).T,
        "byte_accuracy": (jnp.sum(correct, axis=(2, 3)) / byte_count).T,
        "exact_accuracy": jnp.all(correct | ~slot_mask, axis=(2, 3)).T,
        "nonfinite": trajectory.diagnostics["nonfinite"].T,
    }
    if include_latents:
        values.update(
            {
                "clean_h": h0,
                "h_hat": jnp.swapaxes(trajectory.h_hat, 0, 1),
                "input_states": jnp.swapaxes(trajectory.input_states, 0, 1),
            }
        )
    return values


def evaluate(
    params: Any,
    prompt: jax.Array,
    target: jax.Array,
    ids: jax.Array,
    *,
    eta_diff: int = 0,
    lambda_op: float = 0.0,
    operator_root: jax.Array,
    diffusion_root: jax.Array,
    conditioning_mode: str = F3_NATIVE_S_CONDITIONING_TRAINED,
    include_latents: bool = False,
    runtime_batch: int = BATCH,
) -> dict[float, dict[str, np.ndarray]]:
    call = jax.jit(
        lambda model_params, prompts, targets, example_ids, nus: _batch_metrics(
            model_params,
            prompts,
            targets,
            example_ids,
            nus,
            eta_diff=eta_diff,
            lambda_op=lambda_op,
            operator_root=operator_root,
            diffusion_root=diffusion_root,
            conditioning_mode=conditioning_mode,
            include_latents=include_latents,
        )
    )
    result = {}
    for level in LEVELS:
        chunks: dict[str, list[np.ndarray]] = {}
        for start in range(0, len(ids), runtime_batch):
            end = min(start + runtime_batch, len(ids))
            values = jax.device_get(
                call(
                    params,
                    prompt[start:end],
                    target[start:end],
                    ids[start:end],
                    jnp.full((end - start,), level, dtype=jnp.float32),
                )
            )
            for name, value in values.items():
                chunks.setdefault(name, []).append(np.asarray(value))
        result[level] = {name: np.concatenate(parts) for name, parts in chunks.items()}
    return result


def summarize_evaluation(raw: dict[float, dict[str, np.ndarray]], seed_offset: int = 0):
    rows = []
    for level_index, (level, metrics) in enumerate(raw.items()):
        paired = {
            name: paired_summary(
                metrics[name][:, 0],
                metrics[name][:, 1],
                bootstrap_seed=BOOTSTRAP_SEED + seed_offset + 10 * level_index + index,
            )
            for index, name in enumerate(("mse", "nll", "byte_accuracy", "exact_accuracy"))
        }
        paired["byte_accuracy"]["absolute_change_percentage_points"] = (
            100.0 * paired["byte_accuracy"]["absolute_change"]
        )
        rows.append(
            {
                "nu": level,
                "count": len(metrics["nll"]),
                "s1_mse": float(np.mean(metrics["mse"][:, 0])),
                "s2_mse": float(np.mean(metrics["mse"][:, 1])),
                "s1_nll": float(np.mean(metrics["nll"][:, 0])),
                "s2_nll": float(np.mean(metrics["nll"][:, 1])),
                "s1_accuracy": float(np.mean(metrics["byte_accuracy"][:, 0])),
                "s2_accuracy": float(np.mean(metrics["byte_accuracy"][:, 1])),
                "s1_exact_accuracy": float(np.mean(metrics["exact_accuracy"][:, 0])),
                "s2_exact_accuracy": float(np.mean(metrics["exact_accuracy"][:, 1])),
                "nonfinite": bool(np.any(metrics["nonfinite"])),
                "paired": paired,
            }
        )
    return rows


def _load_training_state():
    latest = WORK / "latest.pkl"
    if not latest.exists():
        parent = load_parent()
        return parent["params"], parent["moments"], 0
    state = load_pickle(latest)
    if state["parent_sha256"] != EXPECTED_PARENT_SHA256:
        raise RuntimeError("mismatched F3 resumable state")
    if int(state["step"]) != PARENT_OPTIMIZER_STEP + int(state["f3_rollout_updates"]):
        raise RuntimeError("F3 checkpoint step coordinates disagree")
    return state["params"], state["moments"], int(state["f3_rollout_updates"])


def _save_training_state(params, moments, f3_updates: int, *, checkpoint: bool):
    state = {
        "params": params,
        "moments": moments,
        "step": PARENT_OPTIMIZER_STEP + f3_updates,
        "f3_rollout_updates": f3_updates,
        "parent_sha256": EXPECTED_PARENT_SHA256,
    }
    save_pickle(WORK / "latest.pkl", state)
    if checkpoint:
        save_pickle(WORK / f"rollout_{f3_updates:05d}.pkl", state)


def _frozen_hashes(params: Any, moments: Any) -> dict[str, str]:
    def rho_tree(tree):
        return {
            jax.tree_util.keystr(path): leaf
            for path, leaf in jax.tree.leaves_with_path(tree)
            if "['rho']" in jax.tree_util.keystr(path)
        }

    def teacher_tree(tree):
        return {name: tree["encoder"][name] for name in _CODEC_ENCODER_NAMES}

    return {
        "rho_params": array_hash(rho_tree(params)),
        "rho_moments": array_hash(rho_tree(moments)),
        "teacher_params": array_hash(teacher_tree(params)),
        "teacher_moments": array_hash(tuple(teacher_tree(moment) for moment in moments)),
    }


def _read_validation() -> dict[str, Any]:
    path = EVIDENCE / "validation_s1_s2.json"
    return json.loads(path.read_text()) if path.exists() else {"checkpoints": []}


def _routine_validation(params, f3_updates: int, checkpoint_path: Path):
    prompt, target, ids = split("validation", VALIDATION_COUNT)
    checkpoint_state = load_pickle(checkpoint_path)
    checkpoint = checkpoint_manifest(checkpoint_path, checkpoint_state)
    parent_state = load_parent()
    protocol = evaluation_protocol(
        checkpoint,
        ids,
        split_name="validation",
        conditioning_mode=F3_NATIVE_S_CONDITIONING_TRAINED,
        eta_diff=0,
        lambda_op=0.0,
        diffusion_root_seed=10_402,
        operator_root_seed=OPERATOR_ROOT_SEED,
        operator_root_index=None,
        baseline_checkpoint=checkpoint_manifest(PARENT, parent_state),
    )
    raw = evaluate(
        params,
        prompt,
        target,
        ids,
        operator_root=jax.random.PRNGKey(OPERATOR_ROOT_SEED),
        diffusion_root=VALIDATION_DIFFUSION_ROOT,
    )
    rows = summarize_evaluation(raw, seed_offset=f3_updates)
    f1_raw = evaluate(
        parent_state["params"],
        prompt,
        target,
        ids,
        operator_root=jax.random.PRNGKey(OPERATOR_ROOT_SEED),
        diffusion_root=VALIDATION_DIFFUSION_ROOT,
    )
    f1_rows = summarize_evaluation(f1_raw, seed_offset=50_000 + f3_updates)
    for row, f1_row in zip(rows, f1_rows, strict=True):
        forgetting_input = {
            "trained_s1_accuracy": row["s1_accuracy"],
            "f1_s1_accuracy": f1_row["s1_accuracy"],
            "trained_s1_nll": row["s1_nll"],
            "f1_s1_nll": f1_row["s1_nll"],
        }
        row.update({**forgetting_input, "step0_forgetting": step0_forgetting(forgetting_input)})
        row["effect"] = bucket_effect(row, require_uncertainty=True)
    payload = _read_validation()
    existing = [item for item in payload["checkpoints"] if item["f3_rollout_updates"] == f3_updates]
    if existing:
        require_exact_protocol(existing[0].get("protocol", {}), protocol, str(checkpoint_path))
        return existing[0]["rows"]
    payload["checkpoints"].append(
        {
            "f3_rollout_updates": f3_updates,
            "optimizer_absolute_step": PARENT_OPTIMIZER_STEP + f3_updates,
            "protocol": protocol,
            "rows": rows,
        }
    )
    payload["checkpoints"].sort(key=lambda item: item["f3_rollout_updates"])
    write_json(EVIDENCE / "validation_s1_s2.json", payload)
    return rows


def _validate_recorded_validation() -> set[int]:
    payload = _read_validation()
    if not payload["checkpoints"]:
        return set()
    _, _, ids = split("validation", VALIDATION_COUNT)
    parent_manifest = checkpoint_manifest(PARENT, load_parent())
    recorded = set()
    for item in payload["checkpoints"]:
        f3_updates = int(item["f3_rollout_updates"])
        checkpoint_path = WORK / f"rollout_{f3_updates:05d}.pkl"
        if not checkpoint_path.exists():
            raise RuntimeError(f"validation references missing checkpoint {checkpoint_path}")
        expected = evaluation_protocol(
            checkpoint_manifest(checkpoint_path, load_pickle(checkpoint_path)),
            ids,
            split_name="validation",
            conditioning_mode=F3_NATIVE_S_CONDITIONING_TRAINED,
            eta_diff=0,
            lambda_op=0.0,
            diffusion_root_seed=10_402,
            operator_root_seed=OPERATOR_ROOT_SEED,
            operator_root_index=None,
            baseline_checkpoint=parent_manifest,
        )
        require_exact_protocol(item.get("protocol", {}), expected, str(checkpoint_path))
        recorded.add(f3_updates)
    return recorded


def run_training(max_rollout_updates: int):
    if max_rollout_updates not in (5_000, 10_000):
        raise ValueError("max rollout updates must be 5000 or 10000")
    parent_state = load_parent()
    parent_manifest = checkpoint_manifest(PARENT, parent_state)
    _require_gradient_gate(parent_manifest)
    _require_trial_update(parent_manifest)
    smoke_path = EVIDENCE / "runtime_smoke.json"
    if not smoke_path.exists():
        raise RuntimeError("run the F3 10-update runtime smoke before training")
    smoke = json.loads(smoke_path.read_text())
    if (
        smoke.get("checkpoint") != parent_manifest
        or smoke.get("implementation_source_sha256") != implementation_source_hash()
        or not smoke.get("passed", False)
    ):
        raise RuntimeError("stale, mixed, or failed F3 runtime smoke evidence")
    params, moments, start = _load_training_state()
    expected_frozen = _frozen_hashes(parent_state["params"], parent_state["moments"])
    if _frozen_hashes(params, moments) != expected_frozen:
        raise RuntimeError("resumable F3 state changed frozen F1 rho/teacher parameters or moments")
    if max_rollout_updates == 10_000:
        validation_before_extension = _read_validation()
        if start != 5_000 or not validation_before_extension.get(
            "extension_to_10k_warranted", False
        ):
            raise RuntimeError(
                "10k training requires a completed 5k run that passes the predeclared extension rule"
            )
    prompt, target, ids = split("train", TRAIN_COUNT)
    update = jax.jit(stochastic_s2_rollout_train_step, static_argnames=("config",))
    recorded = _validate_recorded_validation()
    frozen_before = _frozen_hashes(params, moments)
    for f3_updates in range(start + 1, max_rollout_updates + 1):
        absolute_step = PARENT_OPTIMIZER_STEP + f3_updates
        batch = training_batch(prompt, target, ids, absolute_step)
        params, moments, metrics = update(
            params,
            moments,
            absolute_step,
            batch,
            TRAIN_OPERATOR_ROOT,
            TRAIN_DIFFUSION_ROOT,
            config=config(),
        )
        jax.block_until_ready(metrics["loss"])
        if not all(bool(jnp.all(jnp.isfinite(value))) for value in jax.tree.leaves(metrics)):
            raise RuntimeError(f"nonfinite F3 training metric at rollout update {f3_updates}")
        checkpoint = f3_updates in CHECKPOINTS
        if checkpoint:
            if _frozen_hashes(params, moments) != expected_frozen:
                raise RuntimeError("rho or target teacher changed at an F3 checkpoint")
            _save_training_state(params, moments, f3_updates, checkpoint=True)
            checkpoint_path = WORK / f"rollout_{f3_updates:05d}.pkl"
            checkpoint_state = load_pickle(checkpoint_path)
            checkpoint_info = checkpoint_manifest(checkpoint_path, checkpoint_state)
            if f3_updates not in recorded:
                append_jsonl(
                    EVIDENCE / "training.jsonl",
                    {
                        "f3_rollout_updates": f3_updates,
                        "optimizer_absolute_step": absolute_step,
                        "checkpoint": checkpoint_info,
                        "metrics": metrics,
                    },
                )
                _routine_validation(params, f3_updates, checkpoint_path)
            print(f"F3 rollout_updates={f3_updates} absolute_step={absolute_step}", flush=True)
        elif f3_updates % 100 == 0:
            if _frozen_hashes(params, moments) != expected_frozen:
                raise RuntimeError("rho or target teacher changed before an F3 resume point")
            _save_training_state(params, moments, f3_updates, checkpoint=False)
    frozen_after = _frozen_hashes(params, moments)
    if frozen_before != frozen_after or frozen_after != expected_frozen:
        raise RuntimeError("rho or target teacher changed during F3")
    validation = _read_validation()
    checkpoints = {item["f3_rollout_updates"]: item["rows"] for item in validation["checkpoints"]}
    extension = False
    if 2_000 in checkpoints and 5_000 in checkpoints:
        earlier = [rows for update, rows in checkpoints.items() if update < 5_000]
        extension = should_extend_to_10k(
            checkpoints[2_000], checkpoints[5_000]
        ) and is_pareto_best_5k(earlier, checkpoints[5_000])
    validation["extension_to_10k_warranted"] = extension
    validation["trained_through_rollout_updates"] = max_rollout_updates
    write_json(EVIDENCE / "validation_s1_s2.json", validation)
    return {"extension_to_10k_warranted": extension}


def _selected_checkpoint() -> tuple[int, Path]:
    validation = _read_validation()
    candidates = []
    for item in validation["checkpoints"]:
        informative = [row for row in item["rows"] if row["nu"] in INFORMATIVE_LEVELS]
        forgetting = sum(bool(row["step0_forgetting"]) for row in informative)
        if forgetting >= 2 or any(row["nonfinite"] for row in informative):
            continue
        effects = [row["effect"] for row in informative]
        candidates.append(
            (
                effects.count("benefit"),
                -effects.count("degradation"),
                float(np.mean([row["s2_accuracy"] for row in informative])),
                -float(np.mean([row["s2_nll"] for row in informative])),
                item["f3_rollout_updates"],
            )
        )
    if not candidates:
        raise RuntimeError("no eligible F3 checkpoint")
    selected = int(max(candidates)[-1])
    return selected, WORK / f"rollout_{selected:05d}.pkl"


def _write_per_example(
    path: Path,
    raw: dict[float, dict[str, np.ndarray]],
    ids: np.ndarray,
    protocol_sha256: str,
):
    with path.open("w", encoding="utf-8") as stream:
        for level, metrics in raw.items():
            for index, example_id in enumerate(ids):
                row = {
                    "global_example_id": int(example_id),
                    "nu": level,
                    "protocol_sha256": protocol_sha256,
                }
                for name in ("mse", "nll", "byte_accuracy", "exact_accuracy", "nonfinite"):
                    row[name] = metrics[name][index].tolist()
                stream.write(json.dumps(row, sort_keys=True) + "\n")


def _decoder_geometry(raw):
    rows = []
    for level, metrics in raw.items():
        delta_h = metrics["mse"][:, 1] - metrics["mse"][:, 0]
        delta_nll = metrics["nll"][:, 1] - metrics["nll"][:, 0]
        strict = (delta_h != 0) & (delta_nll != 0)
        correlation = float(np.corrcoef(delta_h, delta_nll)[0, 1])
        rows.append(
            {
                "nu": level,
                "correlation_delta_h_mse_delta_byte_nll": correlation,
                "quadrants": {
                    "latent_improves_decoder_improves": int(
                        np.sum((delta_h < 0) & (delta_nll < 0))
                    ),
                    "latent_improves_decoder_worsens": int(np.sum((delta_h < 0) & (delta_nll > 0))),
                    "latent_worsens_decoder_improves": int(np.sum((delta_h > 0) & (delta_nll < 0))),
                    "latent_worsens_decoder_worsens": int(np.sum((delta_h > 0) & (delta_nll > 0))),
                    "ties": int(np.sum(~strict)),
                },
            }
        )
    return {"delta_definition": "S2 minus S1", "per_nu": rows}


def _latent_state_stats(state: np.ndarray, clean: np.ndarray):
    flattened = state.reshape(len(state), -1).astype(np.float64)
    clean_flat = clean.reshape(len(clean), -1).astype(np.float64)
    dot = np.sum(flattened * clean_flat, axis=1)
    cosine = dot / np.maximum(
        np.linalg.norm(flattened, axis=1) * np.linalg.norm(clean_flat, axis=1), 1.0e-12
    )
    distance = np.sqrt(np.mean((flattened - clean_flat) ** 2, axis=1))
    return {
        "rms": float(np.sqrt(np.mean(flattened**2))),
        "per_coordinate_mean": np.mean(flattened, axis=0).tolist(),
        "per_coordinate_variance_summary": summary(np.var(flattened, axis=0, ddof=1)),
        "cosine_to_clean": summary(cosine),
        "rms_distance_to_clean": summary(distance),
    }


def _nearest_reference(state, reference):
    query = state.reshape(len(state), -1).astype(np.float64)
    ref = reference.reshape(len(reference), -1).astype(np.float64)
    distances_squared = np.maximum(
        np.sum(query**2, axis=1)[:, None] + np.sum(ref**2, axis=1)[None, :] - 2.0 * query @ ref.T,
        0.0,
    )
    return summary(np.sqrt(np.min(distances_squared, axis=1) / query.shape[1]))


def _latent_distribution(f3_raw, f1_raw):
    permutation = np.asarray(
        jax.random.permutation(jax.random.PRNGKey(STOCHASTIC_SUBSET_SEED), TEST_COUNT)
    )
    queries, references = permutation[:512], permutation[512:1024]
    rows = []
    for level in LEVELS:
        clean = f3_raw[level]["clean_h"]
        states = {
            "true_clean_h0": clean,
            "f1_h_hat_s1": f1_raw[level]["h_hat"][:, 0],
            "f1_input_s2": f1_raw[level]["input_states"][:, 1],
            "f3_h_hat_s1": f3_raw[level]["h_hat"][:, 0],
            "f3_input_s2": f3_raw[level]["input_states"][:, 1],
            "f3_h_hat_s2": f3_raw[level]["h_hat"][:, 1],
        }
        rows.append(
            {
                "nu": level,
                "states": {
                    name: {
                        **_latent_state_stats(state, clean),
                        "nearest_disjoint_clean_reference_rms_distance": _nearest_reference(
                            state[queries], clean[references]
                        ),
                    }
                    for name, state in states.items()
                },
            }
        )
    return {"query_count": 512, "reference_count": 512, "per_nu": rows}


def run_final_evaluation():
    selected, path = _selected_checkpoint()
    state = load_pickle(path)
    parent = load_parent()
    prompt, target, ids = split("test", TEST_COUNT)
    checkpoint = checkpoint_manifest(path, state)
    parent_checkpoint = checkpoint_manifest(PARENT, parent)
    native_protocol = evaluation_protocol(
        checkpoint,
        ids,
        split_name="test",
        conditioning_mode=F3_NATIVE_S_CONDITIONING_TRAINED,
        eta_diff=0,
        lambda_op=0.0,
        diffusion_root_seed=TEST_DIFFUSION_ROOT_SEED,
        operator_root_seed=OPERATOR_ROOT_SEED,
        operator_root_index=None,
        baseline_checkpoint=parent_checkpoint,
    )
    step0_protocol = evaluation_protocol(
        checkpoint,
        ids,
        split_name="test",
        conditioning_mode=F2_STEP0_CONDITIONING,
        eta_diff=0,
        lambda_op=0.0,
        diffusion_root_seed=TEST_DIFFUSION_ROOT_SEED,
        operator_root_seed=OPERATOR_ROOT_SEED,
        operator_root_index=None,
    )
    eta1_protocol = evaluation_protocol(
        checkpoint,
        ids,
        split_name="test",
        conditioning_mode=F3_NATIVE_S_CONDITIONING_TRAINED,
        eta_diff=1,
        lambda_op=0.0,
        diffusion_root_seed=TEST_DIFFUSION_ROOT_SEED,
        operator_root_seed=OPERATOR_ROOT_SEED,
        operator_root_index=None,
    )
    existing_validation = _read_validation()
    if "full_test_protocol" in existing_validation:
        require_exact_protocol(
            existing_validation["full_test_protocol"],
            native_protocol,
            str(EVIDENCE / "validation_s1_s2.json"),
        )
    raw_path = WORK / "final_native_raw.pkl"
    if raw_path.exists():
        existing_raw = load_pickle(raw_path)
        require_exact_protocol(existing_raw.get("protocol", {}), native_protocol, str(raw_path))
        if existing_raw.get("complete") is not True:
            raise RuntimeError("incomplete deterministic F3 result")
    for result_path, expected_protocol in (
        (EVIDENCE / "step0_conditioning_diagnostic.json", step0_protocol),
        (EVIDENCE / "eta1_diagnostic.json", eta1_protocol),
    ):
        if result_path.exists():
            existing = json.loads(result_path.read_text())
            require_exact_protocol(
                existing.get("protocol", {}), expected_protocol, str(result_path)
            )
    for result_path in (
        EVIDENCE / "decoder_geometry.json",
        EVIDENCE / "latent_distribution.json",
    ):
        if result_path.exists():
            existing = json.loads(result_path.read_text())
            if existing.get("source_protocol_sha256") != native_protocol["protocol_sha256"]:
                raise RuntimeError(f"mixed or stale deterministic F3 result in {result_path}")
    per_example_path = EVIDENCE / "validation_s1_s2_per_example.jsonl"
    if per_example_path.exists():
        first_line = next((line for line in per_example_path.read_text().splitlines() if line), "")
        if (
            not first_line
            or json.loads(first_line).get("protocol_sha256") != native_protocol["protocol_sha256"]
        ):
            raise RuntimeError(f"mixed or stale deterministic F3 result in {per_example_path}")
    operator_root = jax.random.PRNGKey(OPERATOR_ROOT_SEED)
    diffusion_root = jax.random.PRNGKey(TEST_DIFFUSION_ROOT_SEED)
    f3_raw = evaluate(
        state["params"],
        prompt,
        target,
        ids,
        operator_root=operator_root,
        diffusion_root=diffusion_root,
        include_latents=True,
    )
    f1_raw = evaluate(
        parent["params"],
        prompt,
        target,
        ids,
        operator_root=operator_root,
        diffusion_root=diffusion_root,
        include_latents=True,
    )
    step0_raw = evaluate(
        state["params"],
        prompt,
        target,
        ids,
        operator_root=operator_root,
        diffusion_root=diffusion_root,
        conditioning_mode=F2_STEP0_CONDITIONING,
    )
    eta1_raw = evaluate(
        state["params"],
        prompt,
        target,
        ids,
        eta_diff=1,
        operator_root=operator_root,
        diffusion_root=diffusion_root,
    )
    rows = summarize_evaluation(f3_raw, seed_offset=100_000)
    f1_rows = summarize_evaluation(f1_raw, seed_offset=110_000)
    for row, f1_row in zip(rows, f1_rows, strict=True):
        values = {
            "trained_s1_accuracy": row["s1_accuracy"],
            "f1_s1_accuracy": f1_row["s1_accuracy"],
            "trained_s1_nll": row["s1_nll"],
            "f1_s1_nll": f1_row["s1_nll"],
        }
        row.update(values)
        row["delta_s1_accuracy_vs_f1"] = row["s1_accuracy"] - f1_row["s1_accuracy"]
        row["delta_s1_nll_vs_f1"] = row["s1_nll"] - f1_row["s1_nll"]
        row["step0_forgetting"] = step0_forgetting(values)
        row["effect"] = bucket_effect(row, require_uncertainty=True)
    write_json(
        EVIDENCE / "validation_s1_s2.json",
        {
            **_read_validation(),
            "selected_checkpoint": selected,
            "full_test_protocol": native_protocol,
            "full_test_rows": rows,
        },
    )
    write_json(
        EVIDENCE / "step0_conditioning_diagnostic.json",
        {
            "protocol": step0_protocol,
            "conditioning": [0, 0],
            "rows": summarize_evaluation(step0_raw, seed_offset=120_000),
        },
    )
    write_json(
        EVIDENCE / "eta1_diagnostic.json",
        {
            "protocol": eta1_protocol,
            "eta_diff": 1,
            "rows": summarize_evaluation(eta1_raw, seed_offset=130_000),
        },
    )
    write_json(
        EVIDENCE / "decoder_geometry.json",
        {"source_protocol_sha256": native_protocol["protocol_sha256"], **_decoder_geometry(f3_raw)},
    )
    write_json(
        EVIDENCE / "latent_distribution.json",
        {
            "source_protocol_sha256": native_protocol["protocol_sha256"],
            **_latent_distribution(f3_raw, f1_raw),
        },
    )
    _write_per_example(
        per_example_path,
        f3_raw,
        np.asarray(ids),
        native_protocol["protocol_sha256"],
    )
    save_pickle(
        raw_path,
        {"complete": True, "protocol": native_protocol, "raw": f3_raw},
    )
    return {"selected_checkpoint": selected}


def run_operator_confirmation(root_index: int, stochastic_count: int):
    if not 0 <= root_index < ROOT_COUNT:
        raise ValueError("root index must lie in [0, 16)")
    selected, path = _selected_checkpoint()
    state = load_pickle(path)
    prompt, target, ids = split("test", TEST_COUNT)
    if stochastic_count not in (512, 1_024):
        raise ValueError("stochastic count must be 512 or 1024")
    positions = jax.random.permutation(jax.random.PRNGKey(STOCHASTIC_SUBSET_SEED), TEST_COUNT)[
        :stochastic_count
    ]
    subset_ids = ids[positions]
    checkpoint = checkpoint_manifest(path, state)
    protocol = evaluation_protocol(
        checkpoint,
        subset_ids,
        split_name="test",
        conditioning_mode=F3_NATIVE_S_CONDITIONING_TRAINED,
        eta_diff=0,
        lambda_op=1.0,
        diffusion_root_seed=TEST_DIFFUSION_ROOT_SEED,
        operator_root_seed=OPERATOR_ROOT_SEED,
        operator_root_index=root_index,
    )
    shard_path = WORK / "operator" / f"count_{stochastic_count}_root_{root_index:02d}.pkl"
    if shard_path.exists():
        existing = load_pickle(shard_path)
        require_exact_protocol(existing.get("protocol", {}), protocol, str(shard_path))
        if existing.get("complete") is not True:
            raise RuntimeError(f"incomplete F3 operator shard: {shard_path}")
        return existing
    operator_root = jax.random.fold_in(jax.random.PRNGKey(OPERATOR_ROOT_SEED), root_index)
    raw = evaluate(
        state["params"],
        prompt[positions],
        target[positions],
        subset_ids,
        lambda_op=1.0,
        operator_root=operator_root,
        diffusion_root=jax.random.PRNGKey(TEST_DIFFUSION_ROOT_SEED),
        runtime_batch=4,
    )
    save_pickle(
        shard_path,
        {"complete": True, "protocol": protocol, "raw": raw},
    )
    return {"complete": True, "protocol": protocol}


def aggregate_operator_confirmation(stochastic_count: int):
    if stochastic_count not in (512, 1_024):
        raise ValueError("stochastic count must be 512 or 1024")
    roots = []
    _, checkpoint_path = _selected_checkpoint()
    checkpoint_state = load_pickle(checkpoint_path)
    checkpoint = checkpoint_manifest(checkpoint_path, checkpoint_state)
    lambda_zero_bundle = load_pickle(WORK / "final_native_raw.pkl")
    if lambda_zero_bundle.get("complete") is not True:
        raise RuntimeError("incomplete deterministic F3 result")
    _, _, full_ids = split("test", TEST_COUNT)
    expected_lambda_zero_protocol = evaluation_protocol(
        checkpoint,
        full_ids,
        split_name="test",
        conditioning_mode=F3_NATIVE_S_CONDITIONING_TRAINED,
        eta_diff=0,
        lambda_op=0.0,
        diffusion_root_seed=TEST_DIFFUSION_ROOT_SEED,
        operator_root_seed=OPERATOR_ROOT_SEED,
        operator_root_index=None,
        baseline_checkpoint=checkpoint_manifest(PARENT, load_parent()),
    )
    require_exact_protocol(
        lambda_zero_bundle.get("protocol", {}),
        expected_lambda_zero_protocol,
        str(WORK / "final_native_raw.pkl"),
    )
    lambda_zero = lambda_zero_bundle["raw"]
    subset_positions = np.asarray(
        jax.random.permutation(jax.random.PRNGKey(STOCHASTIC_SUBSET_SEED), TEST_COUNT)[
            :stochastic_count
        ]
    )
    shard_protocol_hashes = []
    for root_index in range(ROOT_COUNT):
        path = WORK / "operator" / f"count_{stochastic_count}_root_{root_index:02d}.pkl"
        if not path.exists():
            raise RuntimeError(f"missing operator root {root_index}")
        shard = load_pickle(path)
        expected = evaluation_protocol(
            checkpoint,
            full_ids[subset_positions],
            split_name="test",
            conditioning_mode=F3_NATIVE_S_CONDITIONING_TRAINED,
            eta_diff=0,
            lambda_op=1.0,
            diffusion_root_seed=TEST_DIFFUSION_ROOT_SEED,
            operator_root_seed=OPERATOR_ROOT_SEED,
            operator_root_index=root_index,
        )
        require_exact_protocol(shard.get("protocol", {}), expected, str(path))
        if shard.get("complete") is not True:
            raise RuntimeError(f"incomplete F3 operator shard: {path}")
        shard_protocol_hashes.append(shard["protocol"]["protocol_sha256"])
        roots.append(shard["raw"])
    rows = []
    for level_index, level in enumerate(LEVELS):
        root_changes: dict[str, np.ndarray] = {}
        root_averaged: dict[str, np.ndarray] = {}
        for name in ("mse", "nll", "byte_accuracy"):
            stacked = np.stack([root[level][name] for root in roots])
            root_changes[name] = np.mean(stacked[:, :, 1] - stacked[:, :, 0], axis=1)
            root_averaged[name] = np.mean(stacked, axis=0)
        root_level = {}
        for name, values in root_changes.items():
            stats = summary(values)
            half = ROOT_T_95_DF15 * stats["se"]
            root_level[name] = {
                **stats,
                "student_t_95_ci_low": stats["mean"] - half,
                "student_t_95_ci_high": stats["mean"] + half,
            }
        root_means = {
            name: {
                "s1": float(np.mean(values[:, 0])),
                "s2": float(np.mean(values[:, 1])),
            }
            for name, values in root_averaged.items()
        }
        paired_after_root_averaging = {
            name: paired_summary(
                values[:, 0],
                values[:, 1],
                bootstrap_seed=BOOTSTRAP_SEED + 200_000 + 10 * level_index + index,
            )
            for index, (name, values) in enumerate(root_averaged.items())
        }
        effect_input = {
            "s1_accuracy": root_means["byte_accuracy"]["s1"],
            "s2_accuracy": root_means["byte_accuracy"]["s2"],
            "s1_nll": root_means["nll"]["s1"],
            "s2_nll": root_means["nll"]["s2"],
            "paired": paired_after_root_averaging,
        }
        rows.append(
            {
                "nu": level,
                "root_level_s2_minus_s1": root_level,
                "root_averaged_means": root_means,
                "matching_lambda_zero_subset": {
                    name: {
                        "s1": float(np.mean(lambda_zero[level][name][subset_positions, 0])),
                        "s2": float(np.mean(lambda_zero[level][name][subset_positions, 1])),
                    }
                    for name in ("mse", "nll", "byte_accuracy")
                },
                "paired_after_root_averaging": paired_after_root_averaging,
                "effect_input": effect_input,
                "effect": bucket_effect(effect_input, require_uncertainty=True),
            }
        )
    informative = [row for row in rows if row["nu"] in INFORMATIVE_LEVELS]
    stochastic_effect_rows = [row["effect_input"] for row in informative]
    stochastic_pattern = effect_pattern(stochastic_effect_rows, require_uncertainty=True)
    validation = _read_validation()
    lambda_zero_informative = [
        row for row in validation.get("full_test_rows", []) if row["nu"] in INFORMATIVE_LEVELS
    ]
    lambda_zero_pattern = (
        effect_pattern(lambda_zero_informative, require_uncertainty=True)
        if len(lambda_zero_informative) == 3
        else "missing"
    )
    symmetric_agreement = (
        lambda_zero_pattern in {"benefit", "neutral", "degradation"}
        and stochastic_pattern == lambda_zero_pattern
    )
    high_variance = any(
        (
            row["root_level_s2_minus_s1"]["byte_accuracy"]["student_t_95_ci_low"] <= -0.01
            and row["root_level_s2_minus_s1"]["byte_accuracy"]["student_t_95_ci_high"] >= 0.01
        )
        or (
            row["root_level_s2_minus_s1"]["nll"]["student_t_95_ci_low"]
            <= -0.02 * abs(row["root_averaged_means"]["nll"]["s1"])
            and row["root_level_s2_minus_s1"]["nll"]["student_t_95_ci_high"]
            >= 0.02 * abs(row["root_averaged_means"]["nll"]["s1"])
        )
        for row in informative
    )
    escalation_required = stochastic_count == 512 and (high_variance or not symmetric_agreement)
    aggregation_protocol_body = {
        "contract": EVALUATION_PROTOCOL,
        "checkpoint": checkpoint,
        "deterministic_source_protocol_sha256": expected_lambda_zero_protocol["protocol_sha256"],
        "stochastic_shard_protocol_sha256": shard_protocol_hashes,
        "examples_per_nu": stochastic_count,
        "operator_roots": ROOT_COUNT,
    }
    aggregation_protocol = {
        **aggregation_protocol_body,
        "protocol_sha256": json_hash(aggregation_protocol_body),
    }
    payload = {
        "protocol": aggregation_protocol,
        "examples_per_nu": stochastic_count,
        "operator_roots": ROOT_COUNT,
        "lambda_op": 1.0,
        "sigma_op": 1.0e-3,
        "student_t_df": 15,
        "per_nu": rows,
        "lambda_zero_effect_pattern": lambda_zero_pattern,
        "stochastic_effect_pattern": stochastic_pattern,
        "symmetric_practical_effect_agreement": symmetric_agreement,
        "high_variance": high_variance,
        "escalation_required": escalation_required,
    }
    detail_path = EVIDENCE / f"operator_noise_confirmation_{stochastic_count}.json"
    if detail_path.exists():
        existing = json.loads(detail_path.read_text())
        require_exact_protocol(existing.get("protocol", {}), aggregation_protocol, str(detail_path))
    write_json(detail_path, payload)
    output_path = EVIDENCE / "operator_noise_confirmation.json"
    if output_path.exists():
        existing = json.loads(output_path.read_text())
        same_count = existing.get("examples_per_nu") == stochastic_count
        allowed_escalation = (
            stochastic_count == 1_024
            and existing.get("examples_per_nu") == 512
            and existing.get("escalation_required") is True
        )
        if same_count:
            require_exact_protocol(
                existing.get("protocol", {}), aggregation_protocol, str(output_path)
            )
        elif not allowed_escalation:
            raise RuntimeError(f"mixed or stale F3 operator confirmation in {output_path}")
    write_json(output_path, payload)
    return payload


def aggregate_result():
    validation = _read_validation()
    if "full_test_rows" not in validation:
        raise RuntimeError("run final evaluation before F3 aggregation")
    operator = json.loads((EVIDENCE / "operator_noise_confirmation.json").read_text())
    if operator.get("escalation_required", False):
        raise RuntimeError("finite-Torx ambiguity requires the predeclared 1024-example escalation")
    informative = [row for row in validation["full_test_rows"] if row["nu"] in INFORMATIVE_LEVELS]
    selected_updates, selected_path = _selected_checkpoint()
    selected_manifest = checkpoint_manifest(selected_path, load_pickle(selected_path))
    expected_full_protocol = evaluation_protocol(
        selected_manifest,
        split("test", TEST_COUNT)[2],
        split_name="test",
        conditioning_mode=F3_NATIVE_S_CONDITIONING_TRAINED,
        eta_diff=0,
        lambda_op=0.0,
        diffusion_root_seed=TEST_DIFFUSION_ROOT_SEED,
        operator_root_seed=OPERATOR_ROOT_SEED,
        operator_root_index=None,
        baseline_checkpoint=checkpoint_manifest(PARENT, load_parent()),
    )
    require_exact_protocol(
        validation.get("full_test_protocol", {}),
        expected_full_protocol,
        str(EVIDENCE / "validation_s1_s2.json"),
    )
    operator_protocol = operator.get("protocol", {})
    if (
        operator_protocol.get("checkpoint") != selected_manifest
        or operator_protocol.get("deterministic_source_protocol_sha256")
        != expected_full_protocol["protocol_sha256"]
    ):
        raise RuntimeError("mixed or stale F3 operator confirmation")
    checkpoints = {item["f3_rollout_updates"]: item["rows"] for item in validation["checkpoints"]}
    max_updates = max(checkpoints)
    converged = max_updates >= 5_000 and not (
        max_updates == 5_000 and validation.get("extension_to_10k_warranted", False)
    )
    if max_updates == 10_000 and 5_000 in checkpoints:
        converged = not should_extend_to_10k(checkpoints[5_000], checkpoints[10_000])
    healthy = not any(row["nonfinite"] for row in validation["full_test_rows"])
    label = classify(
        informative,
        healthy=healthy,
        converged=converged,
        stochastic_agrees=operator["symmetric_practical_effect_agreement"],
    )
    if label == PHASE_F_3_BLOCKED:
        raise RuntimeError("blocked result cannot be aggregated as a completed experiment")
    if int(validation["selected_checkpoint"]) != selected_updates:
        raise RuntimeError("selected checkpoint disagrees with deterministic F3 provenance")
    result = {
        "protocol": {
            "contract": EVALUATION_PROTOCOL,
            "deterministic_protocol_sha256": expected_full_protocol["protocol_sha256"],
            "operator_protocol_sha256": operator_protocol.get("protocol_sha256"),
            "selected_checkpoint": selected_manifest,
        },
        "label": label,
        "selected_f3_rollout_updates": selected_updates,
        "selected_optimizer_absolute_step": PARENT_OPTIMIZER_STEP + selected_updates,
        "parent_checkpoint_sha256": EXPECTED_PARENT_SHA256,
        "healthy": healthy,
        "converged": converged,
        "informative_levels": INFORMATIVE_LEVELS,
        "no_s4": True,
    }
    result["protocol"]["protocol_sha256"] = json_hash(result["protocol"])
    result_path = EVIDENCE / "result.json"
    if result_path.exists():
        existing = json.loads(result_path.read_text())
        require_exact_protocol(existing.get("protocol", {}), result["protocol"], str(result_path))
    write_json(result_path, result)
    gradient = json.loads((EVIDENCE / "first_gradient.json").read_text())
    decoder = json.loads((EVIDENCE / "decoder_geometry.json").read_text())
    table = [
        "| nu | F1 S1 acc | F3 S1 acc | F3 S2 acc | F1 S1 NLL | F3 S1 NLL | F3 S2 NLL | effect | forgetting |",
        "|---:|---:|---:|---:|---:|---:|---:|:---|:---|",
    ]
    for row in validation["full_test_rows"]:
        table.append(
            f"| {row['nu']:.2f} | {row['f1_s1_accuracy']:.6f} | {row['s1_accuracy']:.6f} | "
            f"{row['s2_accuracy']:.6f} | {row['f1_s1_nll']:.6f} | {row['s1_nll']:.6f} | "
            f"{row['s2_nll']:.6f} | {row['effect']} | {row['step0_forgetting']} |"
        )
    gradient_lines = [
        f"- step 0 raw norm: `{gradient['gradients']['step0']['raw_global_norm']}`",
        f"- step 1 raw norm: `{gradient['gradients']['step1']['raw_global_norm']}`",
        f"- combined raw norm: `{gradient['gradients']['combined']['raw_global_norm']}`",
        f"- combined clipped norm: `{gradient['gradients']['combined']['clipped_norm']}`",
        f"- global cosine g0/g1: `{gradient['cosines']['global']['cosine_g0_g1']}`",
    ]
    decoder_lines = [
        f"- nu={row['nu']:.2f}: correlation `{row['correlation_delta_h_mse_delta_byte_nll']:.6f}`; "
        f"quadrants `{row['quadrants']}`"
        for row in decoder["per_nu"]
    ]
    training_rows = [
        json.loads(line)
        for line in (EVIDENCE / "training.jsonl").read_text().splitlines()
        if line.strip()
    ]
    training_table = [
        "| F3 updates | absolute step | step0 loss | step1 loss | rollout loss | step0 acc | step1 acc |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in training_rows:
        metrics = row["metrics"]
        training_table.append(
            f"| {row['f3_rollout_updates']} | {row['optimizer_absolute_step']} | "
            f"{metrics['step0_loss']:.6g} | {metrics['step1_loss']:.6g} | "
            f"{metrics['rollout_loss']:.6g} | {metrics['step0_byte_accuracy']:.6f} | "
            f"{metrics['step1_byte_accuracy']:.6f} |"
        )
    (EVIDENCE / "F3_RESULTS.md").write_text(
        "# Phase F.3 results\n\n"
        f"**{label}**\n\n"
        f"Selected checkpoint: {selected_updates} F3 rollout updates "
        f"(absolute optimizer step {PARENT_OPTIMIZER_STEP + selected_updates}).\n\n"
        "## First-gradient diagnostics\n\n"
        + "\n".join(gradient_lines)
        + "\n\n## Training curve\n\n"
        + "\n".join(training_table)
        + "\n\n## Full-test native `[0,1]` result\n\n"
        + "\n".join(table)
        + "\n\n## Decoder geometry\n\n"
        + "\n".join(decoder_lines)
        + "\n\n## Finite Torx\n\n"
        + f"The {operator['operator_roots']}-root, {operator['examples_per_nu']}-example "
        + "confirmation had symmetric practical-effect agreement="
        + f"`{operator['symmetric_practical_effect_agreement']}`.\n\n"
        + "The label uses the full-test eta-zero native-trained `[0,1]` trajectory, paired "
        + "practical gates, F1 step-0 preservation, and finite-Torx confirmation. No S4 "
        + "training or architecture change was performed.\n"
    )
    for history_path, heading in (
        (ROOT / "results/phase_f/RESULTS.md", "Phase F.3"),
        (ROOT / "results/phase_f/DECISION.md", "Phase F.3 decision"),
    ):
        text = history_path.read_text()
        marker = f"## {heading}"
        if marker not in text:
            history_path.write_text(
                text.rstrip()
                + f"\n\n{marker}\n\n**{label}**\n\n"
                + "Self-generated S=2 rollout training continued the accepted F1 optimizer state. "
                + "See `f3/F3_RESULTS.md`; F0/F1/F2 history remains unchanged.\n"
            )
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        required=True,
        choices=(
            "audit",
            "gradient",
            "trial-update",
            "smoke",
            "train",
            "final",
            "operator",
            "aggregate-operator",
            "aggregate",
        ),
    )
    parser.add_argument("--max-rollout-updates", type=int, choices=(5_000, 10_000), default=5_000)
    parser.add_argument("--root-index", type=int, default=0)
    parser.add_argument("--stochastic-count", type=int, choices=(512, 1_024), default=512)
    args = parser.parse_args()
    if args.stage == "audit":
        run_audit()
    elif args.stage == "gradient":
        run_first_gradient()
    elif args.stage == "trial-update":
        run_trial_update()
    elif args.stage == "smoke":
        run_runtime_smoke()
    elif args.stage == "train":
        run_training(args.max_rollout_updates)
    elif args.stage == "final":
        run_final_evaluation()
    elif args.stage == "operator":
        run_operator_confirmation(args.root_index, args.stochastic_count)
    elif args.stage == "aggregate-operator":
        aggregate_operator_confirmation(args.stochastic_count)
    else:
        aggregate_result()


if __name__ == "__main__":
    main()
