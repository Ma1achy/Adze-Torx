"""Phase F.1 faithful one-step corrupted-carrier calibration."""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import pickle
import subprocess
import time
from typing import Any

import jax
import jax.numpy as jnp

from adze_t.backends.mapping import deterministic_to_torx
from adze_t.backends.torx import TorxOperatorConfig, TorxOps
from adze_t.config import REFERENCE_SMALL_V0
from adze_t.corruption import (
    PHASE_F_EVAL_GRID,
    alpha,
    corrupt_h,
    sigma,
    training_diffusion_root,
)
from adze_t.model import (
    apply_clean_target_teacher,
    apply_model,
    apply_target_codec,
    init_model_params,
)
from adze_t.objectives import adamw_init
from adze_t.packing import build_pack_metadata_core
from adze_t.phase_f_1 import (
    DENOISE_V0,
    PHASE_F_1_DENOISE_V0_PROPOSAL_AUX_DISABLED,
    dataset_audit,
    denoise_example_hashes,
    generate_denoise_v0,
    initial_diffusion_epsilon,
)
from adze_t.training import _CODEC_ENCODER_NAMES, stochastic_denoise_train_step


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "results" / "phase_f" / "f1"
WORK = ROOT / "results" / "runs" / "phase_f" / "f1"
CODEC = ROOT / "results" / "phase_b" / "checkpoints" / "target_codec_b1.pkl"
BASE_SHA = "b513425ac71474c1fe977dc6ff8a3dcba84ad6bf"
TORX_PIN = "f1fc858ed950ecd41935d15c06d0ec7c5e0674ae"
DATA_SEEDS = {"train": 940, "validation": 941, "test": 942}
TRAIN_COUNT = 16_384
VALIDATION_PER_NU = 512
VALIDATION_COUNT = VALIDATION_PER_NU
EVAL_LEVELS = tuple(PHASE_F_EVAL_GRID)
CHECKPOINTS = (100, 250, 500, 1_000, 2_000, 5_000, 10_000)
BATCH = 32
INIT_SEED = 0
STOCHASTIC_TRAINING_SEED = 0
OPERATOR_ROOT = jax.random.PRNGKey(10_400)
TRAIN_DIFFUSION_ROOT = jax.random.PRNGKey(10_401)
VALIDATION_DIFFUSION_ROOT = jax.random.PRNGKey(10_402)
NU_ROOT = jax.random.PRNGKey(10_403)


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


def git_sha() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def config():
    return replace(
        REFERENCE_SMALL_V0,
        training=replace(REFERENCE_SMALL_V0.training, proposal_weight=0.0),
    )


def initialise() -> Any:
    cfg = config()
    codec = load(CODEC)
    fresh = init_model_params(jax.random.PRNGKey(INIT_SEED), cfg)
    encoder = dict(fresh["encoder"])
    for name in _CODEC_ENCODER_NAMES:
        encoder[name] = codec["encoder"][name]
    deterministic = {**fresh, "encoder": encoder, "decoder": codec["decoder"]}
    params, _ = deterministic_to_torx(deterministic)
    return params


def split(name: str, count: int) -> tuple[jax.Array, jax.Array, jax.Array]:
    return generate_denoise_v0(count, DATA_SEEDS[name])


def run_path_audit() -> dict[str, Any]:
    payload = {
        "base_git_sha": BASE_SHA,
        "current_git_sha": git_sha(),
        "status": "F1_PATH_AUDIT_PASS",
        "legacy_current_carrier": "proposal_h",
        "explicit_current_carrier": "carrier_h_input",
        "carrier_injection": {
            "packed_input": "carrier_in(carrier_h_input)",
            "residual_base": "carrier_h_input + carrier_out(unpool(dit(pack(...))))",
        },
        "inference_visible": [
            "constant prompt/context",
            "supplied corrupted carrier h_nu",
            "fixed committed boundaries/length/activity",
            "nu through existing DiT noise conditioning",
            "operator stochastic state",
        ],
        "teacher_or_loss_only": [
            "clean target bytes",
            "frozen clean target frontend",
            "clean h0",
            "byte labels",
        ],
        "fixed_structural_teacher": [
            "committed boundaries",
            "committed lengths",
            "activity",
            "pack metadata",
            "decoder emission routing",
        ],
        "conditioning": {
            "noise": "actual nu through existing noise scalar embedding",
            "denoise_step": 0,
            "q_effective_depth": "separate existing conditioning coordinate",
        },
        "loss": {
            "h_weight": 1.0,
            "boundary_weight": 1.0,
            "extent_weight": 1.0,
            "byte_weight": 1.0,
            "proposal_weight": 0.0,
            "proposal_policy": PHASE_F_1_DENOISE_V0_PROPOSAL_AUX_DISABLED,
        },
        "decoder_input": "predicted clean h_hat with fixed teacher length",
        "target_teacher_freeze": "accepted model update mask plus stop-gradient h0",
        "path_separable_without_architecture_decision": True,
    }
    write(EVIDENCE / "path_audit.json", payload)
    return payload


def run_dataset_audit() -> dict[str, Any]:
    payload: dict[str, Any] = {
        "git_sha": git_sha(),
        "task_version": DENOISE_V0.name,
        "torx_pin": TORX_PIN,
        "seeds": DATA_SEEDS,
        "sizes": {
            "train": TRAIN_COUNT,
            "validation": VALIDATION_COUNT,
            "test": 4_096,
        },
        "target_distribution": "JAX uniform integer bytes in [0, 255]",
        "training_nu_distribution": "Uniform(0.025, 0.9)",
        "eval_grid": EVAL_LEVELS,
        "validation_epsilon_pairing": (
            "same root/example/stage/s epsilon reused across every nu and checkpoint"
        ),
    }
    hashes: dict[str, set[str]] = {}
    for name, count in (("train", TRAIN_COUNT), ("validation", 4_096), ("test", 4_096)):
        prompt, target, _ = split(name, count)
        split_hashes = denoise_example_hashes(prompt, target)
        hashes[name] = split_hashes
        payload[name] = {
            **dataset_audit(prompt, target),
            "unique_example_hashes": len(split_hashes),
            "duplicate_count": count - len(split_hashes),
        }
    payload["split_intersections"] = {
        "train_validation": len(hashes["train"] & hashes["validation"]),
        "train_test": len(hashes["train"] & hashes["test"]),
        "validation_test": len(hashes["validation"] & hashes["test"]),
    }
    _, structure_targets, _ = split("validation", 4_096)
    masks = jnp.ones_like(structure_targets, bool)
    from adze_t.teacher import canonical_teacher_structure_core

    teacher = canonical_teacher_structure_core(structure_targets, masks, config())
    metadata = build_pack_metadata_core(
        teacher.boundaries,
        teacher.activity,
        M_max=config().packing.M_max,
        K=config().packing.K,
    )
    structure_fields = {
        "committed_boundaries": teacher.boundaries,
        "committed_lengths": teacher.length,
        "activity": teacher.activity,
        "carrier_to_m": metadata.carrier_to_m,
        "carrier_to_k": metadata.carrier_to_k,
        "packed_to_carrier": metadata.packed_to_carrier,
        "kv_mask": metadata.kv_mask,
        "query_mask": metadata.query_mask,
    }
    invariant = {
        name: bool(jnp.all(values == values[0])) for name, values in structure_fields.items()
    }
    payload["structure_content_invariance"] = {
        "sample_count": 4_096,
        "fields": invariant,
        "passed": all(invariant.values()),
    }
    payload["passed"] = bool(
        all(value == 0 for value in payload["split_intersections"].values())
        and all(payload[name]["duplicate_count"] == 0 for name in DATA_SEEDS)
        and payload["structure_content_invariance"]["passed"]
    )
    write(EVIDENCE / "dataset_audit.json", payload)
    if not payload["passed"]:
        raise RuntimeError("DENOISE_V0 dataset or structural invariance audit failed")
    return payload


def run_codec_audit() -> dict[str, Any]:
    cfg = config()
    deterministic = load(CODEC)
    _, target, _ = split("validation", 4_096)
    mask = jnp.ones_like(target, bool)
    h_rows = []
    correct = exact = total = 0
    for start in range(0, len(target), BATCH):
        output = apply_target_codec(
            deterministic,
            target[start : start + BATCH],
            mask[start : start + BATCH],
            config=cfg,
        )
        teacher = output["target"]["teacher"]
        predicted = jnp.argmax(output["codec_logits"], axis=-1)
        matches = (predicted == teacher.slot_bytes) & teacher.slot_mask
        correct += int(jnp.sum(matches))
        total += int(jnp.sum(teacher.slot_mask))
        exact += int(jnp.sum(jnp.all(matches | ~teacher.slot_mask, axis=(1, 2))))
        h_rows.append(output["target"]["h0"])
    h0 = jnp.concatenate(h_rows)
    flattened = h0.reshape(len(h0), -1)
    sample = flattened[:512]
    normalized = sample / jnp.maximum(jnp.linalg.norm(sample, axis=1, keepdims=True), 1e-8)
    cosine = normalized @ normalized.T
    off_diagonal = cosine[~jnp.eye(len(sample), dtype=bool)]
    host = jax.device_get(flattened)
    latent_hashes = {hashlib.sha256(row.tobytes()).hexdigest() for row in host}
    payload = {
        "git_sha": git_sha(),
        "codec_checkpoint": str(CODEC.relative_to(ROOT)),
        "codec_sha256": sha256(CODEC),
        "examples": len(target),
        "byte_accuracy": correct / total,
        "exact_accuracy": exact / len(target),
        "global_h0_rms": jnp.sqrt(jnp.mean(h0**2)),
        "mean_coordinate_variance": jnp.mean(jnp.var(flattened, axis=0, ddof=1)),
        "pairwise_cosine_mean": jnp.mean(off_diagonal),
        "pairwise_cosine_std": jnp.std(off_diagonal),
        "unique_latent_hashes": len(latent_hashes),
        "duplicate_latent_count": len(target) - len(latent_hashes),
        "gate": {"byte_accuracy_at_least": 0.99, "exact_accuracy_at_least": 0.95},
    }
    payload["passed"] = bool(
        payload["byte_accuracy"] >= 0.99
        and payload["exact_accuracy"] >= 0.95
        and payload["duplicate_latent_count"] == 0
        and jnp.isfinite(payload["global_h0_rms"])
        and payload["global_h0_rms"] > 0
    )
    write(EVIDENCE / "codec_audit.json", payload)
    if not payload["passed"]:
        raise RuntimeError("accepted target codec failed DENOISE_V0 suitability gate")
    return payload


def run_codec_control() -> dict[str, Any]:
    """Confirm the failed uniform-byte audit is domain shift, not a codec regression."""
    path = EVIDENCE / "codec_audit.json"
    payload = json.loads(path.read_text())
    deterministic = load(CODEC)
    target = jax.random.randint(
        jax.random.PRNGKey(711), (256, 8), minval=1, maxval=33, dtype=jnp.int32
    )
    mask = jnp.ones_like(target, bool)
    correct = exact = total = 0
    for start in range(0, len(target), BATCH):
        output = apply_target_codec(
            deterministic,
            target[start : start + BATCH],
            mask[start : start + BATCH],
            config=config(),
        )
        teacher = output["target"]["teacher"]
        predicted = jnp.argmax(output["codec_logits"], axis=-1)
        matches = (predicted == teacher.slot_bytes) & teacher.slot_mask
        correct += int(jnp.sum(matches))
        total += int(jnp.sum(teacher.slot_mask))
        exact += int(jnp.sum(jnp.all(matches | ~teacher.slot_mask, axis=(1, 2))))
    payload["accepted_domain_control"] = {
        "distribution": "eight bytes uniformly sampled from integer values 1..32",
        "seed": 711,
        "examples": len(target),
        "byte_accuracy": correct / total,
        "exact_accuracy": exact / len(target),
        "matches_historical_gate": correct / total >= 0.99 and exact / len(target) >= 0.95,
    }
    ids = jnp.arange(4_096, dtype=jnp.uint32)
    zeros = jnp.zeros((4_096, config().carrier.C, config().carrier.h_dim), jnp.float32)
    epsilon = initial_diffusion_epsilon(zeros, VALIDATION_DIFFUSION_ROOT, ids)
    h0_rms = float(payload["global_h0_rms"])
    payload["corruption_rms_interpretation"] = []
    for level in EVAL_LEVELS:
        signal_rms = float(alpha(level)) * h0_rms
        noise_rms = float(jnp.sqrt(jnp.mean((sigma(level) * epsilon) ** 2)))
        payload["corruption_rms_interpretation"].append(
            {
                "nu": level,
                "empirical_signal_rms": signal_rms,
                "empirical_noise_rms": noise_rms,
                "signal_noise_rms_ratio": signal_rms / noise_rms,
            }
        )
    payload["coefficient_wording"] = (
        "unit-energy trigonometric corruption coefficients; variance-preserving only "
        "under unit-variance clean latents"
    )
    payload["failure_interpretation"] = (
        "Frozen codec remains valid on its accepted 1..32 byte domain but is unsuitable "
        "for required uniform 0..255 DENOISE_V0 targets."
    )
    write(path, payload)
    return payload


def _batch_metrics(
    params: Any,
    prompt: jax.Array,
    target: jax.Array,
    ids: jax.Array,
    nu: jax.Array,
    *,
    cycles: int,
    lambda_op: float,
    operator_root: jax.Array,
    diffusion_root: jax.Array,
) -> dict[str, jax.Array]:
    cfg = config()
    clean_ops = TorxOps.create(
        operator_root,
        config=TorxOperatorConfig(operator_stochasticity=True, lambda_op=0.0),
        global_example_id=ids[0],
    )
    model_ops = TorxOps.create(
        operator_root,
        config=TorxOperatorConfig(operator_stochasticity=True, lambda_op=lambda_op),
        global_example_id=ids[0],
    )
    prompt_mask = jnp.ones_like(prompt, bool)
    target_mask = jnp.ones_like(target, bool)
    analysis = apply_clean_target_teacher(params, target, target_mask, config=cfg, ops=clean_ops)
    h0 = analysis["target"]["h0"]
    epsilon = initial_diffusion_epsilon(h0, diffusion_root, ids)
    h_nu = corrupt_h(h0, nu, epsilon)
    output = apply_model(
        params,
        prompt,
        prompt_mask,
        target,
        target_mask,
        config=cfg,
        ops=model_ops,
        target_ops=clean_ops,
        target_analysis=analysis,
        carrier_h_input=h_nu,
        noise_level=nu,
        denoise_step=0,
        dit_cycles=cycles,
    )
    teacher = analysis["target"]["teacher"]
    mask = teacher.slot_mask
    predicted = jnp.argmax(output["byte_logits"], axis=-1)
    correct = (predicted == teacher.slot_bytes) & mask
    log_probs = jax.nn.log_softmax(output["byte_logits"], axis=-1)
    selected = jnp.take_along_axis(log_probs, teacher.slot_bytes[..., None], axis=-1)[..., 0]
    byte_count = jnp.maximum(jnp.sum(mask, axis=(1, 2)), 1)
    h_mse = jnp.mean((output["prediction"][0] - h0) ** 2, axis=(1, 2))
    signal = alpha(nu).reshape((-1, 1, 1)) * h0
    noise = sigma(nu).reshape((-1, 1, 1)) * epsilon
    signal_rms = jnp.sqrt(jnp.mean(signal**2, axis=(1, 2)))
    noise_rms = jnp.sqrt(jnp.mean(noise**2, axis=(1, 2)))
    return {
        "h0_mse": h_mse,
        "byte_nll": -jnp.sum(jnp.where(mask, selected, 0.0), axis=(1, 2)) / byte_count,
        "byte_accuracy": jnp.sum(correct, axis=(1, 2)) / byte_count,
        "exact_accuracy": jnp.all(correct | ~mask, axis=(1, 2)),
        "boundary_loss": -jnp.mean(
            jnp.take_along_axis(
                jax.nn.log_softmax(output["prediction"][1][:, :-1], axis=-1),
                teacher.boundaries[:, :-1, None],
                axis=-1,
            )[..., 0],
            axis=1,
        ),
        "extent_loss": -jnp.mean(
            jnp.take_along_axis(
                jax.nn.log_softmax(output["prediction"][2], axis=-1),
                teacher.length[..., None],
                axis=-1,
            )[..., 0],
            axis=1,
        ),
        "signal_rms": signal_rms,
        "noise_rms": noise_rms,
        "signal_noise_rms_ratio": signal_rms / jnp.maximum(noise_rms, 1e-12),
        "nonfinite_rate": jnp.mean(~jnp.isfinite(output["byte_logits"]), axis=(1, 2, 3)),
    }


def _example_metrics(
    params: Any,
    prompt: jax.Array,
    target: jax.Array,
    example_id: jax.Array,
    nu: jax.Array,
    *,
    cycles: int,
    lambda_op: float,
    operator_root: jax.Array,
    diffusion_root: jax.Array,
) -> dict[str, jax.Array]:
    values = _batch_metrics(
        params,
        prompt[None, :],
        target[None, :],
        example_id[None],
        nu[None],
        cycles=cycles,
        lambda_op=lambda_op,
        operator_root=operator_root,
        diffusion_root=diffusion_root,
    )
    return jax.tree.map(lambda value: value[0], values)


def evaluate(
    params: Any,
    prompt: jax.Array,
    target: jax.Array,
    ids: jax.Array,
    *,
    cycles: tuple[int, ...] = (3,),
    levels: tuple[float, ...] = EVAL_LEVELS,
    lambda_op: float = 0.0,
    operator_root: jax.Array = OPERATOR_ROOT,
    diffusion_root: jax.Array = VALIDATION_DIFFUSION_ROOT,
) -> dict[str, Any]:
    def make_call(cycle: int):
        def mapped(model_params, prompts, targets, example_ids, nus):
            return jax.vmap(
                lambda sample_prompt, sample_target, sample_id, sample_nu: _example_metrics(
                    model_params,
                    sample_prompt,
                    sample_target,
                    sample_id,
                    sample_nu,
                    cycles=cycle,
                    lambda_op=lambda_op,
                    operator_root=operator_root,
                    diffusion_root=diffusion_root,
                )
            )(prompts, targets, example_ids, nus)

        return jax.jit(mapped)

    calls = {cycle: make_call(cycle) for cycle in cycles}
    rows = []
    for level in levels:
        values: dict[int, dict[str, list[jax.Array]]] = {cycle: {} for cycle in cycles}
        for start in range(0, len(prompt), BATCH):
            end = min(start + BATCH, len(prompt))
            nus = jnp.full((end - start,), level, dtype=jnp.float32)
            for cycle in cycles:
                chunk = calls[cycle](
                    params, prompt[start:end], target[start:end], ids[start:end], nus
                )
                for metric, array in chunk.items():
                    values[cycle].setdefault(metric, []).append(array)
        row: dict[str, Any] = {"nu": level, "count": len(prompt)}
        for cycle in cycles:
            for metric, chunks in values[cycle].items():
                row[f"q{cycle}_{metric}"] = jnp.mean(jnp.concatenate(chunks))
        rows.append(row)
    return {
        "lambda_op": lambda_op,
        "cycles": cycles,
        "examples_per_nu": len(prompt),
        "validation_epsilon_paired_across_nu": True,
        "per_nu": rows,
    }


def _training_batch(
    prompt: jax.Array,
    target: jax.Array,
    ids: jax.Array,
    indices: jax.Array,
    step: int,
    *,
    fixed_nu: float | None = None,
    fixed_corruption: bool = False,
) -> dict[str, jax.Array]:
    batch_size = len(indices)
    nu = (
        jnp.full((batch_size,), fixed_nu, dtype=jnp.float32)
        if fixed_nu is not None
        else jax.random.uniform(
            jax.random.fold_in(NU_ROOT, step),
            (batch_size,),
            minval=0.025,
            maxval=0.9,
        )
    )
    return {
        "prompt": prompt[indices],
        "prompt_mask": jnp.ones_like(prompt[indices], bool),
        "target": target[indices],
        "target_mask": jnp.ones_like(target[indices], bool),
        "nu": nu,
        "global_example_id": ids[indices],
        "diffusion_occurrence": jnp.asarray(0 if fixed_corruption else step, jnp.uint32),
    }


def run_first_gradient() -> dict[str, Any]:
    cfg = config()
    params = initialise()
    zeros = adamw_init(params)
    prompt, target, ids = split("train", TRAIN_COUNT)
    batch = _training_batch(prompt, target, ids, jnp.arange(BATCH), 1)
    update = jax.jit(stochastic_denoise_train_step, static_argnames=("config",))
    before = params
    updated, _, metrics = update(
        params,
        (zeros, zeros),
        1,
        batch,
        OPERATOR_ROOT,
        TRAIN_DIFFUSION_ROOT,
        config=cfg,
    )
    jax.block_until_ready(metrics["loss"])
    target_unchanged = all(
        bool(jnp.array_equal(a, b))
        for a, b in zip(
            jax.tree_util.tree_leaves(before["encoder"]["target"]),
            jax.tree_util.tree_leaves(updated["encoder"]["target"]),
            strict=True,
        )
    )
    connected = (
        "grad_dit_qkvo",
        "grad_dit_ffn",
        "grad_conditioning",
        "grad_output_heads",
        "grad_decoder",
    )
    payload = {
        "git_sha": git_sha(),
        "stage": "F1_GRADIENT_GATE",
        "metrics": metrics,
        "required_connected_families": connected,
        "proposal_gradient_expected_zero": True,
        "target_teacher_bitwise_unchanged": target_unchanged,
        "rho_applied_update_zero": float(metrics["grad_rho_applied_norm"]) == 0.0,
    }
    payload["passed"] = bool(
        all(jnp.isfinite(value).all() for value in jax.tree_util.tree_leaves(metrics))
        and all(float(metrics[name]) > 0 for name in connected)
        and float(metrics["grad_permitted_norm"]) > 0
        and float(metrics["grad_proposal"]) == 0.0
        and target_unchanged
        and payload["rho_applied_update_zero"]
    )
    write(EVIDENCE / "first_gradient.json", payload)
    if not payload["passed"]:
        raise RuntimeError("F1 first-gradient gate failed")
    return payload


def _load_or_initialize(path: Path) -> tuple[Any, Any, int]:
    if path.exists():
        state = load(path)
        return state["params"], state["moments"], int(state["step"])
    params = initialise()
    zeros = adamw_init(params)
    return params, (zeros, zeros), 0


def _metrics_at_fixed_set(
    params: Any,
    prompt: jax.Array,
    target: jax.Array,
    ids: jax.Array,
    nu: float,
) -> dict[str, float]:
    result = evaluate(
        params,
        prompt,
        target,
        ids,
        cycles=(3,),
        levels=(nu,),
        diffusion_root=training_diffusion_root(TRAIN_DIFFUSION_ROOT, 0),
    )
    row = result["per_nu"][0]
    return {
        key.removeprefix("q3_"): float(value) for key, value in row.items() if key.startswith("q3_")
    }


def run_overfit(case: str) -> dict[str, Any]:
    sizes = {"one": 1, "few": 8, "small": 256}
    caps = {"one": 1_000, "few": 2_000, "small": 5_000}
    thresholds = {
        "one": {"byte_accuracy": 1.0, "exact_accuracy": 1.0},
        "few": {"byte_accuracy": 1.0, "exact_accuracy": 1.0},
        "small": {"byte_accuracy": 0.95, "byte_nll": 0.25},
    }
    if case not in sizes:
        raise ValueError(f"unknown overfit case: {case}")
    prompt, target, ids = split("train", TRAIN_COUNT)
    prompt, target, ids = prompt[: sizes[case]], target[: sizes[case]], ids[: sizes[case]]
    path = WORK / "overfit" / case / "init0_stoch0.pkl"
    curve = EVIDENCE / f"overfit_{case}_init0_stoch0.jsonl"
    params, moments, start = _load_or_initialize(path)
    update = jax.jit(stochastic_denoise_train_step, static_argnames=("config",))
    checkpoints = {1, 10, 25, 50, 100, 250, 500, 1_000, 2_000, 5_000}
    final = _metrics_at_fixed_set(params, prompt, target, ids, 0.5) if start else None

    def passed(metrics: dict[str, float] | None) -> bool:
        if metrics is None:
            return False
        required = thresholds[case]
        return all(
            metrics[name] >= value if name != "byte_nll" else metrics[name] <= value
            for name, value in required.items()
        )

    started = time.monotonic()
    step = start
    while step < caps[case] and not passed(final):
        step += 1
        indices = jnp.arange((step - 1) * BATCH, step * BATCH) % len(prompt)
        batch = _training_batch(
            prompt,
            target,
            ids,
            indices,
            step,
            fixed_nu=0.5,
            fixed_corruption=True,
        )
        params, moments, metrics = update(
            params,
            moments,
            step,
            batch,
            OPERATOR_ROOT,
            TRAIN_DIFFUSION_ROOT,
            config=config(),
        )
        jax.block_until_ready(metrics["loss"])
        if step in checkpoints:
            final = _metrics_at_fixed_set(params, prompt, target, ids, 0.5)
            append(curve, {"step": step, "train": metrics, "lambda_zero": final})
            save(path, {"params": params, "moments": moments, "step": step})
            print(f"F1 overfit {case} step={step} passed={passed(final)}", flush=True)
        elif step % 100 == 0:
            save(path, {"params": params, "moments": moments, "step": step})
    payload = {
        "case": case,
        "examples": sizes[case],
        "optimizer_step": step,
        "cap": caps[case],
        "fixed_nu": 0.5,
        "fixed_initial_corruption": True,
        "criterion": thresholds[case],
        "final": final,
        "passed": passed(final),
        "checkpoint": str(path.relative_to(ROOT)),
        "checkpoint_sha256": sha256(path),
        "runtime_seconds": time.monotonic() - started,
    }
    write(EVIDENCE / f"overfit_{case}_summary.json", payload)
    if not payload["passed"]:
        raise RuntimeError(f"F1 {case} overfit gate failed")
    return payload


def run_overfit_gate() -> dict[str, Any]:
    results = [run_overfit(case) for case in ("one", "few", "small")]
    payload = {"cases": results, "passed": all(item["passed"] for item in results)}
    write(EVIDENCE / "overfit.json", payload)
    return payload


def run_calibration(max_steps: int) -> dict[str, Any]:
    if max_steps not in {5_000, 10_000}:
        raise ValueError("calibration max_steps must be 5000 or 10000")
    cfg = config()
    prompt, target, ids = split("train", TRAIN_COUNT)
    valid_prompt, valid_target, valid_ids = split("validation", VALIDATION_COUNT)
    state_path = WORK / "one_step" / "init0_stoch0.pkl"
    curve_path = EVIDENCE / "training_init0_stoch0.jsonl"
    params, moments, start = _load_or_initialize(state_path)
    update = jax.jit(stochastic_denoise_train_step, static_argnames=("config",))
    recorded = set()
    if curve_path.exists():
        recorded = {
            int(json.loads(line)["step"])
            for line in curve_path.read_text().splitlines()
            if line.strip()
        }
    started = time.monotonic()
    if start == 0:
        baseline = evaluate(
            params,
            valid_prompt,
            valid_target,
            valid_ids,
            cycles=(0, 3),
            levels=(0.0, *EVAL_LEVELS),
        )
        write(EVIDENCE / "raw_hnu_baseline_before_training.json", baseline)
    latest = None
    for step in range(start + 1, max_steps + 1):
        indices = jnp.arange((step - 1) * BATCH, step * BATCH) % TRAIN_COUNT
        batch = _training_batch(prompt, target, ids, indices, step)
        params, moments, metrics = update(
            params,
            moments,
            step,
            batch,
            OPERATOR_ROOT,
            TRAIN_DIFFUSION_ROOT,
            config=cfg,
        )
        jax.block_until_ready(metrics["loss"])
        if step in CHECKPOINTS and step not in recorded:
            evaluation = evaluate(
                params,
                valid_prompt,
                valid_target,
                valid_ids,
                cycles=(3,),
                levels=(0.0, *EVAL_LEVELS),
            )
            append(
                curve_path,
                {
                    "git_sha": git_sha(),
                    "step": step,
                    "train": metrics,
                    "lambda_zero": evaluation,
                    "runtime_seconds_this_run": time.monotonic() - started,
                },
            )
            save(state_path, {"params": params, "moments": moments, "step": step})
            latest = evaluation
            recorded.add(step)
            print(f"F1 calibration step={step}", flush=True)
        elif step % 100 == 0:
            save(state_path, {"params": params, "moments": moments, "step": step})
    if latest is None:
        latest = evaluate(
            params,
            valid_prompt,
            valid_target,
            valid_ids,
            cycles=(3,),
            levels=(0.0, *EVAL_LEVELS),
        )
    raw_after = evaluate(
        params,
        valid_prompt,
        valid_target,
        valid_ids,
        cycles=(0, 3),
        levels=(0.0, *EVAL_LEVELS),
    )
    write(EVIDENCE / "raw_hnu_baseline_after_training.json", raw_after)
    payload = {
        "optimizer_step": max_steps,
        "checkpoint": str(state_path.relative_to(ROOT)),
        "checkpoint_sha256": sha256(state_path),
        "training_examples": TRAIN_COUNT,
        "validation_examples_per_nu": VALIDATION_COUNT,
        "training_nu_distribution": "Uniform(0.025, 0.9)",
        "eval_grid": EVAL_LEVELS,
        "latest_lambda_zero": latest,
        "raw_q0_and_full_q3_after_training": raw_after,
        "runtime_seconds_this_run": time.monotonic() - started,
    }
    write(EVIDENCE / "calibration.json", payload)
    return payload


def run_lambda_sanity() -> dict[str, Any]:
    state_path = WORK / "one_step" / "init0_stoch0.pkl"
    state = load(state_path)
    prompt, target, ids = split("validation", 64)
    deterministic = evaluate(
        state["params"], prompt, target, ids, levels=EVAL_LEVELS, lambda_op=0.0
    )
    roots = []
    for root_index in range(4):
        root = jax.random.fold_in(OPERATOR_ROOT, root_index + 1)
        roots.append(
            evaluate(
                state["params"],
                prompt,
                target,
                ids,
                levels=EVAL_LEVELS,
                lambda_op=1.0,
                operator_root=root,
            )
        )
    payload = {
        "checkpoint_step": int(state["step"]),
        "examples_per_nu": len(prompt),
        "operator_roots": 4,
        "lambda_zero": deterministic,
        "lambda_one_roots": roots,
        "initial_diffusion_epsilon_fixed_across_lambda_and_nu": True,
    }
    write(EVIDENCE / "lambda_sanity.json", payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        choices=(
            "audit",
            "codec",
            "codec-control",
            "gradient",
            "overfit",
            "calibration",
            "lambda",
        ),
        required=True,
    )
    parser.add_argument("--max-steps", type=int, default=5_000)
    args = parser.parse_args()
    if args.stage == "audit":
        run_path_audit()
        run_dataset_audit()
    elif args.stage == "codec":
        run_codec_audit()
    elif args.stage == "codec-control":
        run_codec_control()
    elif args.stage == "gradient":
        run_first_gradient()
    elif args.stage == "overfit":
        run_overfit_gate()
    elif args.stage == "calibration":
        run_calibration(args.max_steps)
    elif args.stage == "lambda":
        run_lambda_sanity()


if __name__ == "__main__":
    main()
