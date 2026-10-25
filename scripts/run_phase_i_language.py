"""Phase-I corpus preparation and separately versioned codec-candidate driver.

Real codec execution remains separately authorized. This runner defines the
fail-closed provenance, incremental loading, and crash-recovery path that a
later authorized run must use.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import subprocess
from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence

import jax
import numpy as np

from adze_t.language_data import (
    ByteSegment,
    SegmentationConfig,
    SplitConfig,
    prepare_corpus,
    read_binary_documents,
    read_text_documents,
    validate_corpus_artifacts,
    write_corpus_artifacts,
)
from adze_t.phase_i import (
    CODEC_RUN_PROTOCOL_V0,
    FULL_BYTE_CODEC_V0,
    PINNED_TORX_REVISION,
    ByteEvidence,
    CodecCandidateConfig,
    CodecRunProtocol,
    RuntimeIdentity,
    SmallScaleLMRecipe,
    byte_histogram,
    calibration_prefixed_batch,
    candidate_manifest,
    codec_state_tree_description,
    evaluation_coverage_evidence,
    full_byte_calibration_segments,
    full_byte_codec_train_step,
    full_byte_codec_validation,
    initialise_full_byte_codec_candidate,
    iter_codec_batches,
    load_language_scales,
    reconstruction_evidence,
    training_exposure_evidence,
    validate_codec_resume_state,
    validate_recipe,
    validate_run_protocol,
    write_json,
)
from adze_t.phase_i_artifacts import (
    checkpoint_tree_identity,
    load_checkpoint_state,
    reconcile_transactions,
    recover_transactions,
    save_transaction,
)


CALIBRATION_POLICY_V0 = "full_byte_prefix_each_epoch_v0"
CODEC_EVALUATION_PROTOCOL_V0 = "phase_i_codec_validation_v0"
STRUCTURE_PROTOCOL = "phase_i_supervised_teacher_structure_only_v0"
REFINEMENT_PROTOCOL = "disabled_r0"


def _source_identity(root: Path) -> tuple[str, str]:
    status = subprocess.run(
        ["git", "status", "--porcelain"], cwd=root, check=True, capture_output=True, text=True
    ).stdout
    if status:
        raise RuntimeError("codec execution requires a clean committed source tree")
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, check=True, capture_output=True, text=True
    ).stdout.strip()
    relevant = (
        root / "src/adze_t/language_data.py",
        root / "src/adze_t/phase_i.py",
        root / "src/adze_t/phase_i_artifacts.py",
        root / "scripts/run_phase_i_language.py",
        root / "configs/phase_i_language_scales_v0.yaml",
    )
    digest = hashlib.sha256()
    for path in relevant:
        digest.update(path.relative_to(root).as_posix().encode() + b"\0")
        digest.update(path.read_bytes())
    return commit, digest.hexdigest()


def _runtime_identity() -> RuntimeIdentity:
    try:
        torx_version = importlib.metadata.version("torx")
    except importlib.metadata.PackageNotFoundError:
        torx_version = "unavailable"
    return RuntimeIdentity(
        python=platform.python_version(),
        platform=platform.platform(),
        jax=jax.__version__,
        torx=torx_version,
        torx_revision=PINNED_TORX_REVISION,
        backend=jax.default_backend(),
        x64_enabled=bool(jax.config.read("jax_enable_x64")),
    )


def _prepare(args: argparse.Namespace) -> None:
    if args.protocol == "binary" and args.format != "text":
        raise ValueError("raw-binary preparation does not accept a text/JSONL format")
    documents = (
        read_text_documents(args.inputs, format=args.format, text_field=args.text_field)
        if args.protocol == "text"
        else read_binary_documents(args.inputs)
    )
    prepared = prepare_corpus(
        documents,
        SegmentationConfig(args.sequence_bytes, args.stride_bytes, args.min_bytes),
        SplitConfig(args.seed, args.train_fraction, args.validation_fraction, args.test_fraction),
    )
    write_corpus_artifacts(args.output, prepared)


def _add_counts(left: Sequence[int], right: Sequence[int]) -> tuple[int, ...]:
    return tuple(a + b for a, b in zip(left, right, strict=True))


def _evaluate_segments(
    evaluate: Any,
    params: Any,
    segments: Sequence[ByteSegment],
    *,
    config: Any,
    sequence_bytes: int,
    batch_size: int,
) -> tuple[dict[str, Any], ByteEvidence, ByteEvidence]:
    if not segments:
        raise RuntimeError("codec evaluation split is empty")
    loss_sums = {name: 0.0 for name in ("codec_byte", "codec_b", "codec_l")}
    supports = {name: 0 for name in ("codec_byte", "codec_b", "codec_l")}
    sequence_correct_sum = 0.0
    sequence_support = 0
    counts = np.zeros(256, dtype=np.int64)
    correct = np.zeros(256, dtype=np.int64)
    nonfinite = False
    for batch in iter_codec_batches(
        segments, width=sequence_bytes, batch_size=batch_size, seed=0, epoch=0
    ):
        metrics = evaluate(params, batch, config=config)
        for name in loss_sums:
            support = int(metrics[f"{name}_support"])
            loss_sums[name] += float(metrics[f"{name}_loss_sum"])
            supports[name] += support
        batch_sequences = int(metrics["sequence_support"])
        sequence_correct_sum += float(metrics["sequence_accuracy"]) * batch_sequences
        sequence_support += batch_sequences
        counts += np.asarray(jax.device_get(metrics["per_byte_count"]), dtype=np.int64)
        correct += np.asarray(jax.device_get(metrics["per_byte_correct"]), dtype=np.int64)
        nonfinite = nonfinite or bool(metrics["nonfinite"])
    coverage = evaluation_coverage_evidence(counts.tolist())
    reconstruction = reconstruction_evidence(counts.tolist(), correct.tolist())
    result = {
        **{name: loss_sums[name] / max(supports[name], 1) for name in loss_sums},
        **{f"{name}_support": supports[name] for name in supports},
        "byte_accuracy": float(correct.sum() / max(counts.sum(), 1)),
        "sequence_accuracy": sequence_correct_sum / max(sequence_support, 1),
        "sequence_support": sequence_support,
        "nonfinite": nonfinite,
    }
    return result, coverage, reconstruction


def _protocol(
    args: argparse.Namespace, dataset_manifest: dict[str, object], scale: Any
) -> CodecRunProtocol:
    root = Path(__file__).resolve().parents[1]
    source_commit, implementation_hash = _source_identity(root)
    config = scale.reference
    return CodecRunProtocol(
        version=CODEC_RUN_PROTOCOL_V0,
        source_commit=source_commit,
        implementation_sha256=implementation_hash,
        dataset_manifest_sha256=str(dataset_manifest["manifest_sha256"]),
        dataset_protocol=str(dataset_manifest["protocol"]),
        scale_name=scale.name,
        resolved_config_sha256=scale.resolved_config_sha256,
        candidate_version=FULL_BYTE_CODEC_V0,
        initialization_seed=args.initialization_seed,
        training_seed=args.training_seed,
        sequence_bytes=config.model.target_max_bytes,
        batch_size=scale.batch_size,
        gradient_accumulation=scale.gradient_accumulation,
        max_updates=args.max_updates,
        checkpoint_every_updates=args.checkpoint_every,
        validation_every_updates=args.checkpoint_every,
        optimizer="adamw",
        learning_rate=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
        grad_clip_norm=config.training.grad_clip_norm,
        schedule="constant",
        calibration_policy=CALIBRATION_POLICY_V0,
        evaluation_protocol=CODEC_EVALUATION_PROTOCOL_V0,
        structure_protocol=STRUCTURE_PROTOCOL,
        refinement_protocol=REFINEMENT_PROTOCOL,
        q=0,
        s=0,
        r=0,
        runtime=_runtime_identity(),
    )


def _train_codec(args: argparse.Namespace) -> None:
    dataset_manifest, splits = validate_corpus_artifacts(args.dataset)
    scales = load_language_scales(args.scales)
    scale = scales[args.scale]
    if scale.status != "executable_engineering_candidate":
        raise RuntimeError(f"scale {args.scale} is not approved for engineering execution")
    protocol = _protocol(args, dataset_manifest, scale)
    protocol_hash = validate_run_protocol(protocol)
    candidate = CodecCandidateConfig(initialization_seed=args.initialization_seed)
    binding = candidate_manifest(dataset_manifest, scale, protocol_hash, candidate)
    output = Path(candidate.checkpoint_directory) / protocol_hash
    if output.exists() and json.loads((output / "run_protocol.json").read_text()) != asdict(
        protocol
    ):
        raise RuntimeError("candidate directory contains a different run protocol")
    write_json(output / "run_protocol.json", asdict(protocol))
    binding_path = output / "candidate_manifest.json"
    if binding_path.exists() and json.loads(binding_path.read_text()) != binding:
        raise RuntimeError("candidate manifest differs under the same run protocol")
    write_json(binding_path, binding)

    resume = reconcile_transactions(output, protocol_sha256=protocol_hash)
    if resume.incomplete_steps:
        raise RuntimeError(
            "incomplete candidate transactions require quarantine before resume: "
            f"{resume.incomplete_steps}"
        )
    if resume.checkpoint_record is None:
        params, moments = initialise_full_byte_codec_candidate(scale.reference, candidate)
        start = 0
        exposure = (0,) * 256
        tree_description = codec_state_tree_description(params, moments)
    else:
        state = load_checkpoint_state(output, resume.checkpoint_record)
        start = resume.last_complete_step
        params, moments, exposure, tree_description = validate_codec_resume_state(
            state,
            protocol_sha256=protocol_hash,
            expected_step=start,
            expected_tree_sha256=resume.checkpoint_record.tree_identity_sha256,
        )
    tree_hash = checkpoint_tree_identity(tree_description)
    calibration = full_byte_calibration_segments(scale.reference.model.target_max_bytes)
    update = jax.jit(full_byte_codec_train_step, static_argnames=("config",))
    evaluate = jax.jit(full_byte_codec_validation, static_argnames=("config",))
    for step in range(start + 1, args.max_updates + 1):
        batch, epoch = calibration_prefixed_batch(
            splits.train,
            calibration,
            width=scale.reference.model.target_max_bytes,
            batch_size=scale.batch_size,
            seed=args.training_seed,
            update=step,
        )
        exposure = _add_counts(
            exposure,
            byte_histogram(
                np.asarray(jax.device_get(batch["target"])),
                np.asarray(jax.device_get(batch["target_mask"])),
            ),
        )
        params, moments, train_metrics = update(
            params, moments, step, batch, config=scale.reference
        )
        if step % args.checkpoint_every != 0 and step != args.max_updates:
            continue
        jax.block_until_ready(train_metrics["loss"])
        validation, validation_coverage, validation_reconstruction = _evaluate_segments(
            evaluate,
            params,
            splits.validation,
            config=scale.reference,
            sequence_bytes=scale.reference.model.target_max_bytes,
            batch_size=scale.batch_size,
        )
        test, test_coverage, test_reconstruction = _evaluate_segments(
            evaluate,
            params,
            splits.test,
            config=scale.reference,
            sequence_bytes=scale.reference.model.target_max_bytes,
            batch_size=scale.batch_size,
        )
        calibration_metrics, calibration_coverage, calibration_reconstruction = _evaluate_segments(
            evaluate,
            params,
            calibration,
            config=scale.reference,
            sequence_bytes=scale.reference.model.target_max_bytes,
            batch_size=scale.batch_size,
        )
        metrics = {
            "step": step,
            "epoch": epoch,
            "train": {name: float(value) for name, value in train_metrics.items()},
            "validation": validation,
            "test": test,
            "full_byte_calibration": calibration_metrics,
        }
        state = {
            "schema": "PHASE_I_CODEC_STATE_V0",
            "protocol_sha256": protocol_hash,
            "step": step,
            "params": params,
            "moments": moments,
            "tree_description": tree_description,
            "training_exposure_counts": exposure,
        }
        save_transaction(
            output,
            step=step,
            protocol_sha256=protocol_hash,
            tree_identity_sha256=tree_hash,
            state=jax.device_get(state),
            metrics=metrics,
            training_exposure=asdict(training_exposure_evidence(exposure)),
            evaluation_coverage={
                "validation": asdict(validation_coverage),
                "test": asdict(test_coverage),
                "full_byte_calibration": asdict(calibration_coverage),
            },
            reconstruction={
                "validation": asdict(validation_reconstruction),
                "test": asdict(test_reconstruction),
                "full_byte_calibration": asdict(calibration_reconstruction),
            },
        )


def _recipe(args: argparse.Namespace) -> None:
    recipe = SmallScaleLMRecipe(
        name=args.name,
        run_protocol_sha256=args.run_protocol_sha256,
        codec_checkpoint_sha256=args.codec_checkpoint_sha256,
        codec_checkpoint_step=args.codec_checkpoint_step,
        evaluation_protocol=args.evaluation_protocol,
    )
    validate_recipe(recipe)
    write_json(args.output, asdict(recipe))


def _recover(args: argparse.Namespace) -> None:
    result = recover_transactions(args.run_directory, protocol_sha256=args.protocol_sha256)
    write_json(Path(args.run_directory) / "last_recovery.json", asdict(result))


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    subparsers = result.add_subparsers(dest="stage", required=True)
    data = subparsers.add_parser("prepare")
    data.add_argument("inputs", nargs="+")
    data.add_argument("--protocol", choices=("text", "binary"), required=True)
    data.add_argument("--format", choices=("text", "jsonl"), default="text")
    data.add_argument("--text-field", default="text")
    data.add_argument("--sequence-bytes", type=int, default=128)
    data.add_argument("--stride-bytes", type=int, default=128)
    data.add_argument("--min-bytes", type=int, default=1)
    data.add_argument("--seed", type=int, default=9_200)
    data.add_argument("--train-fraction", type=float, default=0.98)
    data.add_argument("--validation-fraction", type=float, default=0.01)
    data.add_argument("--test-fraction", type=float, default=0.01)
    data.add_argument("--output", default="results/phase_i/data/candidate_v0")

    codec = subparsers.add_parser("codec-train")
    codec.add_argument("--dataset", required=True)
    codec.add_argument("--scales", default=str(Path("configs/phase_i_language_scales_v0.yaml")))
    codec.add_argument("--scale", default="smoke")
    codec.add_argument("--max-updates", type=int, required=True)
    codec.add_argument("--checkpoint-every", type=int, default=100)
    codec.add_argument("--initialization-seed", type=int, default=9_100)
    codec.add_argument("--training-seed", type=int, default=9_101)

    plan = subparsers.add_parser("recipe")
    plan.add_argument("--name", default="phase_i_small_scale_v0")
    plan.add_argument("--run-protocol-sha256", required=True)
    plan.add_argument("--codec-checkpoint-sha256", required=True)
    plan.add_argument("--codec-checkpoint-step", type=int, required=True)
    plan.add_argument("--evaluation-protocol", required=True)
    plan.add_argument("--output", default="results/phase_i/recipes/small_scale_v0.json")
    recovery = subparsers.add_parser("recover")
    recovery.add_argument("--run-directory", required=True)
    recovery.add_argument("--protocol-sha256", required=True)
    return result


def main() -> None:
    args = parser().parse_args()
    if args.stage == "prepare":
        _prepare(args)
    elif args.stage == "codec-train":
        _train_codec(args)
    elif args.stage == "recipe":
        _recipe(args)
    else:
        _recover(args)


if __name__ == "__main__":
    main()
