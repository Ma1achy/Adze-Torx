"""Preparation-only Phase-I corpus and full-byte codec candidate driver."""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from pathlib import Path
from typing import Any

import jax

from adze_t.config import REFERENCE_SMALL_V0
from adze_t.language_data import (
    ByteSegment,
    SegmentationConfig,
    SplitConfig,
    build_corpus_splits,
    read_text_documents,
    segment_corpus,
    split_manifest,
    validate_corpus_artifacts,
    write_segments_jsonl,
)
from adze_t.phase_i import (
    CodecCandidateConfig,
    SmallScaleLMRecipe,
    candidate_manifest,
    codec_batches,
    full_byte_calibration_segments,
    full_byte_codec_train_step,
    full_byte_codec_validation,
    initialise_full_byte_codec_candidate,
    validate_recipe,
    write_json,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare(args: argparse.Namespace) -> None:
    documents = read_text_documents(
        args.inputs,
        format=args.format,
        text_field=args.text_field,
        normalization=args.normalization,
    )
    segmentation = SegmentationConfig(args.sequence_bytes, args.stride_bytes, args.min_bytes)
    split_config = SplitConfig(
        args.seed, args.train_fraction, args.validation_fraction, args.test_fraction
    )
    segments = segment_corpus(documents, segmentation)
    splits = build_corpus_splits(segments, split_config)
    output = Path(args.output)
    for name in ("train", "validation", "test"):
        write_segments_jsonl(output / f"{name}.jsonl", splits.named(name))
    manifest = split_manifest(documents, segments, splits, segmentation, split_config)
    write_json(output / "manifest.json", manifest)


def _save_state(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    with temporary.open("wb") as stream:
        pickle.dump(jax.device_get(state), stream, protocol=pickle.HIGHEST_PROTOCOL)
    temporary.replace(path)


def _evaluate_segments(
    evaluate: Any,
    params: Any,
    segments: tuple[ByteSegment, ...],
    *,
    sequence_bytes: int,
    batch_size: int,
) -> dict[str, Any]:
    if not segments:
        raise RuntimeError("codec validation split is empty")
    totals = {"codec_byte": 0.0, "codec_b": 0.0, "codec_l": 0.0, "sequence": 0.0}
    per_byte_count = None
    per_byte_correct = None
    examples = 0
    nonfinite = False
    for batch in codec_batches(
        segments, width=sequence_bytes, batch_size=batch_size, seed=0, epoch=0
    ):
        metrics = evaluate(params, batch, config=REFERENCE_SMALL_V0)
        count = len(batch["segment_ids"])
        examples += count
        for name in ("codec_byte", "codec_b", "codec_l"):
            totals[name] += float(metrics[name]) * count
        totals["sequence"] += float(metrics["sequence_accuracy"]) * count
        counts = jax.device_get(metrics["per_byte_count"])
        correct = jax.device_get(metrics["per_byte_correct"])
        per_byte_count = counts if per_byte_count is None else per_byte_count + counts
        per_byte_correct = correct if per_byte_correct is None else per_byte_correct + correct
        nonfinite = nonfinite or bool(metrics["nonfinite"])
    assert per_byte_count is not None and per_byte_correct is not None
    covered = per_byte_count > 0
    accuracies = per_byte_correct / per_byte_count.clip(min=1)
    return {
        "segments": examples,
        "codec_byte": totals["codec_byte"] / examples,
        "codec_b": totals["codec_b"] / examples,
        "codec_l": totals["codec_l"] / examples,
        "byte_accuracy": float(per_byte_correct.sum() / per_byte_count.sum()),
        "sequence_accuracy": totals["sequence"] / examples,
        "covered_byte_values": int(covered.sum()),
        "minimum_covered_byte_accuracy": float(accuracies[covered].min()),
        "nonfinite": nonfinite,
    }


def train_codec(args: argparse.Namespace) -> None:
    """Train only a fresh, versioned codec candidate; never read Phase-B weights."""
    dataset = Path(args.dataset)
    if not 1 <= args.sequence_bytes <= REFERENCE_SMALL_V0.model.target_max_bytes:
        raise ValueError(
            "sequence_bytes exceeds the reference-small target codec capacity "
            f"({REFERENCE_SMALL_V0.model.target_max_bytes})"
        )
    manifest, splits = validate_corpus_artifacts(dataset)
    candidate = CodecCandidateConfig(initialization_seed=args.initialization_seed)
    binding = candidate_manifest(manifest, candidate)
    output = Path(candidate.checkpoint_directory)
    output.mkdir(parents=True, exist_ok=True)
    binding_path = output / "manifest.json"
    if binding_path.exists() and json.loads(binding_path.read_text()) != binding:
        raise RuntimeError("codec candidate provenance differs from existing run")
    write_json(binding_path, binding)
    train = splits.train + full_byte_calibration_segments(args.sequence_bytes)
    validation = splits.validation
    latest = output / "latest.pkl"
    if latest.exists():
        with latest.open("rb") as stream:
            state = pickle.load(stream)
        if state["manifest_sha256"] != binding["manifest_sha256"]:
            raise RuntimeError("saved codec state has stale or mixed provenance")
        params, moments, start = state["params"], state["moments"], int(state["step"])
    else:
        params, moments = initialise_full_byte_codec_candidate(REFERENCE_SMALL_V0, candidate)
        start = 0
    update = jax.jit(full_byte_codec_train_step, static_argnames=("config",))
    evaluate = jax.jit(full_byte_codec_validation, static_argnames=("config",))
    rows = output / "training.jsonl"
    steps_per_epoch = max(1, (len(train) + args.batch_size - 1) // args.batch_size)
    for step in range(start + 1, args.max_steps + 1):
        epoch, batch_index = divmod(step - 1, steps_per_epoch)
        batches = codec_batches(
            train,
            width=args.sequence_bytes,
            batch_size=args.batch_size,
            seed=args.training_seed,
            epoch=epoch,
        )
        batch = batches[batch_index % len(batches)]
        params, moments, metrics = update(params, moments, step, batch, config=REFERENCE_SMALL_V0)
        if step % args.checkpoint_every == 0 or step == args.max_steps:
            jax.block_until_ready(metrics["loss"])
            calibration = codec_batches(
                full_byte_calibration_segments(args.sequence_bytes),
                width=args.sequence_bytes,
                batch_size=256,
                seed=0,
                epoch=0,
            )[0]
            calibration_metrics = evaluate(params, calibration, config=REFERENCE_SMALL_V0)
            validation_metrics = _evaluate_segments(
                evaluate,
                params,
                validation,
                sequence_bytes=args.sequence_bytes,
                batch_size=args.batch_size,
            )
            row = {
                "step": step,
                "loss": float(metrics["loss"]),
                "byte_accuracy": float(metrics["byte_accuracy"]),
                "sequence_accuracy": float(metrics["sequence_accuracy"]),
                "gradient_norm": float(metrics["grad_norm"]),
                "calibration_byte_accuracy": float(calibration_metrics["byte_accuracy"]),
                "calibration_minimum_byte_accuracy": float(
                    calibration_metrics["minimum_covered_byte_accuracy"]
                ),
                "calibration_full_byte_coverage": bool(calibration_metrics["full_byte_coverage"]),
                "validation": validation_metrics,
            }
            with rows.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(row, sort_keys=True) + "\n")
            state = {
                "candidate_version": candidate.version,
                "manifest_sha256": binding["manifest_sha256"],
                "step": step,
                "params": params,
                "moments": moments,
            }
            _save_state(latest, state)
            _save_state(output / f"step_{step:08d}.pkl", state)


def recipe(args: argparse.Namespace) -> None:
    manifest, _ = validate_corpus_artifacts(Path(args.dataset))
    value = SmallScaleLMRecipe(
        name=args.name,
        scale=args.scale,
        dataset_manifest_sha256=str(manifest["manifest_sha256"]),
        codec_version=CodecCandidateConfig().version,
        seed=args.seed,
    )
    validate_recipe(value)
    write_json(args.output, value.__dict__)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    subparsers = result.add_subparsers(dest="stage", required=True)
    data = subparsers.add_parser("prepare")
    data.add_argument("inputs", nargs="+")
    data.add_argument("--format", choices=("text", "jsonl"), default="text")
    data.add_argument("--text-field", default="text")
    data.add_argument("--normalization", choices=("none", "lf", "nfc"), default="none")
    data.add_argument("--sequence-bytes", type=int, default=128)
    data.add_argument("--stride-bytes", type=int, default=128)
    data.add_argument("--min-bytes", type=int, default=16)
    data.add_argument("--seed", type=int, default=9_200)
    data.add_argument("--train-fraction", type=float, default=0.98)
    data.add_argument("--validation-fraction", type=float, default=0.01)
    data.add_argument("--test-fraction", type=float, default=0.01)
    data.add_argument("--output", default="results/phase_i/data/natural_text_v0")
    codec = subparsers.add_parser("codec-train")
    codec.add_argument("--dataset", required=True)
    codec.add_argument("--sequence-bytes", type=int, default=128)
    codec.add_argument("--batch-size", type=int, default=32)
    codec.add_argument("--max-steps", type=int, default=10_000)
    codec.add_argument("--checkpoint-every", type=int, default=100)
    codec.add_argument("--initialization-seed", type=int, default=9_100)
    codec.add_argument("--training-seed", type=int, default=9_101)
    plan = subparsers.add_parser("recipe")
    plan.add_argument("--dataset", required=True)
    plan.add_argument("--name", default="phase_i_small_scale_v0")
    plan.add_argument("--scale", choices=("smoke", "small", "medium"), default="smoke")
    plan.add_argument("--seed", type=int, default=9_300)
    plan.add_argument("--output", default="results/phase_i/recipes/small_scale_v0.json")
    return result


def main() -> None:
    args = parser().parse_args()
    if args.stage == "prepare":
        prepare(args)
    elif args.stage == "codec-train":
        train_codec(args)
    else:
        recipe(args)


if __name__ == "__main__":
    main()
