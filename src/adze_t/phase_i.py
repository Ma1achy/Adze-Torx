"""Phase-I preparation contracts for a separately versioned full-byte codec."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from collections.abc import Mapping
from typing import Any, Iterable, Sequence

import jax
import jax.numpy as jnp

from .config import REFERENCE_SMALL_V0, ReferenceConfig
from .language_data import ByteSegment, pad_segments
from .model import apply_target_codec, init_model_params
from .objectives import codec_loss_components, emitted_metrics
from .training import codec_pretrain_step


FULL_BYTE_CODEC_V0 = "PHASE_I_FULL_BYTE_CODEC_V0"
NATURAL_TEXT_PIPELINE_V0 = "PHASE_I_NATURAL_TEXT_BYTES_V0"
ACCEPTED_PHASE_B_CODEC_PATH = "results/phase_b/checkpoints/target_codec_b1.pkl"
FULL_BYTE_CODEC_CHECKPOINT_DIR = "results/phase_i/candidates/full_byte_codec_v0"


@dataclass(frozen=True)
class CodecCandidateConfig:
    version: str = FULL_BYTE_CODEC_V0
    initialization_seed: int = 9_100
    byte_domain_min: int = 0
    byte_domain_max: int = 255
    checkpoint_directory: str = FULL_BYTE_CODEC_CHECKPOINT_DIR
    accepted_codec_parent: None = None


@dataclass(frozen=True)
class LanguageScaleConfig:
    name: str
    sequence_bytes: int
    carrier_capacity: int
    max_extent: int
    d_model: int
    physical_blocks: int
    cycles: int
    batch_size: int
    gradient_accumulation: int

    def __post_init__(self) -> None:
        if self.sequence_bytes > self.carrier_capacity * self.max_extent:
            raise ValueError("language sequence exceeds carrier emission capacity")
        if (
            min(
                self.sequence_bytes,
                self.carrier_capacity,
                self.max_extent,
                self.d_model,
                self.physical_blocks,
                self.cycles,
                self.batch_size,
                self.gradient_accumulation,
            )
            <= 0
        ):
            raise ValueError("scale dimensions must be positive")


@dataclass(frozen=True)
class SmallScaleLMRecipe:
    name: str
    scale: str
    dataset_manifest_sha256: str
    codec_version: str
    seed: int
    optimizer: str = "adamw"
    learning_rate: float = 3.0e-4
    weight_decay: float = 0.01
    grad_clip_norm: float = 1.0
    validation_every_updates: int = 100
    checkpoint_every_updates: int = 100
    authoritative: bool = False


LANGUAGE_SCALES = {
    "smoke": LanguageScaleConfig("smoke", 128, 32, 4, 128, 4, 3, 8, 1),
    "small": LanguageScaleConfig("small", 256, 64, 4, 256, 6, 3, 16, 2),
    "medium": LanguageScaleConfig("medium", 512, 128, 4, 384, 8, 4, 16, 4),
}


def candidate_manifest(
    dataset_manifest: Mapping[str, object],
    config: CodecCandidateConfig = CodecCandidateConfig(),
) -> dict[str, Any]:
    """Bind a fresh codec candidate to data without inheriting Phase-B weights."""
    payload: dict[str, Any] = {
        "candidate": asdict(config),
        "dataset_manifest_sha256": dataset_manifest.get("manifest_sha256"),
        "dataset_protocol": dataset_manifest.get("protocol"),
        "byte_vocab": 256,
        "target_codec_topology": "shared_frontend_target_analysis_mamba_decoder",
        "initialization": "fresh_seeded_parameters",
        "accepted_phase_b_codec": {
            "path": ACCEPTED_PHASE_B_CODEC_PATH,
            "policy": "immutable_not_a_parent",
        },
        "authoritative_training_allowed": False,
    }
    payload["manifest_sha256"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return payload


def full_byte_calibration_segments(sequence_bytes: int = 128) -> tuple[ByteSegment, ...]:
    """Create deterministic codec-only calibration data covering every byte."""
    if not 1 <= sequence_bytes <= 256:
        raise ValueError("sequence_bytes must be in [1, 256]")
    values = bytes(range(256))
    result = []
    for start in range(0, 256, sequence_bytes):
        content = values[start : start + sequence_bytes]
        digest = hashlib.sha256(content).hexdigest()
        result.append(
            ByteSegment(
                segment_id=f"full-byte-{start:03d}-{digest[:16]}",
                document_id="phase-i-full-byte-calibration-v0",
                source="generated://phase-i/full-byte-calibration-v0",
                start_byte=start,
                end_byte=start + len(content),
                content_sha256=digest,
                values=tuple(content),
            )
        )
    return tuple(result)


def deterministic_epoch_order(
    segment_ids: Iterable[str], *, seed: int, epoch: int
) -> tuple[int, ...]:
    """Shuffle by keyed hashes without ambient Python/NumPy RNG state."""
    keyed = [
        (hashlib.sha256(f"{seed}:{epoch}:{identifier}".encode()).digest(), index)
        for index, identifier in enumerate(segment_ids)
    ]
    return tuple(index for _, index in sorted(keyed))


def codec_batches(
    segments: Sequence[ByteSegment],
    *,
    width: int,
    batch_size: int,
    seed: int,
    epoch: int,
) -> tuple[dict[str, Any], ...]:
    if not segments:
        raise ValueError("codec training requires at least one segment")
    order = deterministic_epoch_order(
        (segment.segment_id for segment in segments), seed=seed, epoch=epoch
    )
    batches = []
    for offset in range(0, len(order), batch_size):
        selected = [segments[index] for index in order[offset : offset + batch_size]]
        values, mask, ids = pad_segments(selected, width)
        batches.append(
            {
                "target": jnp.asarray(values),
                "target_mask": jnp.asarray(mask),
                "segment_ids": ids,
            }
        )
    return tuple(batches)


def initialise_full_byte_codec_candidate(
    config: ReferenceConfig = REFERENCE_SMALL_V0,
    candidate: CodecCandidateConfig = CodecCandidateConfig(),
) -> tuple[Any, tuple[Any, Any]]:
    """Fresh initialization; the accepted Phase-B/F codec is never loaded."""
    params = init_model_params(jax.random.PRNGKey(candidate.initialization_seed), config)
    first_moments = jax.tree_util.tree_map(jnp.zeros_like, params)
    second_moments = jax.tree_util.tree_map(jnp.zeros_like, params)
    return params, (first_moments, second_moments)


def full_byte_codec_train_step(
    params: Any,
    moments: tuple[Any, Any],
    step: int | jax.Array,
    batch: dict[str, jax.Array],
    *,
    config: ReferenceConfig = REFERENCE_SMALL_V0,
) -> tuple[Any, tuple[Any, Any], dict[str, jax.Array]]:
    target = batch["target"]
    if target.dtype not in (jnp.int32, jnp.int64):
        raise ValueError("codec target bytes must be integer-valued")
    return codec_pretrain_step(params, moments, step, batch, config=config)


def full_byte_codec_validation(
    params: Any,
    batch: dict[str, jax.Array],
    *,
    config: ReferenceConfig = REFERENCE_SMALL_V0,
) -> dict[str, jax.Array]:
    outputs = apply_target_codec(params, batch["target"], batch["target_mask"], config=config)
    losses = codec_loss_components(outputs)
    teacher = outputs["target"]["teacher"]
    byte_accuracy, sequence_accuracy = emitted_metrics(
        outputs["codec_logits"], teacher.slot_bytes, teacher.slot_mask
    )
    observed = jnp.bincount(
        batch["target"].reshape(-1),
        weights=batch["target_mask"].reshape(-1),
        length=256,
    )
    predicted = jnp.argmax(outputs["codec_logits"], axis=-1)
    labels = teacher.slot_bytes
    mask = teacher.slot_mask
    per_byte_correct = jnp.bincount(
        labels.reshape(-1),
        weights=((predicted == labels) & mask).reshape(-1),
        length=256,
    )
    per_byte_accuracy = per_byte_correct / jnp.maximum(observed, 1)
    return {
        **losses,
        "byte_accuracy": byte_accuracy,
        "sequence_accuracy": sequence_accuracy,
        "covered_byte_values": jnp.sum(observed > 0),
        "full_byte_coverage": jnp.all(observed > 0),
        "per_byte_count": observed,
        "per_byte_correct": per_byte_correct,
        "per_byte_accuracy": per_byte_accuracy,
        "minimum_covered_byte_accuracy": jnp.min(jnp.where(observed > 0, per_byte_accuracy, 1.0)),
        "nonfinite": jnp.any(~jnp.isfinite(outputs["codec_logits"])),
    }


def validate_recipe(recipe: SmallScaleLMRecipe) -> None:
    if recipe.scale not in LANGUAGE_SCALES:
        raise ValueError(f"unknown language scale: {recipe.scale}")
    if recipe.codec_version != FULL_BYTE_CODEC_V0:
        raise ValueError("Phase-I recipes must name the separately versioned full-byte codec")
    if recipe.authoritative:
        raise ValueError("authoritative language training is gated on completed F/G/H integration")


def write_json(path: str | Path, payload: object) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
