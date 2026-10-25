"""Phase-I full-byte codec, scale, evidence, and provenance contracts."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

import jax
import jax.numpy as jnp
import numpy as np
import yaml

from .config import CarrierConfig, ModelConfig, PackingConfig, ReferenceConfig, TrainingConfig
from .language_data import ByteSegment, make_generated_segment, pad_segments
from .model import apply_target_codec, init_model_params
from .objectives import codec_loss_components, emitted_metrics
from .training import codec_pretrain_step, codec_update_mask


FULL_BYTE_CODEC_V0 = "PHASE_I_FULL_BYTE_CODEC_V0"
NATURAL_TEXT_PIPELINE_V0 = "PHASE_I_NATURAL_TEXT_BYTES_V0"
RAW_BINARY_PIPELINE_V0 = "PHASE_I_RAW_BINARY_V0"
CODEC_RUN_PROTOCOL_V0 = "PHASE_I_CODEC_RUN_PROTOCOL_V0"
CODEC_CHECKPOINT_SCHEMA_V0 = "PHASE_I_CODEC_CHECKPOINT_V0"
EVIDENCE_SCHEMA_V0 = "PHASE_I_BYTE_EVIDENCE_V0"
PINNED_TORX_REVISION = "f1fc858ed950ecd41935d15c06d0ec7c5e0674ae"
ACCEPTED_PHASE_B_CODEC_PATH = "results/phase_b/checkpoints/target_codec_b1.pkl"
FULL_BYTE_CODEC_CHECKPOINT_DIR = "results/phase_i/candidates/full_byte_codec_v0"
DEFAULT_SCALE_CONFIG_PATH = Path("configs/phase_i_language_scales_v0.yaml")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_GIT_COMMIT = re.compile(r"^[0-9a-f]{40,64}$")


def canonical_json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def canonical_hash(value: object) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


def validate_sha256(value: str, name: str) -> None:
    if not _SHA256.fullmatch(value):
        raise ValueError(f"{name} must be a lowercase SHA-256")


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
    status: str
    batch_size: int
    gradient_accumulation: int
    reference: ReferenceConfig
    resolved_config_sha256: str


@dataclass(frozen=True)
class RuntimeIdentity:
    python: str
    platform: str
    jax: str
    torx: str
    torx_revision: str
    backend: str
    x64_enabled: bool


@dataclass(frozen=True)
class CodecRunProtocol:
    version: str
    source_commit: str
    implementation_sha256: str
    dataset_manifest_sha256: str
    dataset_protocol: str
    scale_name: str
    resolved_config_sha256: str
    candidate_version: str
    initialization_seed: int
    training_seed: int
    sequence_bytes: int
    batch_size: int
    gradient_accumulation: int
    max_updates: int
    checkpoint_every_updates: int
    validation_every_updates: int
    optimizer: str
    learning_rate: float
    weight_decay: float
    grad_clip_norm: float
    schedule: str
    calibration_policy: str
    evaluation_protocol: str
    structure_protocol: str
    refinement_protocol: str
    q: int
    s: int
    r: int
    runtime: RuntimeIdentity
    parent_checkpoint_sha256: str | None = None


@dataclass(frozen=True)
class SmallScaleLMRecipe:
    name: str
    run_protocol_sha256: str
    codec_checkpoint_sha256: str
    codec_checkpoint_step: int
    evaluation_protocol: str
    authoritative: bool = False


@dataclass(frozen=True)
class ByteEvidence:
    schema: str
    kind: str
    counts: tuple[int, ...]
    correct: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        if self.schema != EVIDENCE_SCHEMA_V0:
            raise ValueError("unsupported byte-evidence schema")
        if self.kind not in ("training_exposure", "evaluation_coverage", "reconstruction"):
            raise ValueError("unsupported byte-evidence kind")
        if len(self.counts) != 256 or any(
            not isinstance(value, int) or isinstance(value, bool) or value < 0
            for value in self.counts
        ):
            raise ValueError("byte evidence requires 256 nonnegative counts")
        if self.kind == "reconstruction":
            if self.correct is None or len(self.correct) != 256:
                raise ValueError("reconstruction evidence requires 256 correct counts")
            if any(
                not isinstance(value, int) or isinstance(value, bool) or value < 0
                for value in self.correct
            ) or any(right > total for right, total in zip(self.correct, self.counts)):
                raise ValueError("invalid reconstruction counts")
        elif self.correct is not None:
            raise ValueError("only reconstruction evidence may contain correct counts")


def _reference_payload(config: ReferenceConfig) -> dict[str, object]:
    return asdict(config)


def _validate_reference_config(config: ReferenceConfig) -> None:
    if config.model.byte_vocab != 256:
        raise ValueError("language scales require the full 256-byte vocabulary")
    if config.model.target_max_bytes > config.carrier.C * config.carrier.L_max:
        raise ValueError("target width exceeds carrier emission capacity")
    if config.model.d_model != config.model.heads * config.model.head_dim:
        raise ValueError("d_model must equal heads * head_dim")
    if config.packing.M_max < math.ceil(config.carrier.C / config.packing.K):
        raise ValueError("packing capacity cannot represent the carrier")


def load_language_scales(
    path: str | Path = DEFAULT_SCALE_CONFIG_PATH,
) -> dict[str, LanguageScaleConfig]:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if raw.get("version") != "PHASE_I_LANGUAGE_SCALES_V0":
        raise ValueError("unsupported language-scale registry")
    if raw.get("byte_vocab") != 256 or raw.get("tokeniser") != "none":
        raise ValueError("language-scale registry violates the byte interface")
    scales: dict[str, LanguageScaleConfig] = {}
    for name, record in raw["scales"].items():
        resolved = record["reference_config"]
        config = ReferenceConfig(
            carrier=CarrierConfig(**resolved["carrier"]),
            packing=PackingConfig(**resolved["packing"]),
            model=ModelConfig(**resolved["model"]),
            training=TrainingConfig(**resolved["training"]),
        )
        _validate_reference_config(config)
        batch_size = int(record["batch_size"])
        accumulation = int(record["gradient_accumulation"])
        if batch_size <= 0 or accumulation <= 0 or config.training.batch_size != batch_size:
            raise ValueError(f"invalid batch/accumulation for scale {name}")
        scales[name] = LanguageScaleConfig(
            name=name,
            status=str(record["status"]),
            batch_size=batch_size,
            gradient_accumulation=accumulation,
            reference=config,
            resolved_config_sha256=canonical_hash(_reference_payload(config)),
        )
    return scales


def candidate_manifest(
    dataset_manifest: Mapping[str, object],
    scale: LanguageScaleConfig,
    run_protocol_sha256: str,
    config: CodecCandidateConfig = CodecCandidateConfig(),
) -> dict[str, Any]:
    dataset_hash = str(dataset_manifest.get("manifest_sha256", ""))
    validate_sha256(dataset_hash, "dataset_manifest_sha256")
    validate_sha256(run_protocol_sha256, "run_protocol_sha256")
    payload: dict[str, Any] = {
        "candidate": asdict(config),
        "dataset_manifest_sha256": dataset_hash,
        "dataset_protocol": dataset_manifest.get("protocol"),
        "scale_name": scale.name,
        "resolved_config_sha256": scale.resolved_config_sha256,
        "run_protocol_sha256": run_protocol_sha256,
        "byte_vocab": 256,
        "target_codec_topology": "shared_frontend_target_analysis_mamba_decoder",
        "initialization": "fresh_seeded_full_candidate_codec_leaves_only_trainable",
        "accepted_phase_b_codec": {
            "path": ACCEPTED_PHASE_B_CODEC_PATH,
            "policy": "immutable_not_a_parent",
        },
        "authoritative_training_allowed": False,
    }
    payload["manifest_sha256"] = canonical_hash(payload)
    return payload


def full_byte_calibration_segments(sequence_bytes: int = 128) -> tuple[ByteSegment, ...]:
    if sequence_bytes <= 0:
        raise ValueError("sequence_bytes must be positive")
    values = bytes(range(256))
    return tuple(
        make_generated_segment(
            values[start : start + sequence_bytes],
            document_id="phase-i-full-byte-calibration-v0",
            source="generated://phase-i/full-byte-calibration-v0",
            start_byte=start,
        )
        for start in range(0, 256, sequence_bytes)
    )


def deterministic_epoch_order(
    segment_ids: Iterable[str], *, seed: int, epoch: int
) -> tuple[int, ...]:
    keyed = [
        (hashlib.sha256(f"{seed}:{epoch}:{identifier}".encode()).digest(), index)
        for index, identifier in enumerate(segment_ids)
    ]
    return tuple(index for _, index in sorted(keyed))


def codec_batch_count(segment_count: int, batch_size: int) -> int:
    if segment_count <= 0 or batch_size <= 0:
        raise ValueError("segment_count and batch_size must be positive")
    return math.ceil(segment_count / batch_size)


def codec_batch_at(
    segments: Sequence[ByteSegment],
    *,
    width: int,
    batch_size: int,
    order: Sequence[int],
    batch_index: int,
) -> dict[str, Any]:
    count = codec_batch_count(len(segments), batch_size)
    if len(order) != len(segments) or sorted(order) != list(range(len(segments))):
        raise ValueError("order must be a permutation of segment indices")
    if not 0 <= batch_index < count:
        raise IndexError("batch_index outside epoch")
    start = batch_index * batch_size
    selected = [segments[index] for index in order[start : start + batch_size]]
    values, mask, ids = pad_segments(selected, width)
    return {
        "target": jnp.asarray(values),
        "target_mask": jnp.asarray(mask),
        "segment_ids": ids,
    }


def iter_codec_batches(
    segments: Sequence[ByteSegment],
    *,
    width: int,
    batch_size: int,
    seed: int,
    epoch: int,
) -> Iterator[dict[str, Any]]:
    order = deterministic_epoch_order(
        (segment.segment_id for segment in segments), seed=seed, epoch=epoch
    )
    for batch_index in range(codec_batch_count(len(segments), batch_size)):
        yield codec_batch_at(
            segments,
            width=width,
            batch_size=batch_size,
            order=order,
            batch_index=batch_index,
        )


def calibration_prefixed_batch(
    natural: Sequence[ByteSegment],
    calibration: Sequence[ByteSegment],
    *,
    width: int,
    batch_size: int,
    seed: int,
    update: int,
) -> tuple[dict[str, Any], int]:
    """Select one batch, exposing every calibration byte before natural data."""
    if update <= 0:
        raise ValueError("update must be positive")
    calibration_batches = codec_batch_count(len(calibration), batch_size)
    natural_batches = codec_batch_count(len(natural), batch_size)
    updates_per_epoch = calibration_batches + natural_batches
    epoch, index = divmod(update - 1, updates_per_epoch)
    if index < calibration_batches:
        return (
            codec_batch_at(
                calibration,
                width=width,
                batch_size=batch_size,
                order=tuple(range(len(calibration))),
                batch_index=index,
            ),
            epoch,
        )
    order = deterministic_epoch_order(
        (segment.segment_id for segment in natural), seed=seed, epoch=epoch
    )
    return (
        codec_batch_at(
            natural,
            width=width,
            batch_size=batch_size,
            order=order,
            batch_index=index - calibration_batches,
        ),
        epoch,
    )


def byte_histogram(values: np.ndarray, mask: np.ndarray) -> tuple[int, ...]:
    selected = np.asarray(values)[np.asarray(mask, dtype=bool)]
    if selected.size and (selected.min() < 0 or selected.max() > 255):
        raise ValueError("byte evidence contains a value outside 0..255")
    return tuple(np.bincount(selected.astype(np.int64), minlength=256).tolist())


def training_exposure_evidence(counts: Sequence[int]) -> ByteEvidence:
    return ByteEvidence(EVIDENCE_SCHEMA_V0, "training_exposure", tuple(counts))


def evaluation_coverage_evidence(counts: Sequence[int]) -> ByteEvidence:
    return ByteEvidence(EVIDENCE_SCHEMA_V0, "evaluation_coverage", tuple(counts))


def reconstruction_evidence(counts: Sequence[int], correct: Sequence[int]) -> ByteEvidence:
    return ByteEvidence(EVIDENCE_SCHEMA_V0, "reconstruction", tuple(counts), tuple(correct))


def validate_run_protocol(protocol: CodecRunProtocol) -> str:
    if protocol.version != CODEC_RUN_PROTOCOL_V0:
        raise ValueError("unsupported codec run protocol")
    if not _GIT_COMMIT.fullmatch(protocol.source_commit):
        raise ValueError("source_commit must be a full Git object ID")
    for name in ("implementation_sha256", "dataset_manifest_sha256", "resolved_config_sha256"):
        validate_sha256(str(getattr(protocol, name)), name)
    if protocol.parent_checkpoint_sha256 is not None:
        raise ValueError("the full-byte codec candidate cannot inherit a parent checkpoint")
    if protocol.candidate_version != FULL_BYTE_CODEC_V0:
        raise ValueError("run does not use the separately versioned full-byte codec")
    if protocol.dataset_protocol not in (NATURAL_TEXT_PIPELINE_V0, RAW_BINARY_PIPELINE_V0):
        raise ValueError("unsupported dataset protocol")
    positive_integers = (
        protocol.sequence_bytes,
        protocol.batch_size,
        protocol.gradient_accumulation,
        protocol.max_updates,
        protocol.checkpoint_every_updates,
        protocol.validation_every_updates,
    )
    if any(
        not isinstance(value, int) or isinstance(value, bool) or value <= 0
        for value in positive_integers
    ):
        raise ValueError("run protocol contains invalid cadence values")
    if any(
        not isinstance(value, int) or isinstance(value, bool) or value < 0
        for value in (protocol.initialization_seed, protocol.training_seed)
    ):
        raise ValueError("run protocol contains invalid seeds")
    optimizer_scalars = (
        protocol.learning_rate,
        protocol.weight_decay,
        protocol.grad_clip_norm,
    )
    if not all(math.isfinite(value) for value in optimizer_scalars):
        raise ValueError("run protocol contains nonfinite optimizer values")
    if protocol.learning_rate <= 0 or protocol.grad_clip_norm <= 0 or protocol.weight_decay < 0:
        raise ValueError("run protocol contains invalid optimizer/cadence values")
    if protocol.optimizer != "adamw" or protocol.schedule != "constant":
        raise ValueError("unsupported optimizer or schedule")
    if protocol.validation_every_updates != protocol.checkpoint_every_updates:
        raise ValueError("validation and checkpoint cadence must match for atomic evidence")
    if any(
        not isinstance(value, int) or isinstance(value, bool) or value < 0
        for value in (protocol.q, protocol.s, protocol.r)
    ):
        raise ValueError("Q/S/R must be nonnegative")
    if protocol.runtime.torx_revision != PINNED_TORX_REVISION:
        raise ValueError("run protocol does not use the pinned Torx revision")
    textual_fields = (
        protocol.scale_name,
        protocol.calibration_policy,
        protocol.evaluation_protocol,
        protocol.structure_protocol,
        protocol.refinement_protocol,
        protocol.runtime.python,
        protocol.runtime.platform,
        protocol.runtime.jax,
        protocol.runtime.torx,
        protocol.runtime.backend,
    )
    if any(not value for value in textual_fields):
        raise ValueError("run protocol contains an empty identity field")
    return canonical_hash(asdict(protocol))


def validate_recipe(recipe: SmallScaleLMRecipe) -> None:
    if not recipe.name or not recipe.evaluation_protocol:
        raise ValueError("recipe name and evaluation protocol must be non-empty")
    validate_sha256(recipe.run_protocol_sha256, "run_protocol_sha256")
    validate_sha256(recipe.codec_checkpoint_sha256, "codec_checkpoint_sha256")
    if recipe.codec_checkpoint_step < 0:
        raise ValueError("codec checkpoint step must be nonnegative")
    if recipe.authoritative:
        raise ValueError("authoritative language training is gated on completed G/H integration")


def codec_state_tree_description(params: Any, moments: tuple[Any, Any]) -> list[dict[str, object]]:
    """Describe parameter ownership and both Adam trees for resume identity."""
    mask = codec_update_mask(params)
    parameter_leaves, _ = jax.tree_util.tree_flatten_with_path(params)
    mask_leaves, _ = jax.tree_util.tree_flatten_with_path(mask)
    if [path for path, _ in parameter_leaves] != [path for path, _ in mask_leaves]:
        raise RuntimeError("codec ownership mask does not match the parameter tree")
    description = [
        {
            "tree": "params",
            "path": jax.tree_util.keystr(path),
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "ownership": "codec_trainable" if bool(enabled) else "fresh_frozen_not_trained",
        }
        for (path, value), (_, enabled) in zip(parameter_leaves, mask_leaves, strict=True)
    ]
    parameter_paths = [path for path, _ in parameter_leaves]
    for tree_name, moment_tree in zip(("adam_first", "adam_second"), moments, strict=True):
        moment_leaves, _ = jax.tree_util.tree_flatten_with_path(moment_tree)
        if [path for path, _ in moment_leaves] != parameter_paths:
            raise RuntimeError(f"{tree_name} does not match the parameter tree")
        for (parameter_path, parameter), (moment_path, moment) in zip(
            parameter_leaves, moment_leaves, strict=True
        ):
            if moment.shape != parameter.shape or moment.dtype != parameter.dtype:
                raise RuntimeError(f"{tree_name} leaf does not match {parameter_path}")
            description.append(
                {
                    "tree": tree_name,
                    "path": jax.tree_util.keystr(moment_path),
                    "shape": list(moment.shape),
                    "dtype": str(moment.dtype),
                    "ownership": "optimizer_state",
                }
            )
    return description


def validate_codec_resume_state(
    state: Any,
    *,
    protocol_sha256: str,
    expected_step: int,
    expected_tree_sha256: str,
) -> tuple[Any, tuple[Any, Any], tuple[int, ...], list[dict[str, object]]]:
    """Fail closed unless a loaded state exactly matches its durable record."""
    if not isinstance(state, dict) or state.get("schema") != "PHASE_I_CODEC_STATE_V0":
        raise RuntimeError("checkpoint state has an unsupported schema")
    if state.get("protocol_sha256") != protocol_sha256 or state.get("step") != expected_step:
        raise RuntimeError("checkpoint state identity does not match its transaction")
    params = state.get("params")
    moments = state.get("moments")
    if not isinstance(moments, tuple) or len(moments) != 2:
        raise RuntimeError("checkpoint Adam state is malformed")
    exposure_raw = state.get("training_exposure_counts")
    if not isinstance(exposure_raw, (tuple, list)) or len(exposure_raw) != 256:
        raise RuntimeError("checkpoint training exposure is malformed")
    exposure = tuple(exposure_raw)
    if any(
        not isinstance(value, int) or isinstance(value, bool) or value < 0 for value in exposure
    ):
        raise RuntimeError("checkpoint training exposure contains invalid counts")
    computed_description = codec_state_tree_description(params, moments)
    if state.get("tree_description") != computed_description:
        raise RuntimeError("checkpoint tree description does not match the saved trees")
    if canonical_hash(computed_description) != expected_tree_sha256:
        raise RuntimeError("checkpoint tree identity does not match its transaction")
    return params, moments, exposure, computed_description


def initialise_full_byte_codec_candidate(
    config: ReferenceConfig,
    candidate: CodecCandidateConfig = CodecCandidateConfig(),
) -> tuple[Any, tuple[Any, Any]]:
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
    config: ReferenceConfig,
) -> tuple[Any, tuple[Any, Any], dict[str, jax.Array]]:
    target = batch["target"]
    if target.dtype not in (jnp.int32, jnp.int64):
        raise ValueError("codec target bytes must be integer-valued")
    return codec_pretrain_step(params, moments, step, batch, config=config)


def full_byte_codec_validation(
    params: Any,
    batch: dict[str, jax.Array],
    *,
    config: ReferenceConfig,
) -> dict[str, jax.Array]:
    outputs = apply_target_codec(params, batch["target"], batch["target_mask"], config=config)
    losses = codec_loss_components(outputs)
    teacher = outputs["target"]["teacher"]
    byte_accuracy, sequence_accuracy = emitted_metrics(
        outputs["codec_logits"], teacher.slot_bytes, teacher.slot_mask
    )
    byte_support = jnp.sum(teacher.slot_mask)
    boundary_support = teacher.boundaries[:, :-1].size
    extent_support = teacher.length.size
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
    return {
        **losses,
        "codec_byte_loss_sum": losses["codec_byte"] * byte_support,
        "codec_byte_support": byte_support,
        "codec_b_loss_sum": losses["codec_b"] * boundary_support,
        "codec_b_support": jnp.asarray(boundary_support),
        "codec_l_loss_sum": losses["codec_l"] * extent_support,
        "codec_l_support": jnp.asarray(extent_support),
        "byte_accuracy": byte_accuracy,
        "sequence_accuracy": sequence_accuracy,
        "sequence_support": jnp.asarray(batch["target"].shape[0]),
        "per_byte_count": observed,
        "per_byte_correct": per_byte_correct,
        "nonfinite": jnp.any(~jnp.isfinite(outputs["codec_logits"])),
    }


def write_json(path: str | Path, payload: object) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(output)
