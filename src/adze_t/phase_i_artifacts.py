"""Crash-safe, protocol-bound Phase-I candidate checkpoint transactions."""

from __future__ import annotations

import hashlib
import json
import os
import pickle
import re
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

from .phase_i import CODEC_CHECKPOINT_SCHEMA_V0, canonical_hash, validate_sha256


@dataclass(frozen=True)
class CheckpointRecord:
    schema: str
    step: int
    protocol_sha256: str
    tree_identity_sha256: str
    checkpoint_file: str
    checkpoint_sha256: str


@dataclass(frozen=True)
class EvidenceRecord:
    schema: str
    step: int
    protocol_sha256: str
    checkpoint_sha256: str
    metrics_sha256: str
    training_exposure_sha256: str
    evaluation_coverage_sha256: str
    reconstruction_sha256: str
    transaction_sha256: str
    metrics: Mapping[str, object]
    training_exposure: Mapping[str, object]
    evaluation_coverage: Mapping[str, object]
    reconstruction: Mapping[str, object]


@dataclass(frozen=True)
class Reconciliation:
    last_complete_step: int
    checkpoint_record: CheckpointRecord | None
    evidence_record: EvidenceRecord | None
    incomplete_steps: tuple[int, ...]


@dataclass(frozen=True)
class RecoveryResult:
    reconciliation: Reconciliation
    quarantine_directory: str | None
    moved_artifacts: tuple[str, ...]


_CHECKPOINT_PAYLOAD = re.compile(r"^step_(\d{8})\.pkl$")
_STEP_RECORD = re.compile(r"^step_(\d{8})\.json$")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def _atomic_json(path: Path, payload: object) -> None:
    _atomic_bytes(path, (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode())


def checkpoint_tree_identity(tree_description: object) -> str:
    """Hash a caller-supplied canonical tree of paths/shapes/dtypes/ownership."""
    return canonical_hash(tree_description)


def save_transaction(
    directory: str | Path,
    *,
    step: int,
    protocol_sha256: str,
    tree_identity_sha256: str,
    state: object,
    metrics: Mapping[str, object],
    training_exposure: Mapping[str, object],
    evaluation_coverage: Mapping[str, object],
    reconstruction: Mapping[str, object],
) -> tuple[CheckpointRecord, EvidenceRecord]:
    """Persist checkpoint, metadata, evidence, then latest pointer in that order."""
    if step < 0:
        raise ValueError("checkpoint step must be nonnegative")
    validate_sha256(protocol_sha256, "protocol_sha256")
    validate_sha256(tree_identity_sha256, "tree_identity_sha256")
    root = Path(directory)
    checkpoints = root / "checkpoints"
    evidence = root / "evidence"
    checkpoint_path = checkpoints / f"step_{step:08d}.pkl"
    checkpoint_meta_path = checkpoints / f"step_{step:08d}.json"
    evidence_path = evidence / f"step_{step:08d}.json"
    if checkpoint_path.exists() or checkpoint_meta_path.exists() or evidence_path.exists():
        raise RuntimeError(f"refusing to overwrite durable step {step}")
    if root.exists():
        prior = reconcile_transactions(root, protocol_sha256=protocol_sha256)
        if prior.incomplete_steps:
            raise RuntimeError("incomplete transactions require explicit recovery before save")
        if prior.checkpoint_record is not None and step <= prior.last_complete_step:
            raise RuntimeError("checkpoint steps must increase monotonically")
    _atomic_bytes(checkpoint_path, pickle.dumps(state, protocol=pickle.HIGHEST_PROTOCOL))
    checkpoint_record = CheckpointRecord(
        schema=CODEC_CHECKPOINT_SCHEMA_V0,
        step=step,
        protocol_sha256=protocol_sha256,
        tree_identity_sha256=tree_identity_sha256,
        checkpoint_file=checkpoint_path.relative_to(root).as_posix(),
        checkpoint_sha256=_file_sha256(checkpoint_path),
    )
    _atomic_json(checkpoint_meta_path, asdict(checkpoint_record))

    metrics_sha256 = canonical_hash(metrics)
    training_exposure_sha256 = canonical_hash(training_exposure)
    evaluation_coverage_sha256 = canonical_hash(evaluation_coverage)
    reconstruction_sha256 = canonical_hash(reconstruction)
    transaction_payload = {
        "schema": CODEC_CHECKPOINT_SCHEMA_V0,
        "step": step,
        "protocol_sha256": protocol_sha256,
        "checkpoint_sha256": checkpoint_record.checkpoint_sha256,
        "metrics_sha256": metrics_sha256,
        "training_exposure_sha256": training_exposure_sha256,
        "evaluation_coverage_sha256": evaluation_coverage_sha256,
        "reconstruction_sha256": reconstruction_sha256,
    }
    evidence_record = EvidenceRecord(
        **transaction_payload,
        transaction_sha256=canonical_hash(transaction_payload),
        metrics=metrics,
        training_exposure=training_exposure,
        evaluation_coverage=evaluation_coverage,
        reconstruction=reconstruction,
    )
    _atomic_json(evidence_path, asdict(evidence_record))
    _atomic_json(
        root / "latest.json",
        {
            "schema": CODEC_CHECKPOINT_SCHEMA_V0,
            "step": step,
            "protocol_sha256": protocol_sha256,
            "transaction_sha256": evidence_record.transaction_sha256,
        },
    )
    return checkpoint_record, evidence_record


def _load_checkpoint_record(path: Path) -> CheckpointRecord:
    record = CheckpointRecord(**json.loads(path.read_text(encoding="utf-8")))
    if record.schema != CODEC_CHECKPOINT_SCHEMA_V0 or record.step < 0:
        raise ValueError(f"invalid checkpoint record: {path}")
    validate_sha256(record.protocol_sha256, "protocol_sha256")
    validate_sha256(record.tree_identity_sha256, "tree_identity_sha256")
    validate_sha256(record.checkpoint_sha256, "checkpoint_sha256")
    expected_file = f"checkpoints/step_{record.step:08d}.pkl"
    if record.checkpoint_file != expected_file:
        raise ValueError(f"noncanonical checkpoint path: {record.checkpoint_file}")
    return record


def _load_evidence_record(path: Path) -> EvidenceRecord:
    record = EvidenceRecord(**json.loads(path.read_text(encoding="utf-8")))
    expected = canonical_hash(
        {
            "schema": record.schema,
            "step": record.step,
            "protocol_sha256": record.protocol_sha256,
            "checkpoint_sha256": record.checkpoint_sha256,
            "metrics_sha256": record.metrics_sha256,
            "training_exposure_sha256": record.training_exposure_sha256,
            "evaluation_coverage_sha256": record.evaluation_coverage_sha256,
            "reconstruction_sha256": record.reconstruction_sha256,
        }
    )
    if record.schema != CODEC_CHECKPOINT_SCHEMA_V0 or record.transaction_sha256 != expected:
        raise ValueError(f"invalid evidence transaction: {path}")
    if record.metrics_sha256 != canonical_hash(record.metrics):
        raise ValueError(f"evidence metrics hash mismatch: {path}")
    if record.training_exposure_sha256 != canonical_hash(record.training_exposure):
        raise ValueError(f"training-exposure evidence hash mismatch: {path}")
    if record.evaluation_coverage_sha256 != canonical_hash(record.evaluation_coverage):
        raise ValueError(f"evaluation-coverage evidence hash mismatch: {path}")
    if record.reconstruction_sha256 != canonical_hash(record.reconstruction):
        raise ValueError(f"reconstruction evidence hash mismatch: {path}")
    return record


def _reconcile_transactions(
    directory: str | Path, *, protocol_sha256: str, validate_latest: bool
) -> Reconciliation:
    validate_sha256(protocol_sha256, "protocol_sha256")
    root = Path(directory)
    temporary = [
        path
        for parent in (root / "checkpoints", root / "evidence", root)
        if parent.exists()
        for path in parent.glob("*.tmp")
    ]
    if temporary:
        raise ValueError("temporary transaction artifacts require explicit recovery")
    checkpoint_paths: dict[int, Path] = {}
    evidence_paths: dict[int, Path] = {}
    for path in (root / "checkpoints").glob("step_*.json"):
        match = _STEP_RECORD.fullmatch(path.name)
        if match is None:
            raise ValueError(f"malformed checkpoint record name: {path.name}")
        checkpoint_paths[int(match.group(1))] = path
    for path in (root / "evidence").glob("step_*.json"):
        match = _STEP_RECORD.fullmatch(path.name)
        if match is None:
            raise ValueError(f"malformed evidence record name: {path.name}")
        evidence_paths[int(match.group(1))] = path
    payload_steps: set[int] = set()
    for path in (root / "checkpoints").glob("step_*.pkl"):
        match = _CHECKPOINT_PAYLOAD.fullmatch(path.name)
        if match is None:
            raise ValueError(f"malformed checkpoint payload name: {path.name}")
        payload_steps.add(int(match.group(1)))
    all_steps = set(checkpoint_paths) | set(evidence_paths) | payload_steps
    incomplete: list[int] = []
    complete: list[tuple[int, CheckpointRecord, EvidenceRecord]] = []
    for step in sorted(all_steps):
        if step not in checkpoint_paths or step not in evidence_paths:
            incomplete.append(step)
            continue
        checkpoint = _load_checkpoint_record(checkpoint_paths[step])
        evidence = _load_evidence_record(evidence_paths[step])
        checkpoint_file = root / checkpoint.checkpoint_file
        if checkpoint.step != evidence.step or checkpoint.step != step:
            raise ValueError(f"step identity mismatch at transaction {step}")
        if checkpoint.protocol_sha256 != evidence.protocol_sha256:
            raise ValueError(f"mixed protocol identities at transaction {step}")
        if checkpoint.protocol_sha256 != protocol_sha256:
            raise ValueError(f"stale run protocol at transaction {step}")
        if checkpoint.checkpoint_sha256 != evidence.checkpoint_sha256:
            raise ValueError(f"mixed checkpoint identities at transaction {step}")
        if (
            not checkpoint_file.exists()
            or checkpoint_file.is_symlink()
            or _file_sha256(checkpoint_file) != checkpoint.checkpoint_sha256
        ):
            raise ValueError(f"checkpoint payload mismatch at transaction {step}")
        complete.append((step, checkpoint, evidence))
    if not complete:
        result = Reconciliation(0, None, None, tuple(incomplete))
    else:
        step, checkpoint, evidence = complete[-1]
        result = Reconciliation(step, checkpoint, evidence, tuple(incomplete))
    if validate_latest:
        latest_path = root / "latest.json"
        if result.evidence_record is None:
            if latest_path.exists():
                raise ValueError("latest pointer exists without a complete transaction")
        else:
            expected_latest = {
                "schema": CODEC_CHECKPOINT_SCHEMA_V0,
                "step": result.last_complete_step,
                "protocol_sha256": protocol_sha256,
                "transaction_sha256": result.evidence_record.transaction_sha256,
            }
            if not latest_path.exists():
                raise ValueError("latest pointer is missing")
            try:
                observed_latest = json.loads(latest_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as error:
                raise ValueError("latest pointer is invalid JSON") from error
            if observed_latest != expected_latest:
                raise ValueError("latest pointer does not match the last complete transaction")
    return result


def reconcile_transactions(directory: str | Path, *, protocol_sha256: str) -> Reconciliation:
    """Return the last mutually durable pair and validate its latest pointer."""
    return _reconcile_transactions(directory, protocol_sha256=protocol_sha256, validate_latest=True)


def _next_quarantine(root: Path) -> Path:
    quarantine = root / "quarantine"
    indices = []
    for path in quarantine.glob("recovery_*"):
        try:
            indices.append(int(path.name.removeprefix("recovery_")))
        except ValueError:
            continue
    return quarantine / f"recovery_{max(indices, default=0) + 1:06d}"


def _move_to_quarantine(root: Path, destination: Path, path: Path) -> str:
    relative = path.relative_to(root)
    target = destination / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise RuntimeError(f"quarantine target already exists: {target}")
    shutil.move(path.as_posix(), target.as_posix())
    return relative.as_posix()


def recover_transactions(directory: str | Path, *, protocol_sha256: str) -> RecoveryResult:
    """Explicitly quarantine incomplete/stale artifacts, then reconcile.

    This operation never deletes or overwrites candidate evidence. Normal resume
    remains fail-closed; callers must deliberately invoke recovery.
    """
    validate_sha256(protocol_sha256, "protocol_sha256")
    root = Path(directory)
    expected: dict[int, dict[str, Path]] = {}
    malformed: list[Path] = []
    for path in (root / "checkpoints").glob("step_*"):
        payload_match = _CHECKPOINT_PAYLOAD.fullmatch(path.name)
        record_match = _STEP_RECORD.fullmatch(path.name)
        if payload_match:
            expected.setdefault(int(payload_match.group(1)), {})["payload"] = path
        elif record_match:
            expected.setdefault(int(record_match.group(1)), {})["checkpoint"] = path
        else:
            malformed.append(path)
    for path in (root / "evidence").glob("step_*"):
        record_match = _STEP_RECORD.fullmatch(path.name)
        if record_match:
            expected.setdefault(int(record_match.group(1)), {})["evidence"] = path
        else:
            malformed.append(path)
    temporary = [
        path
        for parent in (root / "checkpoints", root / "evidence", root)
        if parent.exists()
        for path in parent.glob("*.tmp")
    ]

    invalid: list[Path] = malformed + temporary
    for step, paths in sorted(expected.items()):
        if set(paths) != {"payload", "checkpoint", "evidence"}:
            invalid.extend(paths.values())
            continue
        try:
            checkpoint = _load_checkpoint_record(paths["checkpoint"])
            evidence = _load_evidence_record(paths["evidence"])
            valid = (
                checkpoint.step == evidence.step == step
                and checkpoint.protocol_sha256 == evidence.protocol_sha256 == protocol_sha256
                and checkpoint.checkpoint_sha256 == evidence.checkpoint_sha256
                and checkpoint.checkpoint_file == paths["payload"].relative_to(root).as_posix()
                and not paths["payload"].is_symlink()
                and _file_sha256(paths["payload"]) == checkpoint.checkpoint_sha256
            )
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            valid = False
        if not valid:
            invalid.extend(paths.values())

    invalid = sorted(set(invalid), key=lambda path: path.as_posix())
    quarantine: Path | None = None
    moved: list[str] = []
    if invalid:
        quarantine = _next_quarantine(root)
        quarantine.mkdir(parents=True, exist_ok=False)
        for path in invalid:
            if path.exists():
                moved.append(_move_to_quarantine(root, quarantine, path))

    latest = root / "latest.json"
    reconciliation = _reconcile_transactions(
        root, protocol_sha256=protocol_sha256, validate_latest=False
    )
    expected_latest = (
        None
        if reconciliation.evidence_record is None
        else {
            "schema": CODEC_CHECKPOINT_SCHEMA_V0,
            "step": reconciliation.last_complete_step,
            "protocol_sha256": protocol_sha256,
            "transaction_sha256": reconciliation.evidence_record.transaction_sha256,
        }
    )
    existing_latest: object | None = None
    if latest.exists():
        try:
            existing_latest = json.loads(latest.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            existing_latest = None
    if existing_latest != expected_latest:
        if latest.exists():
            if quarantine is None:
                quarantine = _next_quarantine(root)
                quarantine.mkdir(parents=True, exist_ok=False)
            moved.append(_move_to_quarantine(root, quarantine, latest))
        if expected_latest is not None:
            _atomic_json(latest, expected_latest)
    if quarantine is not None:
        manifest = {
            "schema": "PHASE_I_RECOVERY_QUARANTINE_V0",
            "protocol_sha256": protocol_sha256,
            "moved_artifacts": moved,
        }
        manifest["manifest_sha256"] = canonical_hash(manifest)
        _atomic_json(quarantine / "manifest.json", manifest)
    return RecoveryResult(
        reconciliation=reconciliation,
        quarantine_directory=(
            None if quarantine is None else quarantine.relative_to(root).as_posix()
        ),
        moved_artifacts=tuple(moved),
    )


def load_checkpoint_state(directory: str | Path, record: CheckpointRecord) -> Any:
    if record.checkpoint_file != f"checkpoints/step_{record.step:08d}.pkl":
        raise ValueError("checkpoint record has a noncanonical payload path")
    path = Path(directory) / record.checkpoint_file
    if path.is_symlink() or _file_sha256(path) != record.checkpoint_sha256:
        raise ValueError("checkpoint payload hash mismatch")
    with path.open("rb") as stream:
        return pickle.load(stream)  # noqa: S301 - trusted, hash-bound candidate artifact
