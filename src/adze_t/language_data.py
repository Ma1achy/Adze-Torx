"""Canonical, tokeniser-free Phase-I corpus preparation.

Text and arbitrary binary inputs are separate protocols. Text is strict UTF-8
with no implicit newline or Unicode normalization. Exact duplicate documents
form one provenance group, that group is assigned to one split, and only then
is its canonical representative segmented.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal, Sequence, TypeAlias

import numpy as np


TEXT_PROTOCOL_V0 = "PHASE_I_NATURAL_TEXT_BYTES_V0"
BINARY_PROTOCOL_V0 = "PHASE_I_RAW_BINARY_V0"
SplitName = Literal["train", "validation", "test"]


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical_json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _canonical_hash(value: object) -> str:
    return _sha256(_canonical_json(value))


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _manifest_hash(payload: dict[str, object]) -> str:
    unsigned = {key: value for key, value in payload.items() if key != "manifest_sha256"}
    return _canonical_hash(unsigned)


@dataclass(frozen=True)
class TextDocument:
    document_id: str
    source: str
    text: str
    source_sha256: str = ""
    group_id: str = ""
    protocol: str = TEXT_PROTOCOL_V0

    def as_bytes(self) -> bytes:
        return self.text.encode("utf-8")


@dataclass(frozen=True)
class BinaryDocument:
    document_id: str
    source: str
    values: tuple[int, ...]
    source_sha256: str
    group_id: str
    protocol: str = BINARY_PROTOCOL_V0

    def as_bytes(self) -> bytes:
        return bytes(self.values)


CorpusDocument: TypeAlias = TextDocument | BinaryDocument


@dataclass(frozen=True)
class ByteSegment:
    segment_id: str
    document_id: str
    group_id: str
    source: str
    source_sha256: str
    protocol: str
    split: str
    start_byte: int
    end_byte: int
    content_sha256: str
    values: tuple[int, ...]
    record_sha256: str

    @property
    def length(self) -> int:
        return len(self.values)

    def as_bytes(self) -> bytes:
        return bytes(self.values)


@dataclass(frozen=True)
class SegmentationConfig:
    max_bytes: int
    stride_bytes: int
    min_bytes: int = 1

    def __post_init__(self) -> None:
        if self.max_bytes <= 0:
            raise ValueError("max_bytes must be positive")
        if not 1 <= self.stride_bytes <= self.max_bytes:
            raise ValueError("stride_bytes must be in [1, max_bytes]")
        if not 1 <= self.min_bytes <= self.max_bytes:
            raise ValueError("min_bytes must be in [1, max_bytes]")


@dataclass(frozen=True)
class SplitConfig:
    seed: int
    train_fraction: float = 0.98
    validation_fraction: float = 0.01
    test_fraction: float = 0.01

    def __post_init__(self) -> None:
        fractions = (self.train_fraction, self.validation_fraction, self.test_fraction)
        if any(value < 0.0 for value in fractions):
            raise ValueError("split fractions must be non-negative")
        if abs(sum(fractions) - 1.0) > 1.0e-12:
            raise ValueError("split fractions must sum to one")

    def fraction(self, split: SplitName) -> float:
        return {
            "train": self.train_fraction,
            "validation": self.validation_fraction,
            "test": self.test_fraction,
        }[split]


@dataclass(frozen=True)
class CorpusSplits:
    train: tuple[ByteSegment, ...]
    validation: tuple[ByteSegment, ...]
    test: tuple[ByteSegment, ...]

    def named(self, split: SplitName) -> tuple[ByteSegment, ...]:
        return getattr(self, split)


@dataclass(frozen=True)
class PreparedCorpus:
    documents: tuple[CorpusDocument, ...]
    canonical_documents: tuple[CorpusDocument, ...]
    split_groups: dict[str, tuple[str, ...]]
    splits: CorpusSplits
    segmentation: SegmentationConfig
    split_config: SplitConfig
    protocol: str


def encode_utf8(text: str) -> tuple[int, ...]:
    """Encode Python text to raw UTF-8 bytes without normalization."""
    return tuple(text.encode("utf-8", errors="strict"))


def _document_identity(protocol: str, source: str, source_hash: str, content: bytes) -> str:
    return _canonical_hash(
        {
            "protocol": protocol,
            "source": source,
            "source_sha256": source_hash,
            "content_sha256": _sha256(content),
        }
    )


def _text_document(
    source: str, source_bytes: bytes, text: str, *, ordinal: int = 0
) -> TextDocument:
    content = text.encode("utf-8", errors="strict")
    source_hash = _sha256(source_bytes)
    source_identity = source if ordinal == 0 else f"{source}#record={ordinal}"
    return TextDocument(
        document_id=_document_identity(TEXT_PROTOCOL_V0, source_identity, source_hash, content),
        source=source_identity,
        text=text,
        source_sha256=source_hash,
        group_id=_sha256(content),
    )


def read_text_documents(
    paths: Sequence[str | Path],
    *,
    format: Literal["text", "jsonl"] = "text",
    text_field: str = "text",
) -> tuple[TextDocument, ...]:
    """Read strict raw UTF-8 without newline or Unicode normalization."""
    documents: list[TextDocument] = []
    for raw_path in sorted((Path(path) for path in paths), key=lambda item: item.as_posix()):
        source = raw_path.as_posix()
        source_bytes = raw_path.read_bytes()
        decoded = source_bytes.decode("utf-8", errors="strict")
        if format == "text":
            documents.append(_text_document(source, source_bytes, decoded))
            continue
        if format != "jsonl":
            raise ValueError(f"unsupported corpus format: {format}")
        for line_number, line in enumerate(decoded.splitlines(), 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict) or not isinstance(row.get(text_field), str):
                raise ValueError(f"{source}:{line_number}: missing string field {text_field!r}")
            documents.append(
                _text_document(source, source_bytes, row[text_field], ordinal=line_number)
            )
    return tuple(documents)


def read_binary_documents(paths: Sequence[str | Path]) -> tuple[BinaryDocument, ...]:
    """Read arbitrary 0..255 data under a protocol distinct from UTF-8 text."""
    documents = []
    for raw_path in sorted((Path(path) for path in paths), key=lambda item: item.as_posix()):
        source = raw_path.as_posix()
        content = raw_path.read_bytes()
        source_hash = _sha256(content)
        documents.append(
            BinaryDocument(
                document_id=_document_identity(BINARY_PROTOCOL_V0, source, source_hash, content),
                source=source,
                values=tuple(content),
                source_sha256=source_hash,
                group_id=_sha256(content),
            )
        )
    return tuple(documents)


def _utf8_offsets(text: str) -> tuple[int, ...]:
    offsets = [0]
    total = 0
    for scalar in text:
        total += len(scalar.encode("utf-8"))
        offsets.append(total)
    return tuple(offsets)


def _segment_identity(
    *,
    protocol: str,
    split: str,
    document_id: str,
    group_id: str,
    source: str,
    source_sha256: str,
    start: int,
    end: int,
    content_sha256: str,
) -> dict[str, object]:
    return {
        "protocol": protocol,
        "split": split,
        "document_id": document_id,
        "group_id": group_id,
        "source": source,
        "source_sha256": source_sha256,
        "start_byte": start,
        "end_byte": end,
        "content_sha256": content_sha256,
    }


def _make_segment(
    document: CorpusDocument, split: str, start: int, end: int, content: bytes
) -> ByteSegment:
    content_hash = _sha256(content)
    identity = _segment_identity(
        protocol=document.protocol,
        split=split,
        document_id=document.document_id,
        group_id=document.group_id,
        source=document.source,
        source_sha256=document.source_sha256,
        start=start,
        end=end,
        content_sha256=content_hash,
    )
    segment_id = _canonical_hash(identity)
    record = {**identity, "segment_id": segment_id, "values": list(content)}
    return ByteSegment(
        segment_id=segment_id,
        document_id=document.document_id,
        group_id=document.group_id,
        source=document.source,
        source_sha256=document.source_sha256,
        protocol=document.protocol,
        split=split,
        start_byte=start,
        end_byte=end,
        content_sha256=content_hash,
        values=tuple(content),
        record_sha256=_canonical_hash(record),
    )


def make_generated_segment(
    values: bytes,
    *,
    document_id: str,
    source: str,
    start_byte: int,
    split: str = "calibration",
    protocol: str = BINARY_PROTOCOL_V0,
) -> ByteSegment:
    """Construct a canonical in-memory segment for generated calibration data."""
    source_hash = _sha256(source.encode("utf-8"))
    document = BinaryDocument(
        document_id=document_id,
        source=source,
        values=tuple(values),
        source_sha256=source_hash,
        group_id=_sha256(values),
        protocol=protocol,
    )
    return _make_segment(
        document,
        split,
        start_byte,
        start_byte + len(values),
        values,
    )


def segment_document(
    document: CorpusDocument, config: SegmentationConfig, *, split: str
) -> tuple[ByteSegment, ...]:
    encoded = document.as_bytes()
    if not encoded:
        return ()
    offsets = (
        _utf8_offsets(document.text)
        if isinstance(document, TextDocument)
        else tuple(range(len(encoded) + 1))
    )
    segments: list[ByteSegment] = []
    requested_start = 0
    while requested_start < len(encoded):
        start_candidates = [offset for offset in offsets if offset >= requested_start]
        if not start_candidates:
            break
        start = start_candidates[0]
        end_candidates = [
            offset for offset in offsets if start < offset <= start + config.max_bytes
        ]
        if not end_candidates:
            raise ValueError("max_bytes cannot contain one complete UTF-8 scalar")
        end = end_candidates[-1]
        content = encoded[start:end]
        if len(content) >= config.min_bytes:
            segments.append(_make_segment(document, split, start, end, content))
        if end == len(encoded):
            break
        target_start = start + config.stride_bytes
        preceding_boundaries = [offset for offset in offsets if start < offset <= target_start]
        if preceding_boundaries:
            requested_start = preceding_boundaries[-1]
        else:
            requested_start = next(offset for offset in offsets if offset > start)
    return tuple(segments)


def _split_for_group(group_id: str, config: SplitConfig) -> SplitName:
    keyed = hashlib.sha256(f"{config.seed}:{group_id}".encode()).digest()
    value = int.from_bytes(keyed[:8], "big") / float(1 << 64)
    if value < config.train_fraction:
        return "train"
    if value < config.train_fraction + config.validation_fraction:
        return "validation"
    return "test"


def prepare_corpus(
    documents: Sequence[CorpusDocument],
    segmentation: SegmentationConfig,
    split_config: SplitConfig,
    *,
    require_nonempty_splits: bool = True,
) -> PreparedCorpus:
    if not documents:
        raise ValueError("corpus contains no documents")
    protocols = {document.protocol for document in documents}
    if len(protocols) != 1:
        raise ValueError("text and raw-binary protocols must be prepared separately")
    groups: dict[str, list[CorpusDocument]] = {}
    for document in documents:
        if document.group_id != _sha256(document.as_bytes()):
            raise ValueError(f"document group hash mismatch: {document.document_id}")
        groups.setdefault(document.group_id, []).append(document)
    canonical = tuple(
        min(group, key=lambda document: (document.document_id, document.source))
        for _, group in sorted(groups.items())
    )
    by_split: dict[SplitName, list[ByteSegment]] = {
        "train": [],
        "validation": [],
        "test": [],
    }
    split_groups: dict[str, list[str]] = {"train": [], "validation": [], "test": []}
    for document in canonical:
        split = _split_for_group(document.group_id, split_config)
        split_groups[split].append(document.group_id)
        by_split[split].extend(segment_document(document, segmentation, split=split))
    if require_nonempty_splits:
        for split in ("train", "validation", "test"):
            if split_config.fraction(split) > 0 and not by_split[split]:
                raise ValueError(f"positive-fraction {split} split is empty")
    return PreparedCorpus(
        documents=tuple(documents),
        canonical_documents=canonical,
        split_groups={name: tuple(values) for name, values in split_groups.items()},
        splits=CorpusSplits(**{name: tuple(values) for name, values in by_split.items()}),
        segmentation=segmentation,
        split_config=split_config,
        protocol=next(iter(protocols)),
    )


def pad_segments(
    segments: Sequence[ByteSegment], width: int
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
    if any(segment.length > width for segment in segments):
        raise ValueError("segment exceeds requested padded width")
    values = np.zeros((len(segments), width), dtype=np.int32)
    mask = np.zeros((len(segments), width), dtype=np.bool_)
    for row, segment in enumerate(segments):
        values[row, : segment.length] = segment.values
        mask[row, : segment.length] = True
    return values, mask, tuple(segment.segment_id for segment in segments)


def _ordered_record_hash(segments: Sequence[ByteSegment]) -> str:
    return _canonical_hash([segment.record_sha256 for segment in segments])


def corpus_manifest(prepared: PreparedCorpus) -> dict[str, object]:
    named = {name: prepared.splits.named(name) for name in ("train", "validation", "test")}
    group_to_split = {
        group_id: name
        for name, group_ids in prepared.split_groups.items()
        for group_id in group_ids
    }
    source_instances = [
        {
            "document_id": document.document_id,
            "group_id": document.group_id,
            "source": document.source,
            "source_sha256": document.source_sha256,
            "content_sha256": _sha256(document.as_bytes()),
            "bytes": len(document.as_bytes()),
            "assigned_split": group_to_split[document.group_id],
        }
        for document in sorted(prepared.documents, key=lambda value: value.document_id)
    ]
    byte_values = {
        value for segments in named.values() for segment in segments for value in segment.values
    }
    payload: dict[str, object] = {
        "protocol": prepared.protocol,
        "encoding": "UTF-8" if prepared.protocol == TEXT_PROTOCOL_V0 else "raw-bytes",
        "normalization": "none",
        "tokeniser_free": True,
        "byte_domain": [0, 255],
        "documents": len(prepared.documents),
        "empty_documents_recorded": sum(not document.as_bytes() for document in prepared.documents),
        "document_groups": len(prepared.canonical_documents),
        "exact_duplicate_documents_removed": len(prepared.documents)
        - len(prepared.canonical_documents),
        "source_instances": source_instances,
        "source_instances_sha256": _canonical_hash(source_instances),
        "segmentation": asdict(prepared.segmentation),
        "split_config": asdict(prepared.split_config),
        "splits": {
            name: {
                "count": len(segments),
                "bytes": sum(segment.length for segment in segments),
                "group_ids": list(prepared.split_groups[name]),
                "group_ids_sha256": _canonical_hash(list(prepared.split_groups[name])),
                "ordered_records_sha256": _ordered_record_hash(segments),
            }
            for name, segments in named.items()
        },
        "cross_split_group_disjoint": True,
        "observed_byte_values": sorted(byte_values),
        "full_byte_input_coverage": len(byte_values) == 256,
    }
    payload["manifest_sha256"] = _manifest_hash(payload)
    return payload


def _validate_segment(segment: ByteSegment, expected_split: str) -> None:
    if segment.protocol not in (TEXT_PROTOCOL_V0, BINARY_PROTOCOL_V0):
        raise ValueError(f"invalid segment protocol: {segment.protocol}")
    if segment.split != expected_split:
        raise ValueError(f"segment {segment.segment_id} has forged split ownership")
    if segment.start_byte < 0 or segment.end_byte <= segment.start_byte:
        raise ValueError(f"invalid byte offsets for segment {segment.segment_id}")
    if segment.end_byte - segment.start_byte != segment.length:
        raise ValueError(f"offset/length mismatch for segment {segment.segment_id}")
    if any(not isinstance(value, int) or not 0 <= value <= 255 for value in segment.values):
        raise ValueError(f"byte outside 0..255 in segment {segment.segment_id}")
    if _sha256(segment.as_bytes()) != segment.content_sha256:
        raise ValueError(f"content hash mismatch for segment {segment.segment_id}")
    identity = _segment_identity(
        protocol=segment.protocol,
        split=segment.split,
        document_id=segment.document_id,
        group_id=segment.group_id,
        source=segment.source,
        source_sha256=segment.source_sha256,
        start=segment.start_byte,
        end=segment.end_byte,
        content_sha256=segment.content_sha256,
    )
    if segment.segment_id != _canonical_hash(identity):
        raise ValueError(f"segment ID mismatch for segment {segment.segment_id}")
    record = {**identity, "segment_id": segment.segment_id, "values": list(segment.values)}
    if segment.record_sha256 != _canonical_hash(record):
        raise ValueError(f"record hash mismatch for segment {segment.segment_id}")


def write_segments_jsonl(path: str | Path, segments: Sequence[ByteSegment]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as stream:
        for segment in segments:
            row = asdict(segment)
            row["values"] = list(segment.values)
            stream.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")


def read_segments_jsonl(path: str | Path, *, expected_split: str) -> tuple[ByteSegment, ...]:
    segments = []
    seen: set[str] = set()
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        row = json.loads(line)
        if not isinstance(row, dict) or not isinstance(row.get("values"), list):
            raise ValueError(f"invalid segment record at line {line_number}")
        row["values"] = tuple(row["values"])
        segment = ByteSegment(**row)
        _validate_segment(segment, expected_split)
        if segment.segment_id in seen:
            raise ValueError(f"duplicate segment ID: {segment.segment_id}")
        seen.add(segment.segment_id)
        segments.append(segment)
    return tuple(segments)


def write_corpus_artifacts(directory: str | Path, prepared: PreparedCorpus) -> dict[str, object]:
    output = Path(directory)
    for name in ("train", "validation", "test"):
        write_segments_jsonl(output / f"{name}.jsonl", prepared.splits.named(name))
    manifest = corpus_manifest(prepared)
    output.mkdir(parents=True, exist_ok=True)
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def validate_corpus_artifacts(
    directory: str | Path,
) -> tuple[dict[str, object], CorpusSplits]:
    root = Path(directory)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError("dataset manifest must be a JSON object")
    if manifest.get("protocol") not in (TEXT_PROTOCOL_V0, BINARY_PROTOCOL_V0):
        raise ValueError("unsupported dataset protocol")
    expected_encoding = "UTF-8" if manifest["protocol"] == TEXT_PROTOCOL_V0 else "raw-bytes"
    if manifest.get("encoding") != expected_encoding or manifest.get("normalization") != "none":
        raise ValueError("dataset encoding/normalization contract mismatch")
    if manifest.get("tokeniser_free") is not True or manifest.get("byte_domain") != [0, 255]:
        raise ValueError("dataset is not the full-byte tokeniser-free protocol")
    if manifest.get("manifest_sha256") != _manifest_hash(manifest):
        raise ValueError("dataset manifest hash mismatch")
    try:
        segmentation = SegmentationConfig(**manifest["segmentation"])
        split_config = SplitConfig(**manifest["split_config"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("dataset preparation configuration is invalid") from error
    source_instances = manifest.get("source_instances")
    if not isinstance(source_instances, list) or manifest.get(
        "source_instances_sha256"
    ) != _canonical_hash(source_instances):
        raise ValueError("source-instance provenance hash mismatch")
    documents: dict[str, dict[str, object]] = {}
    for record in source_instances:
        if not isinstance(record, dict):
            raise ValueError("source-instance record must be an object")
        required = {
            "document_id",
            "group_id",
            "source",
            "source_sha256",
            "content_sha256",
            "bytes",
            "assigned_split",
        }
        if set(record) != required:
            raise ValueError("source-instance record has an invalid schema")
        document_id = str(record["document_id"])
        if document_id in documents:
            raise ValueError(f"duplicate document ID: {document_id}")
        if record["group_id"] != record["content_sha256"]:
            raise ValueError(f"document group/content mismatch: {document_id}")
        if not all(
            _is_sha256(record[name]) for name in ("group_id", "source_sha256", "content_sha256")
        ):
            raise ValueError(f"invalid source-instance hashes: {document_id}")
        expected_id = _canonical_hash(
            {
                "protocol": manifest["protocol"],
                "source": record["source"],
                "source_sha256": record["source_sha256"],
                "content_sha256": record["content_sha256"],
            }
        )
        if (
            document_id != expected_id
            or not isinstance(record["bytes"], int)
            or record["bytes"] < 0
        ):
            raise ValueError(f"invalid source-instance identity: {document_id}")
        if record["assigned_split"] not in ("train", "validation", "test"):
            raise ValueError(f"invalid document split assignment: {document_id}")
        if record["assigned_split"] != _split_for_group(str(record["group_id"]), split_config):
            raise ValueError(f"noncanonical document split assignment: {document_id}")
        documents[document_id] = record

    unique_groups = {str(record["group_id"]) for record in documents.values()}
    empty_documents = sum(record["bytes"] == 0 for record in documents.values())
    if (
        manifest.get("documents") != len(documents)
        or manifest.get("document_groups") != len(unique_groups)
        or manifest.get("exact_duplicate_documents_removed") != len(documents) - len(unique_groups)
        or manifest.get("empty_documents_recorded") != empty_documents
    ):
        raise ValueError("dataset document accounting mismatch")

    named = {
        name: read_segments_jsonl(root / f"{name}.jsonl", expected_split=name)
        for name in ("train", "validation", "test")
    }
    declared = manifest.get("splits")
    if not isinstance(declared, dict):
        raise ValueError("dataset manifest has no split records")
    group_sets: dict[str, set[str]] = {}
    for name, segments in named.items():
        record = declared.get(name)
        if not isinstance(record, dict):
            raise ValueError(f"dataset manifest has no {name} split record")
        declared_groups = record.get("group_ids")
        if not isinstance(declared_groups, list) or not all(
            isinstance(group_id, str) for group_id in declared_groups
        ):
            raise ValueError(f"{name} split has invalid group IDs")
        expected_record = {
            "count": len(segments),
            "bytes": sum(segment.length for segment in segments),
            "group_ids": declared_groups,
            "group_ids_sha256": _canonical_hash(declared_groups),
            "ordered_records_sha256": _ordered_record_hash(segments),
        }
        if record != expected_record:
            raise ValueError(f"{name} split differs from its dataset manifest")
        assigned_groups = {
            str(document["group_id"])
            for document in documents.values()
            if document["assigned_split"] == name
        }
        if (
            len(declared_groups) != len(set(declared_groups))
            or set(declared_groups) != assigned_groups
        ):
            raise ValueError(f"{name} document-group ownership mismatch")
        for segment in segments:
            document = documents.get(segment.document_id)
            if document is None:
                raise ValueError(f"segment references unknown document: {segment.document_id}")
            document_bytes = document["bytes"]
            if not isinstance(document_bytes, int):
                raise ValueError(f"invalid document byte count: {segment.document_id}")
            if (
                segment.protocol != manifest["protocol"]
                or segment.group_id != document["group_id"]
                or segment.source != document["source"]
                or segment.source_sha256 != document["source_sha256"]
                or segment.end_byte > document_bytes
                or segment.group_id not in declared_groups
                or not segmentation.min_bytes <= segment.length <= segmentation.max_bytes
            ):
                raise ValueError(f"segment/source binding mismatch: {segment.segment_id}")
        group_sets[name] = set(declared_groups)
    disjoint = not (
        group_sets["train"] & group_sets["validation"]
        or group_sets["train"] & group_sets["test"]
        or group_sets["validation"] & group_sets["test"]
    )
    if not disjoint or manifest.get("cross_split_group_disjoint") is not True:
        raise ValueError("dataset document groups are not split-disjoint")
    observed = sorted(
        {value for segments in named.values() for segment in segments for value in segment.values}
    )
    if manifest.get("observed_byte_values") != observed or manifest.get(
        "full_byte_input_coverage"
    ) != (len(observed) == 256):
        raise ValueError("dataset byte-coverage accounting mismatch")
    return manifest, CorpusSplits(**named)
