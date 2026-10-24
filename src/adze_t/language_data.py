"""Deterministic, tokeniser-free natural-text dataset preparation.

This module deliberately has no model or JAX dependency.  It prepares raw
UTF-8 bytes and explicit masks; byte value zero is data, never padding.
"""

from __future__ import annotations

import hashlib
import json
import unicodedata
from bisect import bisect_left, bisect_right
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Literal, Sequence

import numpy as np


Normalization = Literal["none", "lf", "nfc"]
SplitName = Literal["train", "validation", "test"]


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical_json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _manifest_hash(payload: dict[str, object]) -> str:
    unsigned = {key: value for key, value in payload.items() if key != "manifest_sha256"}
    return _sha256(_canonical_json(unsigned))


def _ordered_content_hash(segments: Sequence[ByteSegment]) -> str:
    return _sha256(_canonical_json([segment.content_sha256 for segment in segments]))


@dataclass(frozen=True)
class TextDocument:
    document_id: str
    source: str
    text: str


@dataclass(frozen=True)
class ByteSegment:
    segment_id: str
    document_id: str
    source: str
    start_byte: int
    end_byte: int
    content_sha256: str
    values: tuple[int, ...]

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
    utf8_boundary_safe: bool = True

    def __post_init__(self) -> None:
        if self.max_bytes < 4:
            raise ValueError("max_bytes must accommodate one complete UTF-8 scalar")
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


@dataclass(frozen=True)
class CorpusSplits:
    train: tuple[ByteSegment, ...]
    validation: tuple[ByteSegment, ...]
    test: tuple[ByteSegment, ...]

    def named(self, split: SplitName) -> tuple[ByteSegment, ...]:
        return getattr(self, split)


def normalize_text(text: str, policy: Normalization = "none") -> str:
    if policy == "none":
        return text
    if policy == "lf":
        return text.replace("\r\n", "\n").replace("\r", "\n")
    if policy == "nfc":
        return unicodedata.normalize("NFC", text)
    raise ValueError(f"unsupported normalization policy: {policy}")


def encode_utf8(text: str) -> tuple[int, ...]:
    """Encode text as the complete 0..255 byte alphabet."""
    return tuple(text.encode("utf-8"))


def read_text_documents(
    paths: Sequence[str | Path],
    *,
    format: Literal["text", "jsonl"] = "text",
    text_field: str = "text",
    normalization: Normalization = "none",
) -> tuple[TextDocument, ...]:
    """Read local corpus files without downloading or implicit shuffling."""
    documents: list[TextDocument] = []
    for raw_path in sorted((Path(path) for path in paths), key=lambda item: item.as_posix()):
        source = raw_path.as_posix()
        if format == "text":
            text = normalize_text(raw_path.read_text(encoding="utf-8"), normalization)
            document_id = _sha256(source.encode() + b"\0" + text.encode("utf-8"))
            documents.append(TextDocument(document_id, source, text))
            continue
        if format != "jsonl":
            raise ValueError(f"unsupported corpus format: {format}")
        for line_number, line in enumerate(raw_path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict) or not isinstance(row.get(text_field), str):
                raise ValueError(f"{source}:{line_number}: missing string field {text_field!r}")
            text = normalize_text(row[text_field], normalization)
            identity = f"{source}:{line_number}".encode() + b"\0" + text.encode("utf-8")
            documents.append(TextDocument(_sha256(identity), source, text))
    return tuple(documents)


def _utf8_offsets(text: str) -> tuple[int, ...]:
    offsets = [0]
    total = 0
    for scalar in text:
        total += len(scalar.encode("utf-8"))
        offsets.append(total)
    return tuple(offsets)


def segment_document(document: TextDocument, config: SegmentationConfig) -> tuple[ByteSegment, ...]:
    encoded = document.text.encode("utf-8")
    if not encoded:
        return ()
    offsets = (
        _utf8_offsets(document.text)
        if config.utf8_boundary_safe
        else tuple(range(len(encoded) + 1))
    )
    segments: list[ByteSegment] = []
    requested_start = 0
    while requested_start < len(encoded):
        start_index = bisect_left(offsets, requested_start)
        start = offsets[min(start_index, len(offsets) - 1)]
        if start >= len(encoded):
            break
        requested_end = min(start + config.max_bytes, len(encoded))
        end_index = bisect_right(offsets, requested_end) - 1
        end = offsets[max(start_index + 1, end_index)]
        content = encoded[start:end]
        if len(content) >= config.min_bytes:
            digest = _sha256(content)
            identity = _canonical_json(
                {"document_id": document.document_id, "start": start, "end": end, "sha256": digest}
            )
            segments.append(
                ByteSegment(
                    segment_id=_sha256(identity),
                    document_id=document.document_id,
                    source=document.source,
                    start_byte=start,
                    end_byte=end,
                    content_sha256=digest,
                    values=tuple(content),
                )
            )
        if end == len(encoded):
            break
        requested_start = start + config.stride_bytes
    return tuple(segments)


def segment_corpus(
    documents: Iterable[TextDocument], config: SegmentationConfig
) -> tuple[ByteSegment, ...]:
    return tuple(
        segment for document in documents for segment in segment_document(document, config)
    )


def deduplicate_segments(segments: Iterable[ByteSegment]) -> tuple[ByteSegment, ...]:
    """Deduplicate globally before splitting to prevent cross-split leakage."""
    by_content: dict[str, ByteSegment] = {}
    for segment in segments:
        incumbent = by_content.get(segment.content_sha256)
        if incumbent is None or segment.segment_id < incumbent.segment_id:
            by_content[segment.content_sha256] = segment
    return tuple(sorted(by_content.values(), key=lambda item: item.segment_id))


def _split_for_digest(digest: str, config: SplitConfig) -> SplitName:
    keyed = hashlib.sha256(f"{config.seed}:{digest}".encode()).digest()
    value = int.from_bytes(keyed[:8], "big") / float(1 << 64)
    if value < config.train_fraction:
        return "train"
    if value < config.train_fraction + config.validation_fraction:
        return "validation"
    return "test"


def build_corpus_splits(segments: Iterable[ByteSegment], config: SplitConfig) -> CorpusSplits:
    buckets: dict[SplitName, list[ByteSegment]] = {"train": [], "validation": [], "test": []}
    for segment in deduplicate_segments(segments):
        buckets[_split_for_digest(segment.content_sha256, config)].append(segment)
    return CorpusSplits(**{name: tuple(values) for name, values in buckets.items()})


def pad_segments(
    segments: Sequence[ByteSegment], width: int
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
    """Return integer bytes and a separate mask so byte 0 remains representable."""
    if any(segment.length > width for segment in segments):
        raise ValueError("segment exceeds requested padded width")
    values = np.zeros((len(segments), width), dtype=np.int32)
    mask = np.zeros((len(segments), width), dtype=np.bool_)
    for row, segment in enumerate(segments):
        values[row, : segment.length] = segment.values
        mask[row, : segment.length] = True
    return values, mask, tuple(segment.segment_id for segment in segments)


def split_manifest(
    documents: Sequence[TextDocument],
    segments_before_dedup: Sequence[ByteSegment],
    splits: CorpusSplits,
    segmentation: SegmentationConfig,
    split_config: SplitConfig,
) -> dict[str, object]:
    named = {name: splits.named(name) for name in ("train", "validation", "test")}
    all_digests = [segment.content_sha256 for values in named.values() for segment in values]
    split_digests = {
        name: [item.content_sha256 for item in values] for name, values in named.items()
    }
    disjoint = not (
        set(split_digests["train"]) & set(split_digests["validation"])
        or set(split_digests["train"]) & set(split_digests["test"])
        or set(split_digests["validation"]) & set(split_digests["test"])
    )
    byte_values = {
        value for values in named.values() for segment in values for value in segment.values
    }
    payload: dict[str, object] = {
        "protocol": "PHASE_I_NATURAL_TEXT_BYTES_V0",
        "encoding": "UTF-8",
        "tokeniser_free": True,
        "byte_domain": [0, 255],
        "documents": len(documents),
        "segments_before_dedup": len(segments_before_dedup),
        "segments_after_dedup": len(all_digests),
        "duplicates_removed": len(segments_before_dedup) - len(all_digests),
        "segmentation": asdict(segmentation),
        "split_config": asdict(split_config),
        "splits": {
            name: {
                "count": len(values),
                "bytes": sum(item.length for item in values),
                "ordered_content_sha256": _ordered_content_hash(values),
            }
            for name, values in named.items()
        },
        "cross_split_content_disjoint": disjoint,
        "observed_byte_values": sorted(byte_values),
        "full_byte_coverage": len(byte_values) == 256,
    }
    payload["manifest_sha256"] = _manifest_hash(payload)
    return payload


def write_segments_jsonl(path: str | Path, segments: Sequence[ByteSegment]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as stream:
        for segment in segments:
            row = asdict(segment)
            row["values"] = list(segment.values)
            stream.write(json.dumps(row, sort_keys=True) + "\n")


def read_segments_jsonl(path: str | Path) -> tuple[ByteSegment, ...]:
    segments = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        row["values"] = tuple(row["values"])
        segment = ByteSegment(**row)
        if _sha256(segment.as_bytes()) != segment.content_sha256:
            raise ValueError(f"content hash mismatch for segment {segment.segment_id}")
        segments.append(segment)
    return tuple(segments)


def validate_corpus_artifacts(
    directory: str | Path,
) -> tuple[dict[str, object], CorpusSplits]:
    """Load a prepared corpus only when every artifact matches its manifest."""
    root = Path(directory)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError("dataset manifest must be a JSON object")
    if manifest.get("protocol") != "PHASE_I_NATURAL_TEXT_BYTES_V0":
        raise ValueError("dataset protocol is not PHASE_I_NATURAL_TEXT_BYTES_V0")
    if manifest.get("encoding") != "UTF-8" or manifest.get("tokeniser_free") is not True:
        raise ValueError("dataset is not the raw UTF-8 tokeniser-free protocol")
    if manifest.get("byte_domain") != [0, 255]:
        raise ValueError("dataset byte domain must be exactly [0, 255]")
    if manifest.get("manifest_sha256") != _manifest_hash(manifest):
        raise ValueError("dataset manifest hash mismatch")

    named = {
        name: read_segments_jsonl(root / f"{name}.jsonl")
        for name in ("train", "validation", "test")
    }
    declared = manifest.get("splits")
    if not isinstance(declared, dict):
        raise ValueError("dataset manifest has no split records")
    for name, segments in named.items():
        record = declared.get(name)
        if not isinstance(record, dict):
            raise ValueError(f"dataset manifest has no {name} split record")
        observed = {
            "count": len(segments),
            "bytes": sum(segment.length for segment in segments),
            "ordered_content_sha256": _ordered_content_hash(segments),
        }
        if record != observed:
            raise ValueError(f"{name} split differs from its dataset manifest")

    digest_sets = {
        name: {segment.content_sha256 for segment in segments} for name, segments in named.items()
    }
    disjoint = not (
        digest_sets["train"] & digest_sets["validation"]
        or digest_sets["train"] & digest_sets["test"]
        or digest_sets["validation"] & digest_sets["test"]
    )
    if not disjoint or manifest.get("cross_split_content_disjoint") is not True:
        raise ValueError("dataset splits are not content-disjoint")
    return manifest, CorpusSplits(**named)
