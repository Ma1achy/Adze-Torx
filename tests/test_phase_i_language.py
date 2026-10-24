import json

import numpy as np
import pytest

from adze_t.language_data import (
    SegmentationConfig,
    SplitConfig,
    TextDocument,
    build_corpus_splits,
    encode_utf8,
    pad_segments,
    read_segments_jsonl,
    read_text_documents,
    segment_corpus,
    segment_document,
    split_manifest,
    validate_corpus_artifacts,
    write_segments_jsonl,
)
from adze_t.language_eval import (
    ComputeObservation,
    conditional_case,
    evaluate_outputs,
    infilling_case,
    run_generation_harness,
)
from adze_t.phase_i import (
    ACCEPTED_PHASE_B_CODEC_PATH,
    FULL_BYTE_CODEC_V0,
    SmallScaleLMRecipe,
    candidate_manifest,
    deterministic_epoch_order,
    full_byte_calibration_segments,
    validate_recipe,
)


def test_utf8_encoding_uses_raw_bytes_and_allows_zero():
    assert encode_utf8("A\x00£🙂") == tuple("A\x00£🙂".encode("utf-8"))
    assert 0 in encode_utf8("A\x00£🙂")


def test_local_text_and_jsonl_ingestion_is_ordered_and_explicit(tmp_path):
    second = tmp_path / "b.txt"
    first = tmp_path / "a.txt"
    second.write_text("second\r\n")
    first.write_text("first\r")
    documents = read_text_documents((second, first), normalization="lf")
    assert [item.source for item in documents] == [first.as_posix(), second.as_posix()]
    assert [item.text for item in documents] == ["first\n", "second\n"]
    rows = tmp_path / "rows.jsonl"
    rows.write_text('{"body":"one"}\n{"body":"two"}\n')
    assert [
        item.text for item in read_text_documents((rows,), format="jsonl", text_field="body")
    ] == [
        "one",
        "two",
    ]


def test_segmentation_preserves_utf8_scalar_boundaries_and_round_trips():
    document = TextDocument("doc", "memory", "αβγ🙂z" * 5)
    segments = segment_document(document, SegmentationConfig(max_bytes=9, stride_bytes=7))
    assert segments
    assert all(segment.length <= 9 for segment in segments)
    assert all(segment.as_bytes().decode("utf-8") for segment in segments)
    assert segments == segment_document(document, SegmentationConfig(max_bytes=9, stride_bytes=7))


def test_global_dedup_precedes_deterministic_disjoint_split():
    documents = tuple(
        TextDocument(f"doc-{index}", "memory", f"unique text {index:04d}") for index in range(200)
    ) + (TextDocument("duplicate", "memory", "unique text 0000"),)
    segmentation = SegmentationConfig(64, 64)
    segments = segment_corpus(documents, segmentation)
    split_config = SplitConfig(72, 0.8, 0.1, 0.1)
    first = build_corpus_splits(segments, split_config)
    second = build_corpus_splits(reversed(segments), split_config)
    assert first == second
    sets = [
        set(item.content_sha256 for item in getattr(first, name))
        for name in ("train", "validation", "test")
    ]
    assert not sets[0] & sets[1]
    assert not sets[0] & sets[2]
    assert not sets[1] & sets[2]
    assert sum(map(len, sets)) == 200


def test_padding_mask_does_not_conflate_zero_byte_with_padding():
    segment = full_byte_calibration_segments(128)[0]
    values, mask, _ = pad_segments((segment,), 130)
    assert values[0, 0] == 0
    assert bool(mask[0, 0])
    assert values[0, 129] == 0
    assert not bool(mask[0, 129])
    assert values.dtype == np.int32


def test_segment_jsonl_round_trip_and_hash_validation(tmp_path):
    segments = full_byte_calibration_segments(128)
    path = tmp_path / "segments.jsonl"
    write_segments_jsonl(path, segments)
    assert read_segments_jsonl(path) == segments
    row = json.loads(path.read_text().splitlines()[0])
    row["values"][0] = 99
    path.write_text(json.dumps(row) + "\n")
    with pytest.raises(ValueError, match="content hash mismatch"):
        read_segments_jsonl(path)


def test_prepared_bundle_rejects_stale_or_mixed_split_artifacts(tmp_path):
    segments = full_byte_calibration_segments(64)
    splits = build_corpus_splits(segments, SplitConfig(4, 1.0, 0.0, 0.0))
    for name in ("train", "validation", "test"):
        write_segments_jsonl(tmp_path / f"{name}.jsonl", splits.named(name))
    manifest = split_manifest(
        (), segments, splits, SegmentationConfig(64, 64), SplitConfig(4, 1.0, 0.0, 0.0)
    )
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    loaded_manifest, loaded_splits = validate_corpus_artifacts(tmp_path)
    assert loaded_manifest == manifest
    assert loaded_splits == splits

    rows = (tmp_path / "train.jsonl").read_text().splitlines()
    (tmp_path / "train.jsonl").write_text("\n".join(reversed(rows)) + "\n")
    with pytest.raises(ValueError, match="train split differs"):
        validate_corpus_artifacts(tmp_path)


def test_prepared_bundle_rejects_tampered_manifest(tmp_path):
    segments = full_byte_calibration_segments(128)
    splits = build_corpus_splits(segments, SplitConfig(4, 1.0, 0.0, 0.0))
    for name in ("train", "validation", "test"):
        write_segments_jsonl(tmp_path / f"{name}.jsonl", splits.named(name))
    manifest = split_manifest(
        (), segments, splits, SegmentationConfig(128, 128), SplitConfig(4, 1.0, 0.0, 0.0)
    )
    manifest["protocol"] = "stale"
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="dataset protocol"):
        validate_corpus_artifacts(tmp_path)


def test_manifest_records_dedup_disjointness_and_byte_coverage():
    documents = (TextDocument("calibration", "generated", bytes(range(256)).decode("latin1")),)
    # Latin-1 text is re-encoded as UTF-8, so use generated byte segments for the coverage assertion.
    segments = full_byte_calibration_segments(128)
    splits = build_corpus_splits(segments, SplitConfig(4, 1.0, 0.0, 0.0))
    manifest = split_manifest(
        documents, segments, splits, SegmentationConfig(128, 128), SplitConfig(4, 1.0, 0.0, 0.0)
    )
    assert manifest["cross_split_content_disjoint"]
    assert manifest["full_byte_coverage"]
    assert manifest["observed_byte_values"] == list(range(256))
    assert len(str(manifest["manifest_sha256"])) == 64


def test_full_byte_codec_candidate_is_fresh_and_versioned():
    manifest: dict[str, object] = {
        "protocol": "PHASE_I_NATURAL_TEXT_BYTES_V0",
        "manifest_sha256": "a" * 64,
    }
    candidate = candidate_manifest(manifest)
    assert candidate["candidate"]["version"] == FULL_BYTE_CODEC_V0
    assert candidate["candidate"]["accepted_codec_parent"] is None
    assert candidate["accepted_phase_b_codec"] == {
        "path": ACCEPTED_PHASE_B_CODEC_PATH,
        "policy": "immutable_not_a_parent",
    }
    assert candidate["authoritative_training_allowed"] is False


def test_full_byte_calibration_and_epoch_order_are_complete_and_reproducible():
    segments = full_byte_calibration_segments(31)
    assert {value for segment in segments for value in segment.values} == set(range(256))
    ids = tuple(segment.segment_id for segment in segments)
    first = deterministic_epoch_order(ids, seed=3, epoch=0)
    assert first == deterministic_epoch_order(ids, seed=3, epoch=0)
    assert sorted(first) == list(range(len(ids)))
    assert first != deterministic_epoch_order(ids, seed=3, epoch=1)


def test_recipe_stays_non_authoritative_until_integration_gates_pass():
    recipe = SmallScaleLMRecipe("smoke", "smoke", "b" * 64, FULL_BYTE_CODEC_V0, 8)
    validate_recipe(recipe)
    with pytest.raises(ValueError, match="authoritative language training"):
        validate_recipe(
            SmallScaleLMRecipe("bad", "smoke", "b" * 64, FULL_BYTE_CODEC_V0, 8, authoritative=True)
        )


def test_conditional_and_infilling_harness_metrics():
    conditional = conditional_case("conditional", b"abcdef", 3)
    infill = infilling_case("infill", "a£z".encode(), 1, 3)
    metrics = evaluate_outputs(
        (conditional, infill),
        (b"def", "£".encode()),
        ComputeObservation(2.0, peak_rss_bytes=100, model_applications=4),
    )
    assert metrics["exact_accuracy"] == 1.0
    assert metrics["aligned_byte_accuracy"] == 1.0
    assert metrics["utf8_valid_fraction"] == 1.0
    assert metrics["generated_bytes"] == 5
    assert metrics["bytes_per_second"] == 2.5


def test_generation_harness_passes_suffix_and_deterministic_seed():
    calls = []

    class Generator:
        def generate(self, prompt, *, suffix, max_bytes, seed):
            calls.append((prompt, suffix, max_bytes, seed))
            return b"x"

    cases = (infilling_case("one", b"abc", 1, 2),)
    outputs, _ = run_generation_harness(Generator(), cases, seed=41, max_bytes=8)
    assert outputs == (b"x",)
    assert calls == [(b"a", b"c", 1, 41)]
