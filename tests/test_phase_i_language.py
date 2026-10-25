from __future__ import annotations

import json
import shutil
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest

from adze_t.language_data import (
    BINARY_PROTOCOL_V0,
    TEXT_PROTOCOL_V0,
    SegmentationConfig,
    SplitConfig,
    prepare_corpus,
    read_binary_documents,
    read_segments_jsonl,
    read_text_documents,
    validate_corpus_artifacts,
    write_corpus_artifacts,
)
from adze_t.language_eval import (
    ComputeObservation,
    EvaluationTask,
    GenerationRequest,
    conditional_case,
    evaluate_outputs,
    infilling_case,
    reconstruction_case,
    run_generation_harness,
    unconditional_case,
)
from adze_t.phase_i import (
    CODEC_RUN_PROTOCOL_V0,
    EVIDENCE_SCHEMA_V0,
    FULL_BYTE_CODEC_V0,
    PINNED_TORX_REVISION,
    ByteEvidence,
    CodecRunProtocol,
    RuntimeIdentity,
    SmallScaleLMRecipe,
    byte_histogram,
    calibration_prefixed_batch,
    candidate_manifest,
    codec_state_tree_description,
    codec_batch_at,
    codec_batch_count,
    deterministic_epoch_order,
    evaluation_coverage_evidence,
    full_byte_calibration_segments,
    load_language_scales,
    reconstruction_evidence,
    training_exposure_evidence,
    validate_codec_resume_state,
    validate_recipe,
    validate_run_protocol,
)
from adze_t.phase_i_artifacts import (
    checkpoint_tree_identity,
    reconcile_transactions,
    recover_transactions,
    save_transaction,
)


ROOT = Path(__file__).resolve().parents[1]
SCALES = ROOT / "configs/phase_i_language_scales_v0.yaml"


def _prepared(tmp_path, contents, *, binary=False):
    paths = []
    for index, content in enumerate(contents):
        path = tmp_path / f"source-{index}.{'bin' if binary else 'txt'}"
        path.write_bytes(content if isinstance(content, bytes) else content.encode("utf-8"))
        paths.append(path)
    documents = read_binary_documents(paths) if binary else read_text_documents(paths)
    return prepare_corpus(
        documents,
        SegmentationConfig(8, 4),
        SplitConfig(7, 1.0, 0.0, 0.0),
    )


def _protocol(**changes):
    value = CodecRunProtocol(
        version=CODEC_RUN_PROTOCOL_V0,
        source_commit="a" * 40,
        implementation_sha256="b" * 64,
        dataset_manifest_sha256="c" * 64,
        dataset_protocol=TEXT_PROTOCOL_V0,
        scale_name="smoke",
        resolved_config_sha256="d" * 64,
        candidate_version=FULL_BYTE_CODEC_V0,
        initialization_seed=1,
        training_seed=2,
        sequence_bytes=128,
        batch_size=8,
        gradient_accumulation=1,
        max_updates=100,
        checkpoint_every_updates=10,
        validation_every_updates=10,
        optimizer="adamw",
        learning_rate=3e-4,
        weight_decay=0.01,
        grad_clip_norm=1.0,
        schedule="constant",
        calibration_policy="full_byte_prefix_each_epoch_v0",
        evaluation_protocol="codec_eval_v0",
        structure_protocol="teacher_only_v0",
        refinement_protocol="disabled_r0",
        q=0,
        s=0,
        r=0,
        runtime=RuntimeIdentity("3.12", "test", "x", "y", PINNED_TORX_REVISION, "cpu", False),
    )
    return replace(value, **changes)


def test_strict_utf8_preserves_raw_newlines_and_rejects_invalid_input(tmp_path):
    path = tmp_path / "raw.txt"
    path.write_bytes(b"a\r\nb\r")
    document = read_text_documents((path,))[0]
    assert document.text == "a\r\nb\r"
    assert document.as_bytes() == b"a\r\nb\r"
    path.write_bytes(b"\xff")
    with pytest.raises(UnicodeDecodeError):
        read_text_documents((path,))


def test_nonoverlapping_utf8_segmentation_never_drops_a_crossing_scalar(tmp_path):
    path = tmp_path / "unicode.txt"
    path.write_text("aa££", encoding="utf-8")
    document = read_text_documents((path,))[0]
    prepared = prepare_corpus(
        (document,),
        SegmentationConfig(5, 5),
        SplitConfig(1, 1.0, 0.0, 0.0),
    )
    assert b"".join(segment.as_bytes() for segment in prepared.splits.train) == path.read_bytes()


def test_canonically_equivalent_unicode_is_not_normalized_or_deduplicated(tmp_path):
    prepared = _prepared(tmp_path, ["é", "e\N{COMBINING ACUTE ACCENT}"])
    assert len(prepared.canonical_documents) == 2
    assert {document.as_bytes() for document in prepared.canonical_documents} == {
        "é".encode(),
        "e\N{COMBINING ACUTE ACCENT}".encode(),
    }


def test_text_and_binary_are_distinct_and_binary_exposes_every_byte(tmp_path):
    path = tmp_path / "all.bin"
    path.write_bytes(bytes(range(256)))
    document = read_binary_documents((path,))[0]
    assert document.protocol == BINARY_PROTOCOL_V0
    assert set(document.values) == set(range(256))
    with pytest.raises(UnicodeDecodeError):
        read_text_documents((path,))


def test_exact_duplicate_documents_share_one_group_and_one_canonical_segmentation(tmp_path):
    prepared = _prepared(tmp_path, ["abcdefghijk", "abcdefghijk"])
    assert len(prepared.documents) == 2
    assert len(prepared.canonical_documents) == 1
    assert len({segment.group_id for segment in prepared.splits.train}) == 1
    starts = [segment.start_byte for segment in prepared.splits.train]
    assert starts == [0, 4]


def test_overlapping_windows_from_one_document_cannot_cross_splits(tmp_path):
    path = tmp_path / "long.txt"
    path.write_text("abcdefgh" * 20, encoding="utf-8")
    documents = read_text_documents((path,))
    prepared = prepare_corpus(
        documents,
        SegmentationConfig(16, 8),
        SplitConfig(9, 0.34, 0.33, 0.33),
        require_nonempty_splits=False,
    )
    nonempty = [name for name in ("train", "validation", "test") if prepared.splits.named(name)]
    assert len(nonempty) == 1


def test_positive_fraction_empty_split_fails_closed(tmp_path):
    path = tmp_path / "one.txt"
    path.write_text("one document", encoding="utf-8")
    with pytest.raises(ValueError, match="positive-fraction .* split is empty"):
        prepare_corpus(
            read_text_documents((path,)),
            SegmentationConfig(16, 16),
            SplitConfig(3, 0.8, 0.1, 0.1),
        )


def test_empty_document_is_recorded_with_group_split_ownership(tmp_path):
    prepared = _prepared(tmp_path, ["", "nonempty document"])
    manifest = write_corpus_artifacts(tmp_path / "dataset", prepared)
    assert manifest["empty_documents_recorded"] == 1
    validated, _ = validate_corpus_artifacts(tmp_path / "dataset")
    source_instances = cast(list[dict[str, Any]], validated["source_instances"])
    empty = [record for record in source_instances if record["bytes"] == 0]
    assert len(empty) == 1
    assert empty[0]["assigned_split"] == "train"


def test_artifacts_bind_group_source_offsets_ids_order_and_split(tmp_path):
    prepared = _prepared(tmp_path, ["abcdefghijk", "different text"])
    write_corpus_artifacts(tmp_path / "dataset", prepared)
    manifest, splits = validate_corpus_artifacts(tmp_path / "dataset")
    assert manifest["protocol"] == TEXT_PROTOCOL_V0
    assert splits == prepared.splits

    path = tmp_path / "dataset/train.jsonl"
    rows = path.read_text().splitlines()
    row = json.loads(rows[0])
    row["start_byte"] = -1
    rows[0] = json.dumps(row)
    path.write_text("\n".join(rows) + "\n")
    with pytest.raises(ValueError, match="offset"):
        validate_corpus_artifacts(tmp_path / "dataset")


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("segment_id", "0" * 64, "segment ID"),
        ("split", "test", "forged split"),
        ("source", "forged", "segment ID"),
        ("group_id", "1" * 64, "segment ID"),
        ("end_byte", 999, "offset/length"),
        ("values", [999], "offset/length|outside 0..255"),
    ],
)
def test_adversarial_segment_mutations_are_rejected(tmp_path, field, value, message):
    prepared = _prepared(tmp_path, ["abcdefghijk"])
    write_corpus_artifacts(tmp_path / "dataset", prepared)
    path = tmp_path / "dataset/train.jsonl"
    row = json.loads(path.read_text().splitlines()[0])
    row[field] = value
    path.write_text(json.dumps(row) + "\n")
    with pytest.raises(ValueError, match=message):
        read_segments_jsonl(path, expected_split="train")


def test_duplicate_segment_ids_and_changed_order_are_rejected(tmp_path):
    prepared = _prepared(tmp_path, ["abcdefghijk"])
    write_corpus_artifacts(tmp_path / "dataset", prepared)
    path = tmp_path / "dataset/train.jsonl"
    rows = path.read_text().splitlines()
    path.write_text("\n".join(rows + [rows[0]]) + "\n")
    with pytest.raises(ValueError, match="duplicate segment ID"):
        validate_corpus_artifacts(tmp_path / "dataset")

    write_corpus_artifacts(tmp_path / "dataset", prepared)
    rows = path.read_text().splitlines()
    path.write_text("\n".join(reversed(rows)) + "\n")
    with pytest.raises(ValueError, match="differs from its dataset manifest"):
        validate_corpus_artifacts(tmp_path / "dataset")


def test_manifest_source_binding_rejects_forged_document_record(tmp_path):
    prepared = _prepared(tmp_path, ["abcdefghijk"])
    write_corpus_artifacts(tmp_path / "dataset", prepared)
    manifest_path = tmp_path / "dataset/manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["source_instances"][0]["bytes"] = 1
    from adze_t.phase_i import canonical_hash

    manifest["source_instances_sha256"] = canonical_hash(manifest["source_instances"])
    unsigned = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    manifest["manifest_sha256"] = canonical_hash(unsigned)
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="segment/source binding mismatch"):
        validate_corpus_artifacts(tmp_path / "dataset")


def test_zero_byte_is_data_not_padding(tmp_path):
    prepared = _prepared(tmp_path, [b"\x00\x01"], binary=True)
    segment = prepared.splits.train[0]
    from adze_t.language_data import pad_segments

    values, mask, _ = pad_segments((segment,), 4)
    assert values.tolist() == [[0, 1, 0, 0]]
    assert mask.tolist() == [[True, True, False, False]]


def test_full_byte_calibration_and_epoch_order_are_complete_and_reproducible():
    segments = full_byte_calibration_segments(31)
    assert {value for segment in segments for value in segment.values} == set(range(256))
    assert len(full_byte_calibration_segments(512)) == 1
    ids = tuple(segment.segment_id for segment in segments)
    assert deterministic_epoch_order(ids, seed=3, epoch=0) == deterministic_epoch_order(
        ids, seed=3, epoch=0
    )
    assert deterministic_epoch_order(ids, seed=3, epoch=0) != deterministic_epoch_order(
        ids, seed=3, epoch=1
    )


def test_incremental_batch_materializes_only_requested_slice():
    segments = full_byte_calibration_segments(16)
    order = tuple(range(len(segments)))
    assert codec_batch_count(len(segments), 3) == 6
    batch = codec_batch_at(segments, width=16, batch_size=3, order=order, batch_index=2)
    assert batch["segment_ids"] == tuple(segment.segment_id for segment in segments[6:9])
    assert batch["target"].shape == (3, 16)


def test_calibration_prefix_guarantees_all_byte_exposure_before_natural_batches(tmp_path):
    natural = _prepared(tmp_path, ["ordinary natural bytes"]).splits.train
    calibration = full_byte_calibration_segments(128)
    batch, epoch = calibration_prefixed_batch(
        natural,
        calibration,
        width=128,
        batch_size=2,
        seed=7,
        update=1,
    )
    assert epoch == 0
    assert set(np.asarray(batch["target"])[np.asarray(batch["target_mask"])].tolist()) == set(
        range(256)
    )


def test_byte_evidence_keeps_exposure_coverage_and_reconstruction_distinct():
    values = np.array([[0, 255, 0]], dtype=np.int32)
    mask = np.array([[True, True, False]])
    counts = byte_histogram(values, mask)
    exposure = training_exposure_evidence(counts)
    coverage = evaluation_coverage_evidence((0,) * 256)
    reconstruction = reconstruction_evidence(counts, (0,) * 255 + (1,))
    assert exposure.kind == "training_exposure"
    assert coverage.kind == "evaluation_coverage"
    assert reconstruction.kind == "reconstruction"
    assert exposure.counts[0] == 1 and exposure.counts[255] == 1
    with pytest.raises(ValueError, match="256 correct counts"):
        ByteEvidence(EVIDENCE_SCHEMA_V0, "reconstruction", counts)
    with pytest.raises(ValueError, match="nonnegative counts"):
        training_exposure_evidence((0,) * 255 + (False,))
    with pytest.raises(ValueError, match="invalid reconstruction counts"):
        reconstruction_evidence((1,) * 256, cast(tuple[int, ...], (1,) * 255 + (0.5,)))


def test_canonical_scale_loader_has_one_resolved_hash_and_gates_proposals():
    first = load_language_scales(SCALES)
    second = load_language_scales(SCALES)
    assert first == second
    assert first["smoke"].reference.model.byte_vocab == 256
    assert first["smoke"].resolved_config_sha256 == second["smoke"].resolved_config_sha256
    assert first["small"].status == "unresolved_numeric_proposal"
    manifest = {"manifest_sha256": "a" * 64, "protocol": TEXT_PROTOCOL_V0}
    candidate = candidate_manifest(manifest, first["smoke"], "b" * 64)
    assert candidate["candidate"]["accepted_codec_parent"] is None
    assert candidate["authoritative_training_allowed"] is False


def test_run_protocol_hash_changes_for_each_load_bearing_field():
    baseline = _protocol()
    baseline_hash = validate_run_protocol(baseline)
    mutations = (
        replace(baseline, max_updates=101),
        replace(baseline, training_seed=3),
        replace(baseline, batch_size=9),
        replace(baseline, resolved_config_sha256="e" * 64),
        replace(baseline, q=1),
        replace(baseline, runtime=replace(baseline.runtime, backend="gpu")),
    )
    assert all(validate_run_protocol(value) != baseline_hash for value in mutations)
    with pytest.raises(ValueError, match="lowercase SHA-256"):
        validate_run_protocol(replace(baseline, dataset_manifest_sha256="bad"))
    for invalid in (
        replace(baseline, training_seed=-1),
        replace(baseline, batch_size=True),
        replace(baseline, learning_rate=float("nan")),
        replace(baseline, q=True),
        replace(baseline, validation_every_updates=11),
        replace(baseline, parent_checkpoint_sha256="f" * 64),
    ):
        with pytest.raises(ValueError):
            validate_run_protocol(invalid)


def test_recipe_requires_exact_checkpoint_and_remains_non_authoritative():
    recipe = SmallScaleLMRecipe("smoke", "a" * 64, "b" * 64, 10, "eval-v0")
    validate_recipe(recipe)
    with pytest.raises(ValueError, match="authoritative language training"):
        validate_recipe(replace(recipe, authoritative=True))
    with pytest.raises(ValueError, match="must be non-empty"):
        validate_recipe(replace(recipe, evaluation_protocol=""))


def test_resume_state_binds_schema_protocol_step_params_and_both_adam_trees():
    encoder = {
        name: np.zeros((1,), dtype=np.float32)
        for name in (
            "byte_embed",
            "frontend",
            "target",
            "target_slot_embed",
            "target_carrier_embed",
            "target_pool",
            "target_h",
            "target_b",
            "target_l",
        )
    }
    params = {"encoder": encoder, "decoder": np.zeros((2,), dtype=np.float32)}
    moments = (
        {
            "encoder": {name: value.copy() for name, value in encoder.items()},
            "decoder": np.zeros(2, dtype=np.float32),
        },
        {
            "encoder": {name: value.copy() for name, value in encoder.items()},
            "decoder": np.zeros(2, dtype=np.float32),
        },
    )
    description = codec_state_tree_description(params, moments)
    tree_hash = checkpoint_tree_identity(description)
    protocol_hash = "a" * 64
    state = {
        "schema": "PHASE_I_CODEC_STATE_V0",
        "protocol_sha256": protocol_hash,
        "step": 10,
        "params": params,
        "moments": moments,
        "tree_description": description,
        "training_exposure_counts": (0,) * 256,
    }
    restored = validate_codec_resume_state(
        state,
        protocol_sha256=protocol_hash,
        expected_step=10,
        expected_tree_sha256=tree_hash,
    )
    assert restored[2] == (0,) * 256
    bad = {**state, "step": 9}
    with pytest.raises(RuntimeError, match="state identity"):
        validate_codec_resume_state(
            bad,
            protocol_sha256=protocol_hash,
            expected_step=10,
            expected_tree_sha256=tree_hash,
        )
    mismatched_moments = (moments[0], {**moments[1], "decoder": np.zeros(3, dtype=np.float32)})
    with pytest.raises(RuntimeError, match="leaf does not match"):
        validate_codec_resume_state(
            {**state, "moments": mismatched_moments},
            protocol_sha256=protocol_hash,
            expected_step=10,
            expected_tree_sha256=tree_hash,
        )


def test_checkpoint_evidence_transaction_reconciles_only_mutual_state(tmp_path):
    protocol_hash = validate_run_protocol(_protocol())
    checkpoint, evidence = save_transaction(
        tmp_path,
        step=10,
        protocol_sha256=protocol_hash,
        tree_identity_sha256="f" * 64,
        state={"params": [1], "moments": [2]},
        metrics={"loss": 1.0},
        training_exposure={"kind": "training_exposure"},
        evaluation_coverage={"kind": "evaluation_coverage"},
        reconstruction={"kind": "reconstruction"},
    )
    reconciled = reconcile_transactions(tmp_path, protocol_sha256=protocol_hash)
    assert reconciled.last_complete_step == 10
    assert reconciled.checkpoint_record == checkpoint
    assert reconciled.evidence_record == evidence
    with pytest.raises(RuntimeError, match="overwrite durable step"):
        save_transaction(
            tmp_path,
            step=10,
            protocol_sha256=protocol_hash,
            tree_identity_sha256="f" * 64,
            state={},
            metrics={},
            training_exposure={},
            evaluation_coverage={},
            reconstruction={},
        )
    with pytest.raises(RuntimeError, match="increase monotonically"):
        save_transaction(
            tmp_path,
            step=9,
            protocol_sha256=protocol_hash,
            tree_identity_sha256="f" * 64,
            state={},
            metrics={},
            training_exposure={},
            evaluation_coverage={},
            reconstruction={},
        )


def test_reconciliation_reports_checkpoint_ahead_and_rejects_stale_protocol(tmp_path):
    protocol_hash = validate_run_protocol(_protocol())
    (tmp_path / "checkpoints").mkdir()
    (tmp_path / "evidence").mkdir()
    (tmp_path / "checkpoints/step_00000002.json").write_text("{}")
    result = reconcile_transactions(tmp_path, protocol_sha256=protocol_hash)
    assert result.incomplete_steps == (2,)

    other = tmp_path / "other"
    save_transaction(
        other,
        step=1,
        protocol_sha256=protocol_hash,
        tree_identity_sha256="f" * 64,
        state={},
        metrics={},
        training_exposure={},
        evaluation_coverage={},
        reconstruction={},
    )
    with pytest.raises(ValueError, match="stale run protocol"):
        reconcile_transactions(other, protocol_sha256="0" * 64)


@pytest.mark.parametrize(
    "checkpoint_file",
    (
        "../outside.pkl",
        "/tmp/absolute.pkl",
        "checkpoints/alternate.pkl",
        "checkpoints/step_00000002.pkl",
    ),
)
def test_checkpoint_payload_path_must_be_exact_and_canonical(tmp_path, checkpoint_file):
    protocol_hash = validate_run_protocol(_protocol())
    save_transaction(
        tmp_path,
        step=1,
        protocol_sha256=protocol_hash,
        tree_identity_sha256="f" * 64,
        state={},
        metrics={},
        training_exposure={},
        evaluation_coverage={},
        reconstruction={},
    )
    path = tmp_path / "checkpoints/step_00000001.json"
    record = json.loads(path.read_text())
    record["checkpoint_file"] = checkpoint_file
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="noncanonical checkpoint path"):
        reconcile_transactions(tmp_path, protocol_sha256=protocol_hash)


def test_recovery_quarantines_noncanonical_checkpoint_path_without_following_it(tmp_path):
    protocol_hash = validate_run_protocol(_protocol())
    save_transaction(
        tmp_path,
        step=1,
        protocol_sha256=protocol_hash,
        tree_identity_sha256="f" * 64,
        state={},
        metrics={},
        training_exposure={},
        evaluation_coverage={},
        reconstruction={},
    )
    path = tmp_path / "checkpoints/step_00000001.json"
    record = json.loads(path.read_text())
    record["checkpoint_file"] = "../../outside.pkl"
    path.write_text(json.dumps(record))
    recovery = recover_transactions(tmp_path, protocol_sha256=protocol_hash)
    assert recovery.reconciliation.last_complete_step == 0
    assert set(recovery.moved_artifacts) >= {
        "checkpoints/step_00000001.pkl",
        "checkpoints/step_00000001.json",
        "evidence/step_00000001.json",
        "latest.json",
    }


def test_latest_pointer_is_validated_and_explicit_recovery_rebuilds_it(tmp_path):
    protocol_hash = validate_run_protocol(_protocol())
    save_transaction(
        tmp_path,
        step=1,
        protocol_sha256=protocol_hash,
        tree_identity_sha256="f" * 64,
        state={},
        metrics={},
        training_exposure={},
        evaluation_coverage={},
        reconstruction={},
    )
    latest_path = tmp_path / "latest.json"
    latest = json.loads(latest_path.read_text())
    latest["transaction_sha256"] = "0" * 64
    latest_path.write_text(json.dumps(latest))
    with pytest.raises(ValueError, match="latest pointer does not match"):
        reconcile_transactions(tmp_path, protocol_sha256=protocol_hash)
    recovery = recover_transactions(tmp_path, protocol_sha256=protocol_hash)
    assert "latest.json" in recovery.moved_artifacts
    assert reconcile_transactions(tmp_path, protocol_sha256=protocol_hash).last_complete_step == 1


def test_missing_latest_pointer_fails_normal_resume_and_recovery_rebuilds_it(tmp_path):
    protocol_hash = validate_run_protocol(_protocol())
    save_transaction(
        tmp_path,
        step=1,
        protocol_sha256=protocol_hash,
        tree_identity_sha256="f" * 64,
        state={},
        metrics={},
        training_exposure={},
        evaluation_coverage={},
        reconstruction={},
    )
    (tmp_path / "latest.json").unlink()
    with pytest.raises(ValueError, match="latest pointer is missing"):
        reconcile_transactions(tmp_path, protocol_sha256=protocol_hash)
    recovery = recover_transactions(tmp_path, protocol_sha256=protocol_hash)
    assert recovery.reconciliation.last_complete_step == 1
    assert reconcile_transactions(tmp_path, protocol_sha256=protocol_hash).last_complete_step == 1


def test_malformed_payload_name_fails_normal_resume_and_is_quarantined(tmp_path):
    protocol_hash = validate_run_protocol(_protocol())
    (tmp_path / "checkpoints").mkdir()
    malformed = tmp_path / "checkpoints/step_00000001_duplicate.pkl"
    malformed.write_bytes(b"duplicate")
    with pytest.raises(ValueError, match="malformed checkpoint payload name"):
        reconcile_transactions(tmp_path, protocol_sha256=protocol_hash)
    recovery = recover_transactions(tmp_path, protocol_sha256=protocol_hash)
    assert recovery.moved_artifacts == ("checkpoints/step_00000001_duplicate.pkl",)


@pytest.mark.parametrize(
    ("field", "message"),
    [
        ("metrics", "evidence metrics hash mismatch"),
        ("training_exposure", "training-exposure evidence hash mismatch"),
        ("evaluation_coverage", "evaluation-coverage evidence hash mismatch"),
        ("reconstruction", "reconstruction evidence hash mismatch"),
    ],
)
def test_each_byte_evidence_body_is_bound_into_transaction_hash(tmp_path, field, message):
    protocol_hash = validate_run_protocol(_protocol())
    save_transaction(
        tmp_path,
        step=1,
        protocol_sha256=protocol_hash,
        tree_identity_sha256="f" * 64,
        state={},
        metrics={"loss": 1.0},
        training_exposure={"counts": [1]},
        evaluation_coverage={"counts": [2]},
        reconstruction={"correct": [1]},
    )
    path = tmp_path / "evidence/step_00000001.json"
    record = json.loads(path.read_text())
    record[field]["tampered"] = True
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match=message):
        reconcile_transactions(tmp_path, protocol_sha256=protocol_hash)


def test_explicit_recovery_quarantines_all_unmatched_classes_and_is_idempotent(tmp_path):
    protocol_hash = validate_run_protocol(_protocol())
    save_transaction(
        tmp_path,
        step=1,
        protocol_sha256=protocol_hash,
        tree_identity_sha256="f" * 64,
        state={"durable": True},
        metrics={},
        training_exposure={},
        evaluation_coverage={},
        reconstruction={},
    )
    # Evidence-ahead, checkpoint-ahead, payload-only, temp, and malformed duplicate.
    (tmp_path / "evidence/step_00000002.json").write_text("{}")
    (tmp_path / "checkpoints/step_00000003.pkl").write_bytes(b"checkpoint-ahead")
    (tmp_path / "checkpoints/step_00000003.json").write_text("{}")
    (tmp_path / "checkpoints/step_00000004.pkl").write_bytes(b"payload-only")
    (tmp_path / "checkpoints/step_00000005.pkl.tmp").write_bytes(b"temporary")
    (tmp_path / "evidence/step_00000006_duplicate.json").write_text("{}")
    # A fully paired but stale-protocol transaction must also be quarantined.
    stale = "0" * 64
    stale_root = tmp_path / "stale-staging"
    save_transaction(
        stale_root,
        step=7,
        protocol_sha256=stale,
        tree_identity_sha256="f" * 64,
        state={"stale": True},
        metrics={},
        training_exposure={},
        evaluation_coverage={},
        reconstruction={},
    )
    shutil.copy2(
        stale_root / "checkpoints/step_00000007.pkl",
        tmp_path / "checkpoints/step_00000007.pkl",
    )
    shutil.copy2(
        stale_root / "checkpoints/step_00000007.json",
        tmp_path / "checkpoints/step_00000007.json",
    )
    shutil.copy2(
        stale_root / "evidence/step_00000007.json",
        tmp_path / "evidence/step_00000007.json",
    )

    with pytest.raises(ValueError, match="explicit recovery|malformed evidence record"):
        reconcile_transactions(tmp_path, protocol_sha256=protocol_hash)
    recovered = recover_transactions(tmp_path, protocol_sha256=protocol_hash)
    assert recovered.reconciliation.last_complete_step == 1
    assert recovered.quarantine_directory == "quarantine/recovery_000001"
    assert any("step_00000002" in name for name in recovered.moved_artifacts)
    assert any("step_00000003" in name for name in recovered.moved_artifacts)
    assert any("step_00000004" in name for name in recovered.moved_artifacts)
    assert any("step_00000005" in name for name in recovered.moved_artifacts)
    assert any("duplicate" in name for name in recovered.moved_artifacts)
    assert any("step_00000007" in name for name in recovered.moved_artifacts)
    assert (tmp_path / "quarantine/recovery_000001/manifest.json").exists()
    latest = json.loads((tmp_path / "latest.json").read_text())
    assert latest["step"] == 1

    repeated = recover_transactions(tmp_path, protocol_sha256=protocol_hash)
    assert repeated.reconciliation.last_complete_step == 1
    assert repeated.quarantine_directory is None
    assert repeated.moved_artifacts == ()


def test_mutated_checkpoint_payload_is_rejected_and_quarantined(tmp_path):
    protocol_hash = validate_run_protocol(_protocol())
    save_transaction(
        tmp_path,
        step=1,
        protocol_sha256=protocol_hash,
        tree_identity_sha256="f" * 64,
        state={"original": True},
        metrics={},
        training_exposure={},
        evaluation_coverage={},
        reconstruction={},
    )
    (tmp_path / "checkpoints/step_00000001.pkl").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="checkpoint payload mismatch"):
        reconcile_transactions(tmp_path, protocol_sha256=protocol_hash)
    recovered = recover_transactions(tmp_path, protocol_sha256=protocol_hash)
    assert recovered.reconciliation.last_complete_step == 0
    assert set(recovered.moved_artifacts) >= {
        "checkpoints/step_00000001.pkl",
        "checkpoints/step_00000001.json",
        "evidence/step_00000001.json",
        "latest.json",
    }


def test_task_separation_and_answer_independent_budget():
    cases = (
        conditional_case("conditional", b"abcdef", 3, byte_budget=9),
        infilling_case("infill", "a£z".encode(), 1, 3, byte_budget=7),
        unconditional_case("free", byte_budget=5),
    )
    calls: list[GenerationRequest] = []

    class Generator:
        def generate(self, request: GenerationRequest) -> bytes:
            calls.append(request)
            return {
                EvaluationTask.CONDITIONAL_GENERATION: b"def",
                EvaluationTask.INFILLING: "£".encode(),
                EvaluationTask.UNCONDITIONAL_GENERATION: b"hello",
            }[request.task]

    outputs, metrics = run_generation_harness(Generator(), cases, root=41)
    tasks = cast(dict[str, dict[str, Any]], metrics["tasks"])
    assert outputs == (b"def", "£".encode(), b"hello")
    assert [call.byte_budget for call in calls] == [9, 7, 5]
    assert len({call.root for call in calls}) == 3
    assert tasks["conditional_generation"]["exact_accuracy"] == 1.0
    assert "exact_accuracy" not in tasks["unconditional_generation"]
    assert sum(tasks["unconditional_generation"]["generated_byte_counts"]) == 5
    assert tasks["unconditional_generation"]["generated_byte_coverage"] == 4


def test_case_roots_are_stable_under_reordering_and_budget_overrun_fails():
    first = unconditional_case("a", byte_budget=2)
    second = unconditional_case("b", byte_budget=2)
    roots: dict[str, set[int]] = {}

    class Generator:
        def generate(self, request: GenerationRequest) -> bytes:
            roots.setdefault(request.case_id, set()).add(request.root)
            return b"x"

    run_generation_harness(Generator(), (first, second), root=5)
    run_generation_harness(Generator(), (second, first), root=5)
    assert all(len(values) == 1 for values in roots.values())

    class BadGenerator:
        def generate(self, request: GenerationRequest) -> bytes:
            return b"too long"

    with pytest.raises(ValueError, match="exceeded"):
        run_generation_harness(BadGenerator(), (first,), root=5)
    with pytest.raises(ValueError, match="case IDs must be unique"):
        run_generation_harness(Generator(), (first, first), root=5)

    class NonBytesGenerator:
        def generate(self, request: GenerationRequest) -> bytes:
            return cast(bytes, "not-bytes")

    with pytest.raises(TypeError, match="outputs must be bytes"):
        run_generation_harness(NonBytesGenerator(), (first,), root=5)


def test_metrics_are_task_stratified():
    case = conditional_case("one", b"ab", 1, byte_budget=3)
    metrics = evaluate_outputs((case,), (b"b",), ComputeObservation(1.0))
    tasks = cast(dict[str, dict[str, Any]], metrics["tasks"])
    assert set(tasks) == {"conditional_generation"}
    assert tasks["conditional_generation"]["aligned_byte_accuracy"] == 1.0


def test_all_five_generation_tasks_remain_separate():
    cases = (
        reconstruction_case("teacher", b"a", denoised=False, byte_budget=2),
        reconstruction_case("denoised", b"b", denoised=True, byte_budget=2),
        conditional_case("conditional", b"cd", 1, byte_budget=2),
        unconditional_case("unconditional", byte_budget=2),
        infilling_case("infilling", b"efg", 1, 2, byte_budget=2),
    )

    class Generator:
        def generate(self, request: GenerationRequest) -> bytes:
            return {
                "teacher": b"a",
                "denoised": b"b",
                "conditional": b"d",
                "unconditional": b"z",
                "infilling": b"f",
            }[request.case_id]

    _, metrics = run_generation_harness(Generator(), cases, root=2)
    tasks = cast(dict[str, dict[str, Any]], metrics["tasks"])
    assert set(tasks) == {task.value for task in EvaluationTask}
    assert "reference_bytes" not in tasks["unconditional_generation"]
