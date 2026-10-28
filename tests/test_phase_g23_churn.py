"""Exact derived-statistic tests, not model/trainability evidence."""

import hashlib
import json

import numpy as np
import pytest

import summarize_phase_g23_churn as churn
from summarize_phase_g23_churn import paired_churn


def rows(boundary):
    b = np.asarray([[boundary, 0, 1]])
    return {
        "global_example_id": np.asarray([10000]),
        "b": b,
        "ell": np.asarray([[4, 4, 0]]),
        "capacity_cut": np.zeros_like(b),
        "pack_m": np.zeros_like(b),
        "pack_k": np.asarray([[0, 1, 2]]),
        "prediction_b": b.copy(),
        "prediction_ell": np.asarray([[4, 4, 0]]),
    }


def test_rootwise_change_is_not_change_of_averaged_actions():
    zero, one = rows(0), rows(1)
    assert np.array_equal((zero["b"] + one["b"]) / 2, (one["b"] + zero["b"]) / 2)
    result = [paired_churn(zero, one), paired_churn(one, zero)]
    assert np.mean([r["complete_structure_change"] for r in result]) == 1
    assert result[0]["boundary_site_change_terminal_excluded"] == 0.5


def test_teacher_commit_preserved_despite_prediction_change():
    a, b = rows(1), rows(1)
    b["prediction_b"][0, 0] = 0
    result = paired_churn(a, b)
    assert result["complete_structure_change"] == 0
    assert result["predicted_boundary_site_change"] == 0.5
    b["ell"][0, 2] = 1
    assert paired_churn(a, b)["reactivation_site_rate"] == pytest.approx(1 / 3)


def test_wrong_identity_and_invalid_actions_are_rejected():
    a, b = rows(1), rows(1)
    b["global_example_id"][0] = 10001
    with pytest.raises(ValueError, match="ordered"):
        paired_churn(a, b)
    b = rows(1)
    b["b"][0, -1] = 0
    with pytest.raises(ValueError, match="invalid"):
        paired_churn(a, b)


def test_empty_support_is_not_reported_as_meaningful_zero():
    empty = {k: v[:0] for k, v in rows(1).items()}
    with pytest.raises(ValueError, match="invalid"):
        paired_churn(empty, empty)


@pytest.mark.parametrize("mutation", ["none", "checkpoint", "example"])
def test_saved_shard_checkpoint_and_actual_ids_are_verified(tmp_path, monkeypatch, mutation):
    monkeypatch.setattr(churn, "ROOT", tmp_path)
    directory = tmp_path / "results/runs/phase_g/g23/v1/seed0/A"
    checkpoint = directory / "snapshots/update_00000.pkl"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"synthetic-complete-state")
    data = rows(1)
    path = directory / "evaluation/validation_00000_teacher_00_policy.npz"
    path.parent.mkdir()
    np.savez(path, **data)
    provenance = {
        "identity": "protocol",
        "seed": 0,
        "arm": "A",
        "step": 0,
        "split": "validation",
        "mode": "teacher",
        "root": 0,
        "control": "policy",
        "model_tree": "means",
        "checkpoint_sha256": churn.sha(checkpoint),
        "ids_sha256": hashlib.sha256(data["global_example_id"].tobytes()).hexdigest(),
    }
    receipt = directory / "receipts/00000.json"
    receipt.parent.mkdir()
    receipt.write_text(json.dumps({"state_sha256": churn.sha(checkpoint)}))
    point = tmp_path / "results/phase_g/g2/v1/seed0/A/validation_00000.json"
    point.parent.mkdir(parents=True)
    point.write_text(json.dumps({"model_tree": "means", "identity": "protocol"}))
    if mutation == "checkpoint":
        checkpoint.write_bytes(b"wrong")
    elif mutation == "example":
        data["global_example_id"][0] = 99
        np.savez(path, **data)
    path.with_suffix(".json").write_text(
        json.dumps({"provenance": provenance, "sha256": churn.sha(path)})
    )
    if mutation == "none":
        actual, _ = churn.load(0, "A", 0, "teacher", 0, "protocol")
        assert np.array_equal(actual["b"], data["b"])
    else:
        with pytest.raises(ValueError, match="identity mismatch"):
            churn.load(0, "A", 0, "teacher", 0, "protocol")
