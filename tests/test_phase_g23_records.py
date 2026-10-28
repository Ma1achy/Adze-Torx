"""Pure-I/O evidence-export tests; no model or stochastic execution."""

import hashlib
import json

import pytest
import collect_phase_g23_records as export


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def fixture(tmp_path, monkeypatch):
    monkeypatch.setattr(export, "ROOT", tmp_path)
    binding = {"source": {}, "protocols": {}, "initial_parameter_tree": "initial-means"}
    identity = hashlib.sha256(json.dumps(binding, sort_keys=True).encode()).hexdigest()
    put(
        tmp_path / "results/phase_g/g2/v1/G2_G3_PROTOCOL_IDENTITY.json",
        {"binding": binding, "identity": identity},
    )
    run = tmp_path / "results/runs/phase_g/g23/v1/seed0/A"
    parent = None
    for step in (0, 1):
        blob = run / "snapshots" / f"update_{step:05d}.pkl"
        blob.parent.mkdir(parents=True, exist_ok=True)
        blob.write_bytes(f"synthetic-state-{step}".encode())
        metrics = (
            {"initial_tree": "initial-means", "fresh_Adam": True}
            if step == 0
            else {"loss": 1.25, "frozen_state_checked_unchanged": True}
        )
        record = {
            "step": step,
            "identity": identity + ":seed0:A",
            "completed": True,
            "parent": parent,
            "state_sha256": export.sha(blob),
            "snapshot": True,
            "metrics": metrics,
        }
        record["receipt_digest"] = hashlib.sha256(
            json.dumps(record, sort_keys=True).encode()
        ).hexdigest()
        put(run / "receipts" / f"{step:05d}.json", record)
        parent = record["state_sha256"]
    return run


def test_export_is_lossless_immutable_and_initial_record_not_counted(tmp_path, monkeypatch):
    fixture(tmp_path, monkeypatch)
    meta = export.collect(0, "A", 1)
    data = (tmp_path / meta["path"]).read_text()
    rows = [json.loads(x) for x in data.splitlines()]
    assert len(rows) == 2 and meta["counted_updates"] == 1
    assert not rows[0]["counted_update"] and rows[1]["counted_update"]
    assert rows[1]["metrics"]["loss"] == 1.25
    assert export.collect(0, "A", 1) == meta
    (tmp_path / meta["path"]).write_text("tampered")
    with pytest.raises(ValueError, match="existing export differs"):
        export.collect(0, "A", 1)


@pytest.mark.parametrize("mutation", ["checksum", "frozen", "snapshot", "identity"])
def test_bad_evidence_cannot_produce_an_export(tmp_path, monkeypatch, mutation):
    run = fixture(tmp_path, monkeypatch)
    p = run / "receipts/00001.json"
    r = json.loads(p.read_text())
    if mutation == "snapshot":
        (run / "snapshots/update_00001.pkl").write_bytes(b"bad")
    else:
        r["metrics"]["loss"] = 9.0
        if mutation == "frozen":
            r["metrics"]["frozen_state_checked_unchanged"] = False
        elif mutation == "identity":
            r["identity"] = "another-protocol"
        if mutation != "checksum":
            r["receipt_digest"] = hashlib.sha256(
                json.dumps(
                    {k: v for k, v in r.items() if k != "receipt_digest"}, sort_keys=True
                ).encode()
            ).hexdigest()
        put(p, r)
    with pytest.raises(ValueError):
        export.collect(0, "A", 1)
    assert not list((tmp_path / "results/phase_g/g2/v1/seed0/A").glob("*.jsonl"))
