"""Pure-file pause-audit oracles, NOT neural optimizer or resume execution tests."""

import hashlib
import json

import pytest
from audit_phase_g23_pause import journal
from collect_phase_g23_records import sha


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def fixture(tmp_path):
    previous = None
    r = {}
    for step in range(3):
        slot = tmp_path / f"slot_{step % 2}.pkl"
        slot.write_bytes(f"synthetic-payload-{step}".encode())
        metrics = (
            {"fresh_Adam": True, "initial_tree": "means"}
            if step == 0
            else {"loss": 1.25, "frozen_state_checked_unchanged": True}
        )
        r = {
            "step": step,
            "slot": slot.name,
            "identity": "protocol:seed1:B",
            "completed": True,
            "parent": previous,
            "state_sha256": sha(slot),
            "snapshot": step == 0,
            "metrics": metrics,
        }
        if step == 0:
            p = tmp_path / "snapshots/update_00000.pkl"
            p.parent.mkdir(parents=True)
            p.write_bytes(slot.read_bytes())
        r["receipt_digest"] = hashlib.sha256(json.dumps(r, sort_keys=True).encode()).hexdigest()
        put(tmp_path / f"receipts/{step:05d}.json", r)
        previous = r["state_sha256"]
    put(tmp_path / "LATEST.json", r)


def test_nonmilestone_tip_and_both_slots_are_verified(tmp_path):
    fixture(tmp_path)
    result, rows = journal(tmp_path, "protocol:seed1:B", "means")
    assert result["completed_updates"] == 2 and result["next_update"] == 3
    assert len(rows) == 3 and result["payload_sha256"] == sha(tmp_path / "slot_0.pkl")
    assert result["tensor_reexecution_performed"] is False


@pytest.mark.parametrize("mutation", ["tip", "previous_slot", "pointer", "receipt", "gap"])
def test_pause_audit_rejects_corruption(tmp_path, mutation):
    fixture(tmp_path)
    if mutation in ("tip", "previous_slot"):
        (tmp_path / ("slot_0.pkl" if mutation == "tip" else "slot_1.pkl")).write_bytes(b"bad")
    elif mutation == "pointer":
        put(tmp_path / "LATEST.json", {})
    elif mutation == "receipt":
        put(tmp_path / "receipts/00001.json", {})
    else:
        (tmp_path / "receipts/00001.json").rename(tmp_path / "missing.json")
    with pytest.raises((ValueError, KeyError)):
        journal(tmp_path, "protocol:seed1:B", "means")
