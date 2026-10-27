"""Read-only journal verification/export after a resource stop; no model imports."""

import hashlib
import json
import math
from pathlib import Path

from collect_phase_g23_records import ROOT, sha, write_identical


def finite(value):
    if isinstance(value, dict):
        return all(finite(x) for x in value.values())
    if isinstance(value, list):
        return all(finite(x) for x in value)
    return not isinstance(value, float) or math.isfinite(value)


def journal(directory, identity, initial_tree):
    paths = sorted((directory / "receipts").glob("*.json"))
    if not paths:
        raise ValueError("no completed receipts")
    previous, rows = None, []
    for step, path in enumerate(paths):
        record = json.loads(path.read_text())
        body = {k: v for k, v in record.items() if k != "receipt_digest"}
        checksum = hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()
        if (
            path.name != f"{step:05d}.json"
            or record["step"] != step
            or record["identity"] != identity
            or record["parent"] != previous
            or record["completed"] is not True
            or record["receipt_digest"] != checksum
            or record["slot"] != f"slot_{step % 2}.pkl"
            or not finite(record["metrics"])
        ):
            raise ValueError("invalid receipt/chain/metrics")
        metrics = record["metrics"]
        if step == 0:
            if not metrics.get("fresh_Adam") or metrics.get("initial_tree") != initial_tree:
                raise ValueError("not matched fresh initialization")
        elif metrics.get("frozen_state_checked_unchanged") is not True:
            raise ValueError("missing frozen-state proof")
        if record["snapshot"]:
            if sha(directory / "snapshots" / f"update_{step:05d}.pkl") != record["state_sha256"]:
                raise ValueError("snapshot checksum mismatch")
        rows.append({"receipt_file_sha256": sha(path), "receipt": record})
        previous = record["state_sha256"]
    if json.loads((directory / "LATEST.json").read_text()) != rows[-1]["receipt"]:
        raise ValueError("LATEST differs from contiguous verified tip")
    # Verify BOTH alternating slots, not merely the derived LATEST pointer.
    for row in rows[-2:]:
        r = row["receipt"]
        if sha(directory / r["slot"]) != r["state_sha256"]:
            raise ValueError("durable ring payload checksum mismatch")
    tip = rows[-1]["receipt"]
    return {
        "completed_updates": tip["step"],
        "next_update": tip["step"] + 1,
        "payload": str(directory / tip["slot"]),
        "payload_sha256": tip["state_sha256"],
        "receipt_count": len(rows),
        "finite_recorded_metrics": True,
        "frozen_state_receipts_verified": True,
        "tensor_reexecution_performed": False,
    }, rows


def main():
    out = ROOT / "results/phase_g/g2/v1"
    run = ROOT / "results/runs/phase_g/g23/v1"
    protocol = json.loads((out / "G2_G3_PROTOCOL_IDENTITY.json").read_text())
    binding, identity = protocol["binding"], protocol["identity"]
    if hashlib.sha256(json.dumps(binding, sort_keys=True).encode()).hexdigest() != identity:
        raise ValueError("protocol identity checksum mismatch")
    for name, digest in {**binding["source"], **binding["protocols"]}.items():
        if sha(ROOT / name) != digest:
            raise ValueError("bound source/protocol changed:" + name)
    failure_path = run / "failure_000.json"
    failure = json.loads(failure_path.read_text())
    if (
        failure["exception"]
        != "RuntimeError('critical disk capacity; exact durable state preserved')"
    ):
        raise ValueError("not the expected resource failure")
    destination = out / "resource_stop_20261027T170605"
    results = {}
    for directory in sorted(run.glob("seed[012]/[ABC]")):
        key = directory.relative_to(run).as_posix()
        seed, arm = key.split("/")
        audit, rows = journal(
            directory, identity + f":{seed}:{arm}", binding["initial_parameter_tree"]
        )
        results[key] = audit
        # Lossless copy of each receipt, not an invented successful checkpoint.
        data = "\n".join(json.dumps(r, sort_keys=True, allow_nan=False) for r in rows) + "\n"
        write_identical(destination / (key.replace("/", "_") + "_receipts.jsonl"), data.encode())
    record = {
        "identity": identity,
        "failure": failure,
        "failure_file_sha256": sha(failure_path),
        "journals": results,
        "total_counted_updates": sum(r["completed_updates"] for r in results.values()),
        "auditor_source_sha256": sha(Path(__file__).resolve()),
        "no_tensor_execution_or_optimizer_updates": True,
        "scientific_closure": False,
        "protected_selection_exists": (ROOT / "results/phase_g/g3/v1/SELECTION.json").exists(),
    }
    write_identical(
        destination / "PAUSE_RECORD.json", (json.dumps(record, indent=2) + "\n").encode()
    )
    write_identical(
        destination / "campaign_stdout.txt", Path("/private/tmp/g23_v1_campaign.txt").read_bytes()
    )
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
