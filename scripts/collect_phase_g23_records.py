"""Lossless receipt-to-JSONL evidence export; no model import, RNG or decisions."""

import argparse
import hashlib
import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_identical(path, data):
    """Only create immutable exports; retain partial files rather than deleting."""
    if path.exists():
        if path.read_bytes() != data:
            raise ValueError("existing export differs; preserve and investigate")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    if tmp.exists():
        n = len(list(path.parent.glob(path.name + ".orphan_*")))
        tmp.rename(path.with_name(path.name + f".orphan_{n:03d}"))
    with tmp.open("xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    tmp.replace(path)


def collect(seed, arm, through):
    identity_file = ROOT / "results/phase_g/g2/v1/G2_G3_PROTOCOL_IDENTITY.json"
    protocol = json.loads(identity_file.read_text())
    binding = protocol["binding"]
    identity = protocol["identity"]
    if hashlib.sha256(json.dumps(binding, sort_keys=True).encode()).hexdigest() != identity:
        raise ValueError("protocol binding checksum mismatch")
    for name, digest in {**binding["source"], **binding["protocols"]}.items():
        if sha(ROOT / name) != digest:
            raise ValueError("model source/protocol differs:" + name)
    run = ROOT / f"results/runs/phase_g/g23/v1/seed{seed}/{arm}"
    out = ROOT / f"results/phase_g/g2/v1/seed{seed}/{arm}"
    rows, previous, snapshots = [], None, []
    for step in range(through + 1):
        path = run / "receipts" / f"{step:05d}.json"
        record = json.loads(path.read_text())
        content = {k: v for k, v in record.items() if k != "receipt_digest"}
        digest = hashlib.sha256(json.dumps(content, sort_keys=True).encode()).hexdigest()
        if (
            digest != record["receipt_digest"]
            or record["identity"] != identity + f":seed{seed}:{arm}"
            or record["step"] != step
            or record["parent"] != previous
            or record["completed"] is not True
        ):
            raise ValueError("noncontiguous/mixed/invalid completed receipt")
        if step == 0:
            if (
                record["metrics"].get("fresh_Adam") is not True
                or record["metrics"].get("initial_tree") != binding["initial_parameter_tree"]
            ):
                raise ValueError("initialization mismatch")
        elif record["metrics"].get("frozen_state_checked_unchanged") is not True:
            raise ValueError("missing per-update frozen-state proof")
        if record["snapshot"]:
            snapshot = run / "snapshots" / f"update_{step:05d}.pkl"
            if sha(snapshot) != record["state_sha256"]:
                raise ValueError("checkpoint/receipt mismatch")
            snapshots.append(
                {"step": step, "path": str(snapshot.relative_to(ROOT)), "sha256": sha(snapshot)}
            )
        rows.append(
            {
                "step": step,
                "counted_update": step > 0,
                "receipt_sha256": sha(path),
                "state_sha256": record["state_sha256"],
                "metrics": record["metrics"],
            }
        )
        previous = record["state_sha256"]
    if not any(x["step"] == through for x in snapshots):
        raise ValueError("export endpoint is not a durable checkpoint")
    data = ("\n".join(json.dumps(r, sort_keys=True, allow_nan=False) for r in rows) + "\n").encode()
    output = out / f"training_through_{through:05d}.jsonl"
    write_identical(output, data)
    meta = {
        "identity": identity,
        "seed": seed,
        "arm": arm,
        "through": through,
        "counted_updates": through,
        "initial_record_counted": False,
        "path": str(output.relative_to(ROOT)),
        "sha256": sha(output),
        "rows": len(rows),
        "snapshots": snapshots,
        "exporter_source_sha256": sha(Path(__file__).resolve()),
        "lossless_metrics": True,
        "scientific_selection_or_decision": False,
    }
    write_identical(
        output.with_suffix(".json"), (json.dumps(meta, indent=2, allow_nan=False) + "\n").encode()
    )
    return meta


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, choices=(0, 1, 2), required=True)
    parser.add_argument("--arm", choices=("A", "B", "C"), required=True)
    parser.add_argument("--through", type=int, choices=(1000, 2000), required=True)
    args = parser.parse_args()
    print(json.dumps(collect(args.seed, args.arm, args.through), indent=2))
