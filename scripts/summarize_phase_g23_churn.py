"""Derived fixed-example temporal churn; no model import/execution or decisions."""

import hashlib
import json
from pathlib import Path

import numpy as np

from collect_phase_g23_records import ROOT, sha, write_identical


def paired_churn(before, after):
    if not np.array_equal(before["global_example_id"], after["global_example_id"]):
        raise ValueError("temporal examples are not identically ordered")
    for rows in (before, after):
        b, ell = rows["b"], rows["ell"]
        if (
            b.shape != ell.shape
            or b.ndim != 2
            or b.shape[0] == 0
            or b.shape[0] != len(rows["global_example_id"])
            or b.shape[1] < 2
            or not np.isin(b, (0, 1)).all()
            or not (b[:, -1] == 1).all()
            or not np.isin(ell, (0, 1, 2, 3, 4)).all()
        ):
            raise ValueError("invalid physical committed actions")
    if before["b"].shape != after["b"].shape:
        raise ValueError("carrier identity/shape changed")
    b_changed = before["b"] != after["b"]
    ell_changed = before["ell"] != after["ell"]
    return {
        "support_examples": len(before["b"]),
        "boundary_site_change_terminal_excluded": float(b_changed[:, :-1].mean()),
        "extent_site_change": float(ell_changed.mean()),
        "complete_structure_change": float((b_changed.any(1) | ell_changed.any(1)).mean()),
        "total_emitted_length_change": float((before["ell"].sum(1) != after["ell"].sum(1)).mean()),
        "reactivation_site_rate": float(((before["ell"] == 0) & (after["ell"] > 0)).mean()),
        "deactivation_site_rate": float(((before["ell"] > 0) & (after["ell"] == 0)).mean()),
        "capacity_cut_site_change": float((before["capacity_cut"] != after["capacity_cut"]).mean()),
        "packing_coordinate_change": float(
            ((before["pack_m"] != after["pack_m"]) | (before["pack_k"] != after["pack_k"])).mean()
        ),
        "predicted_boundary_site_change": float(
            (before["prediction_b"][:, :-1] != after["prediction_b"][:, :-1]).mean()
        ),
        "predicted_extent_site_change": float(
            (before["prediction_ell"] != after["prediction_ell"]).mean()
        ),
    }


def load(seed, arm, step, mode, root, identity):
    directory = ROOT / f"results/runs/phase_g/g23/v1/seed{seed}/{arm}/evaluation"
    path = directory / f"validation_{step:05d}_{mode}_{root:02d}_policy.npz"
    meta = json.loads(path.with_suffix(".json").read_text())
    provenance = meta["provenance"]
    expected = {
        "identity": identity,
        "seed": seed,
        "arm": arm,
        "step": step,
        "split": "validation",
        "mode": mode,
        "root": root,
        "control": "policy",
    }
    if any(provenance[k] != v for k, v in expected.items()) or sha(path) != meta["sha256"]:
        raise ValueError("mixed or corrupt temporal shard")
    point = json.loads(
        (ROOT / f"results/phase_g/g2/v1/seed{seed}/{arm}/validation_{step:05d}.json").read_text()
    )
    if provenance["model_tree"] != point["model_tree"] or point["identity"] != identity:
        raise ValueError("validation model identity mismatch")
    checkpoint = (
        ROOT / f"results/runs/phase_g/g23/v1/seed{seed}/{arm}/snapshots/update_{step:05d}.pkl"
    )
    receipt = json.loads((checkpoint.parent.parent / "receipts" / f"{step:05d}.json").read_text())
    if sha(checkpoint) != provenance["checkpoint_sha256"] or receipt["state_sha256"] != sha(
        checkpoint
    ):
        raise ValueError("temporal checkpoint identity mismatch")
    with np.load(path) as archive:
        rows = {k: archive[k] for k in archive.files}
    if hashlib.sha256(rows["global_example_id"].tobytes()).hexdigest() != provenance["ids_sha256"]:
        raise ValueError("temporal example identity mismatch")
    return rows, provenance


def main():
    selected = ROOT / "results/phase_g/g3/v1/SELECTION.json"
    record = json.loads(selected.read_text())
    identity = record["identity"]
    protocol = ROOT / "results/phase_g/g2/v1/G2_G3_PROTOCOL_IDENTITY.json"
    saved = json.loads(protocol.read_text())
    binding = saved["binding"]
    if (
        record["cap"] != 1000
        or saved["identity"] != identity
        or hashlib.sha256(json.dumps(binding, sort_keys=True).encode()).hexdigest() != identity
        or binding["runtime"]["packages"]["numpy"] != np.__version__
    ):
        raise ValueError("wrong campaign or cap")
    for name, digest in {**binding["source"], **binding["protocols"]}.items():
        if sha(ROOT / name) != digest:
            raise ValueError("bound source/protocol changed:" + name)
    expected = {f"seed{s}:{a}" for s in (0, 1, 2) for a in ("A", "B", "C")}
    if set(record["selected"]) != expected or any(
        r["selected"] is None for r in record["selected"].values()
    ):
        raise ValueError("no complete frozen selection")
    results = {}
    for seed in (0, 1, 2):
        for arm in ("A", "B", "C"):
            modes = {"A": ("teacher",), "B": ("argmax",), "C": ("argmax", "sample")}[arm]
            for mode in modes:
                transitions = []
                for old, new in ((0, 100), (100, 250), (250, 500), (500, 1000)):
                    roots = []
                    for root in range(4 if mode == "sample" else 1):
                        a, pa = load(seed, arm, old, mode, root, identity)
                        b, pb = load(seed, arm, new, mode, root, identity)
                        if (
                            pa["dataset_sha256"]
                            != binding["datasets"]["ordered_sha256"]["validation"]
                        ):
                            raise ValueError("temporal validation split differs")
                        for name in (
                            "structure_root_key",
                            "operator_root_key",
                            "ids_sha256",
                            "dataset_sha256",
                        ):
                            if pa[name] != pb[name]:
                                raise ValueError("temporal key/example coupling differs")
                        roots.append(paired_churn(a, b))
                    transitions.append(
                        {
                            "from": old,
                            "to": new,
                            "per_root": roots,
                            "mean_of_rootwise_churn": {
                                k: float(np.mean([r[k] for r in roots])) for k in roots[0]
                            },
                        }
                    )
                results[f"seed{seed}:{arm}:{mode}"] = transitions
    value = {
        "identity": identity,
        "selection_sha256": sha(selected),
        "source_sha256": sha(Path(__file__)),
        "optimizer_updates": 0,
        "not_neutral_to_commit_change": True,
        "no_action_averaging_before_churn": True,
        "results": results,
    }
    target = ROOT / "results/phase_g/g3/v1/TEMPORAL_CHURN.json"
    write_identical(target, (json.dumps(value, indent=2, allow_nan=False) + "\n").encode())
    print(target)


if __name__ == "__main__":
    main()
