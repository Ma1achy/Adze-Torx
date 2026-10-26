"""Per-site/class supplemental read-only audit of saved frozen probes."""

from __future__ import annotations

import json
import pickle

import jax.numpy as jnp
import numpy as np

from run_phase_g0 import data, digest, load, write
from run_phase_g01 import OUT, PARENT, ROOT, RUN, features, manifest
from adze_t.phase_g01 import extent_report, probe_apply


def per_site(predicted, truth):
    rows = []
    for site in range(32):
        matrix = np.zeros((5, 5), np.int64)
        np.add.at(matrix, (truth[:, site], predicted[:, site]), 1)
        support = matrix.sum(1)
        rows.append(
            {
                "carrier": site,
                "accuracy": float((truth[:, site] == predicted[:, site]).mean()),
                "class_support": support.tolist(),
                "class_recall": [
                    None if not support[c] else float(matrix[c, c] / support[c]) for c in range(5)
                ],
                "confusion_true_by_pred": matrix.tolist(),
            }
        )
    return rows


def main():
    source = manifest()
    path = OUT / "per_site_class_audit.json"
    if path.exists():
        raise RuntimeError("preserve existing per-site audit")
    parent = load(PARENT)
    rows, stages = features(parent, 32, 1011)
    truth = rows["length"]
    expected = json.loads((OUT / "diagnosis.json").read_text())
    result = {}
    for stage_index, (stage, x) in enumerate(stages.items()):
        head = parent["encoder"]["target_l"]
        current: dict[str, object] = {
            "existing_codec_head": per_site(
                np.asarray(jnp.argmax(jnp.asarray(x) @ head["weight"] + head["bias"], -1)), truth
            )
        }
        for kind_index, mlp in enumerate((False, True)):
            saved_path = RUN / "probes" / f"probe_{stage_index * 2 + kind_index:02d}.pkl"
            with saved_path.open("rb") as f:
                saved = pickle.load(f)  # noqa: S301
            n = saved["normalization"]
            local = jnp.asarray((x - n["local_mean"]) / n["local_scale"])
            glob = jnp.asarray((x.reshape(len(x), -1) - n["global_mean"]) / n["global_scale"])
            ell, _ = probe_apply(saved["params"], local, glob, mlp)
            prediction = np.asarray(jnp.argmax(ell, -1))
            kind = "small_mlp" if mlp else "production_linear"
            report = extent_report(prediction, truth)
            if (
                report["exact_total_length_accuracy"]
                != expected["stages"][stage][kind]["exact_total_length_accuracy"]
            ):
                raise RuntimeError("saved probe does not reproduce held-out diagnosis")
            current[kind] = {
                "weights_sha256": digest(saved_path),
                "by_carrier": per_site(prediction, truth),
            }
        result[stage] = current
    candidate_record = json.loads((OUT / "frozen_codec.json").read_text())
    frozen = ROOT / candidate_record["path"]
    if digest(frozen) != candidate_record["sha256"]:
        raise RuntimeError("candidate changed")
    candidate_rows, candidate_stages = features(load(frozen), 64, 1013)
    head = load(frozen)["encoder"]["target_l"]
    candidate = {
        stage: per_site(
            np.asarray(jnp.argmax(jnp.asarray(x) @ head["weight"] + head["bias"], -1)),
            candidate_rows["length"],
        )
        for stage, x in candidate_stages.items()
    }
    splits = {}
    import hashlib

    for name, count, seed in (
        ("G0_train", 256, 963),
        ("G0_test", 32, 962),
        ("G01_train", 256, 1012),
        ("G01_test", 64, 1013),
    ):
        values, masks = map(np.asarray, data(count, seed))
        splits[name] = {
            hashlib.sha256(x[m].tobytes()).hexdigest() for x, m in zip(values, masks, strict=True)
        }
    cross = {
        f"{a}/{b}": len(splits[a] & splits[b])
        for a in ("G0_train", "G0_test")
        for b in ("G01_train", "G01_test")
    }
    if any(cross.values()):
        raise RuntimeError("G0/G01 example reuse")
    write(
        path,
        {
            "verifier": source,
            "G0_probe_validation": result,
            "G01_test_readout": candidate,
            "actual_G_input": "bitwise aliases the corresponding noisy stage; see real-model control",
            "G0_G01_exact_example_intersections": cross,
        },
    )


if __name__ == "__main__":
    main()
