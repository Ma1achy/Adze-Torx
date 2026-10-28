"""Read-only post-selection G.0.1 controls; never updates/reselects a model."""

from __future__ import annotations

import json

import jax
import jax.numpy as jnp
import numpy as np

from run_phase_g0 import chunks, data, digest, load, save, tree_hash, write
from run_phase_g01 import (
    F1,
    F1_HASH,
    OUT,
    PARENT,
    PARENT_HASH,
    ROOT,
    RUN,
    evaluate,
    features,
    manifest,
    real_g_input_control,
)
from adze_t.backends.torx import TorxOperatorConfig, TorxOps
from adze_t.model import apply_target_codec
from adze_t.phase_g0 import with_frozen_g0_codec


def main():
    source = manifest()
    path = ROOT / json.loads((OUT / "frozen_codec.json").read_text())["path"]
    frozen = json.loads((OUT / "frozen_codec.json").read_text())
    if digest(path) != frozen["sha256"]:
        raise RuntimeError("frozen candidate changed")
    if (OUT / "postselection_audit.json").exists():
        raise RuntimeError("preserve existing post-selection audit")
    candidate, parent = load(path), load(PARENT)
    baseline = evaluate(parent, 64, 1013)
    selected = json.loads((OUT / "heldout.json").read_text())["evaluation"]
    original = load(F1)["params"]
    original_hash = tree_hash(original)
    installed = with_frozen_g0_codec(original, candidate)
    x, m = data(1, 1019)
    ops = TorxOps.create(
        jax.random.key(1023), config=TorxOperatorConfig(operator_stochasticity=True, lambda_op=0.0)
    )
    deterministic = apply_target_codec(candidate, x, m)
    torx = apply_target_codec(installed, x, m, ops=ops)
    parity = {
        key: float(jnp.max(jnp.abs(a - b)))
        for key, a, b in (
            ("h0", deterministic["target"]["h0"], torx["target"]["h0"]),
            ("byte_logits", deterministic["codec_logits"], torx["codec_logits"]),
            ("boundary_logits", deterministic["target"]["b_logits"], torx["target"]["b_logits"]),
            ("extent_logits", deterministic["target"]["l_logits"], torx["target"]["l_logits"]),
        )
    }
    if any(parity.values()):
        raise RuntimeError("deterministic/Torx parity failure")
    validation, stages = features(candidate, 32, 1011)
    real = real_g_input_control(candidate, stages, validation["length"])
    heldout_x, heldout_m = data(64, 1013)
    old_rows = chunks(parent, heldout_x, heldout_m)
    new_rows = chunks(candidate, heldout_x, heldout_m)
    masks = old_rows["mask"]
    old_correct = (old_rows["predicted"] == old_rows["bytes"]) | ~masks
    new_correct = (new_rows["predicted"] == new_rows["bytes"]) | ~masks
    paired = {
        "oracle_byte_accuracy_change_pp": 100
        * float((new_correct[masks].astype(float) - old_correct[masks]).mean()),
        "oracle_byte_nll_change": float((new_rows["ce"][masks] - old_rows["ce"][masks]).mean()),
        "oracle_exact_sequence_accuracy_change_pp": 100
        * float(
            (new_correct.all((1, 2)).astype(float) - old_correct.all((1, 2)).astype(float)).mean()
        ),
        "inferred_byte_accuracy_change_pp": 100
        * (
            selected["inferred_extent_reconstruction"]["byte_accuracy"]
            - baseline["inferred_extent_reconstruction"]["byte_accuracy"]
        ),
        "inferred_exact_sequence_accuracy_change_pp": 100
        * (
            selected["inferred_extent_reconstruction"]["exact_sequence_accuracy"]
            - baseline["inferred_extent_reconstruction"]["exact_sequence_accuracy"]
        ),
    }
    split_sets = {}
    splits = {}
    import hashlib

    for name, count, seed in (
        ("probe_train", 128, 1010),
        ("validation", 32, 1011),
        ("candidate_train", 256, 1012),
        ("test", 64, 1013),
    ):
        values, mask = map(np.asarray, data(count, seed))
        hashes = [
            hashlib.sha256(row[valid].tobytes()).hexdigest()
            for row, valid in zip(values, mask, strict=True)
        ]
        split_sets[name] = set(hashes)
        splits[name] = {
            "count": len(hashes),
            "unique": len(set(hashes)),
            "ordered_sha256": hashlib.sha256("".join(hashes).encode()).hexdigest(),
            "seed": seed,
        }
    intersections = {
        f"{a}/{b}": len(split_sets[a] & split_sets[b])
        for a in split_sets
        for b in split_sets
        if a < b
    }
    if any(intersections.values()):
        raise RuntimeError("dataset split overlap")
    changed_frozen = []
    for (p, a), (_, b) in zip(
        jax.tree_util.tree_leaves_with_path(original),
        jax.tree_util.tree_leaves_with_path(installed),
        strict=True,
    ):
        if "rho" in jax.tree_util.keystr(p) and not jnp.array_equal(a, b):
            changed_frozen.append(str(p))
    if (
        changed_frozen
        or tree_hash(original) != original_hash
        or digest(F1) != F1_HASH
        or digest(PARENT) != PARENT_HASH
    ):
        raise RuntimeError("accepted/frozen state mutation")
    init_path = RUN / "matched_G_initialization.pkl"
    if init_path.exists():
        raise RuntimeError("refusing matched initialization overwrite")
    save(init_path, installed)
    changes = subprocess_names()
    write(
        OUT / "postselection_audit.json",
        {
            "verifier": source,
            "frozen_sha256": frozen["sha256"],
            "G0_paired_test_baseline": baseline,
            "paired_changes": paired,
            "parity_max_absolute_error": parity,
            "actual_G_input_control": real,
            "data": splits,
            "split_intersections": intersections,
            "latent_geometry": {
                "G0_h0_rms": float(np.sqrt(np.mean(old_rows["h0"] ** 2))),
                "G01_h0_rms": float(np.sqrt(np.mean(new_rows["h0"] ** 2))),
                "nu_coefficients_and_unit_epsilon_unchanged": True,
            },
            "frozen_states_unchanged": True,
            "matched_G_initialization": {
                "path": str(init_path.relative_to(ROOT)),
                "sha256": digest(init_path),
                "tree_sha256": tree_hash(installed),
                "all_arm_tree_hashes": {
                    arm: tree_hash(installed) for arm in ("teacher", "argmax", "sample")
                },
                "fresh_Adam_absolute_step": 0,
                "approved_G_data_seeds": [950, 951, 952],
                "G_data_counts": [1024, 256],
            },
            "sampling_sources_changed_from_parent": changes,
        },
    )


def subprocess_names():
    import subprocess

    paths = [
        "src/adze_t/generated_structure.py",
        "src/adze_t/phase_g0.py",
        "src/adze_t/phase_g0_oracle.py",
        "src/adze_t/phase_g_oracle.py",
        "src/adze_t/train/score_bridge.py",
    ]
    changed = subprocess.check_output(
        ["git", "diff", "1b4eaf57b93691952c4046eddfaeb1ceddd64ec6", "--name-only", "--", *paths],
        cwd=ROOT,
        text=True,
    ).splitlines()
    if changed:
        raise RuntimeError("sampling policy/estimator changed; revalidation required")
    return changed


if __name__ == "__main__":
    main()
