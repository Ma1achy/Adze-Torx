"""Prospective disk recovery/state audit; no optimization or oracle resampling."""

import json
import pickle
import shutil
from pathlib import Path

import jax
import numpy as np

import run_phase_g23_v1 as g
from audit_phase_g23_pause import journal
from adze_t.objectives import adamw_init
from adze_t.training import stochastic_model_update_mask


OUT = g.OUT / "resource_recovery_20261027T1812"
BACKUP = g.RUN / "resource_stops/20261027T170605/seed1_B_update00137.pkl"
BACKUP_SHA = "f82269e87457c71f92117102ed6a7d0eb76a827476b9532f18f9dc32dda9af0d"
GIB = 1024**3
BUDGET = {
    "remaining_checkpoints_rings_atomic_staging": GIB,
    "evaluation_shards": 3 * GIB // 2,
    "reports_receipts_and_exports": GIB // 2,
    "backup_and_integration_fixture_copies": GIB // 2,
    "compiler_test_and_validation_temporary_reserve": GIB,
    "additional_volatility_reserve": GIB // 2,
    "unchanged_frozen_guard": 2 * GIB,
}


def main():
    free = shutil.disk_usage(g.ROOT).free
    if free < sum(BUDGET.values()):
        raise RuntimeError("insufficient whole-campaign storage headroom; no updates authorized")
    identity = g.audit()  # Reads accepted G21 proof, NEVER reruns adjudication.
    binding = json.loads(g.IDENTITY.read_text())["binding"]
    assert g.sha(BACKUP) == BACKUP_SHA
    with BACKUP.open("rb") as stream:
        backup = pickle.load(stream)
    assert backup["step"] == 137 and backup["seed"] == 1 and backup["arm"] == "B"
    initial = g.parent()
    mask = stochastic_model_update_mask(initial)
    zeros = (adamw_init(initial), adamw_init(initial))
    expected_frozen = g.frozen_fingerprint(initial, zeros, mask)
    checkpoints = {}
    expected_steps = {
        "seed0/A": 1000,
        "seed0/B": 1000,
        "seed0/C": 1000,
        "seed1/A": 1000,
        "seed1/B": 137,
    }
    for name, step in expected_steps.items():
        directory = g.RUN / name
        seed, arm = name.split("/")
        checked, _ = journal(directory, identity + f":{seed}:{arm}", g.INIT)
        assert checked["completed_updates"] == step
        with Path(checked["payload"]).open("rb") as stream:
            state = pickle.load(stream)
        assert state["step"] == step and state["campaign_identity"] == identity
        assert state["seed"] == int(seed[4:]) and state["arm"] == arm
        p, moments = state["params"], state["moments"]
        assert len(moments) == 2
        for moment in moments:
            assert jax.tree.structure(p) == jax.tree.structure(moment)
            for parameter, value in zip(jax.tree.leaves(p), jax.tree.leaves(moment), strict=True):
                assert np.asarray(parameter).shape == np.asarray(value).shape
                assert np.asarray(parameter).dtype == np.asarray(value).dtype
        assert g.finite((p, moments))
        assert all((np.asarray(x) >= 0).all() for x in jax.tree.leaves(moments[1]))
        assert any(np.any(x) for x in jax.tree.leaves(moments[0]))
        assert any(np.any(x) for x in jax.tree.leaves(moments[1]))
        assert g.frozen_fingerprint(p, moments, mask) == expected_frozen
        if name == "seed1/B":
            assert g.tree_hash(state) == g.tree_hash(backup)
        checkpoints[name] = {
            **checked,
            "model_tree": g.tree_hash(p),
            "optimizer_tree": g.tree_hash(moments),
            "moment_structures_shapes_dtypes_finite_and_nonzero": True,
            "second_moments_nonnegative": True,
            "frozen_fingerprint": expected_frozen,
        }
    datasets = {}
    train = None
    for split, seed, count, offset in (("train", 1140, 32, 0), ("validation", 1141, 16, 10000)):
        data = g.old.make_examples(seed, count, id_offset=offset)
        digest = g.old.v.dataset_hash(data)
        assert digest == binding["datasets"]["ordered_sha256"][split]
        datasets[split] = digest
        if split == "train":
            train = data
    assert train is not None
    keys = {
        name: np.asarray(jax.random.key_data(g.named_key(name, 1))).tolist()
        for name in ("dataset_order", "training_structure", "operator", "evaluation_structure")
    }
    assert len(set(tuple(x) for x in keys.values())) == len(keys)
    extension = json.loads((g.G3 / "EXTENSION_DECISION.json").read_text())
    assert extension["identity"] == identity and extension["cap"] == 1000
    assert not extension["material"]
    assert sum(x["completed_updates"] for x in checkpoints.values()) == 4137
    assert not (g.G3 / "SELECTION.json").exists()
    # Every new artifact is immutable. No saved journal or optimizer is edited.
    record = {
        "identity": identity,
        "auditor_source_sha256": g.sha(Path(__file__)),
        "free_bytes_before_audit": free,
        "free_bytes_after_audit": shutil.disk_usage(g.ROOT).free,
        "whole_campaign_budget_bytes_including_guard": BUDGET,
        "required_total_bytes": sum(BUDGET.values()),
        "deleted_files": [],
        "accepted_backup_sha256": g.sha(BACKUP),
        "backup_and_live_state_bitwise_tree_equal": True,
        "checkpoints": checkpoints,
        "prior_counted_updates": 4137,
        "next_seed": 1,
        "next_arm": "B",
        "next_optimizer_update": 138,
        "next_batch_global_example_ids": np.asarray(
            g.batch_for(train, 137, 1)["global_example_id"]
        ).tolist(),
        "dataset_identities": datasets,
        "seed1_namespace_root_words": keys,
        "common_cap": 1000,
        "extension_not_activated": True,
        "no_protected_test_generation": True,
        "no_model_forward_gradient_or_optimizer_update": True,
    }
    if record["free_bytes_after_audit"] < sum(BUDGET.values()):
        raise RuntimeError("storage headroom changed during audit; no resume")
    g.dump(OUT / "RESUME_AUDIT.json", record)
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
