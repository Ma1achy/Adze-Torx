"""Complete frozen selected-validation sixteen-root evidence, never select/train."""

import fcntl
import json
import pickle
from pathlib import Path

import numpy as np

import run_phase_g23_v1 as g
from audit_phase_g23_pause import journal


def main():
    with (g.RUN / "campaign.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        identity = g.audit()
        path = g.G3 / "SELECTION.json"
        selection = json.loads(path.read_text())
        expected = {f"seed{s}:{arm}" for s in (0, 1, 2) for arm in ("A", "B", "C")}
        if (
            selection["identity"] != identity
            or selection["cap"] != 1000
            or set(selection["selected"]) != expected
            or any(r["selected"] is None for r in selection["selected"].values())
        ):
            raise RuntimeError("no complete validation-only selection")
        for name in sorted(expected):
            seed, arm = name.split(":")
            state, _ = journal(g.RUN / seed / arm, identity + ":" + name, g.INIT)
            if state["completed_updates"] != 1000:
                raise RuntimeError("all matched training caps must complete first")
        _, val = g.old.prepare(g.old.load_pickle(g.old.v.CODEC))
        records = {}
        for name in sorted(expected):
            seed_name, arm = name.split(":")
            seed = int(seed_name[4:])
            chosen = selection["selected"][name]["selected"]
            payload = g.ROOT / chosen["path"]
            if g.sha(payload) != chosen["sha256"]:
                raise RuntimeError("selected payload changed")
            with payload.open("rb") as stream:
                state = pickle.load(stream)
            before = g.tree_hash((state["params"], state["moments"]))
            if (
                state["step"] != chosen["step"]
                or state["campaign_identity"] != identity
                or g.tree_hash(state["params"]) != chosen["model_tree"]
                or g.tree_hash(state["moments"]) != chosen["optimizer_tree"]
            ):
                raise RuntimeError("selected model/optimizer identity mismatch")
            result = {}
            for control in ("policy", "cut", "neutral"):
                g.log(f"selected validation {name} sample16 {control}")
                roots = [
                    g.evaluate(
                        state["params"],
                        val,
                        seed=seed,
                        arm=arm,
                        step=chosen["step"],
                        split="validation",
                        mode="sample",
                        root=root,
                        control=control,
                    )
                    for root in range(16)
                ]
                summaries = [g.summary(r) for r in roots]
                result[control] = {
                    "root_mean_metrics": g.summary(g.mean_roots(roots)),
                    "per_root_summaries": summaries,
                    "root_variability": {
                        metric: {
                            "sd": float(np.std([r[metric] for r in summaries], ddof=1)),
                            "range": [
                                min(r[metric] for r in summaries),
                                max(r[metric] for r in summaries),
                            ],
                        }
                        for metric in (
                            "query_accuracy",
                            "query_loss",
                            "query_nll",
                            "length_accuracy",
                            "peer_reach",
                        )
                    },
                }
            if g.tree_hash((state["params"], state["moments"])) != before:
                raise RuntimeError("evaluation mutated parameters/Adam")
            record = {
                "identity": identity,
                "driver_source_sha256": g.sha(Path(__file__)),
                "selection_sha256": g.sha(path),
                "checkpoint": chosen,
                "roots": list(range(16)),
                "split": "validation",
                "controls": result,
                "optimizer_updates": 0,
                "model_and_moments_unchanged": True,
                "selection_unchanged": True,
                "diagnostic_only_no_decision_override": True,
                "not_logit_averaging_or_ensemble_decoding": True,
            }
            g.dump(g.G3 / "selected_validation16" / f"{seed_name}_{arm}.json", record)
            records[name] = record
        g.dump(
            g.G3 / "SELECTED_VALIDATION16.json",
            {"identity": identity, "models": records, "selection_sha256": g.sha(path)},
        )


if __name__ == "__main__":
    main()
