"""Frozen G2/G3 campaign: matched generated policies, bounded budgets, no H/I."""

import argparse
from dataclasses import asdict
import fcntl
import hashlib
import importlib.metadata
import json
import os
import platform
import resource
import sys
from pathlib import Path
import pickle
import shutil
import subprocess
import time
import traceback
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

import run_phase_g11_teacher as old
from run_phase_g0 import tree_hash
from adze_t.backends.torx import TorxOps, TorxOperatorConfig
from adze_t.objectives import adamw_init
from adze_t.phase_g2 import (
    inference,
    objective,
    train_step,
    named_key,
    aligned_metrics,
    reachability,
)
from adze_t.phase_g2_storage import RollingJournal
from adze_t.phase_g11_records import sha, write_json, finite
from adze_t.phase_g11_training import apply_observation
from adze_t.phase_g0 import dynamic_capacity_routing
from adze_t.phase_g import neutral_structure
from adze_t.training import stochastic_model_update_mask
from adze_t.state import CommittedStructure

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/phase_g/g2/v1"
G3 = ROOT / "results/phase_g/g3/v1"
RUN = ROOT / "results/runs/phase_g/g23/v1"
IDENTITY = OUT / "G2_G3_PROTOCOL_IDENTITY.json"
CFG = old.CFG
PARENT = ROOT / "results/runs/phase_g/g1_1/teacher_v0/official/payloads/100_000.pkl"
PARENT_SHA = "fa798be7d9d6244054b1053bd86c340c4d2f3582290dc72a31e6877c24f81a93"
INIT = "94f4b7de78406336b6e6bb87416981330a98e9d74b9bfe7c4280529807184bd2"
CHECKPOINTS = (0, 100, 250, 500, 1000, 2000)


ADJUDICATION_MANIFEST_SHA = "0eb4e639db7c7ed3f5ab021b861a3ed0ce2c377bf8eff88e283873b67608544f"


def verified_adjudication():
    from adjudicate_phase_g21 import audit_identity

    path = ROOT / "results/phase_g/g2_1/MANIFEST.json"
    if sha(path) != ADJUDICATION_MANIFEST_SHA:
        raise RuntimeError("adjudication manifest mismatch")
    manifest = json.loads(path.read_text())
    for name, digest in manifest["files"].items():
        if sha(ROOT / name) != digest:
            raise RuntimeError("adjudication evidence mismatch:" + name)
    identity = audit_identity()
    rows = {
        name: json.loads((ROOT / f"results/phase_g/g2_1/{name}.json").read_text())
        for name in ("audit", "calibration", "confirmation", "decision")
    }
    if any(x["identity"] != identity or not x["passed"] for x in rows.values()):
        raise RuntimeError("no matching confirmed adjudication")
    if rows["decision"]["label"] != "G21_GRADIENT_ESTIMATOR_CONFIRMED":
        raise RuntimeError("adjudication not confirmed")
    c = rows["confirmation"]
    if (c["family_size"], c["roots_per_fixture"], c["draws_per_root"]) != (42, 64, 4096):
        raise RuntimeError("incomplete adjudication family")
    if not c["new_and_original_keys_disjoint_unique"]:
        raise RuntimeError("adjudication key independence failed")
    for fixture in ("uniform", "asymmetric", "peaked"):
        if len(c["fixtures"][fixture]["accepted"]) != 14 or not all(
            c["fixtures"][fixture]["accepted"]
        ):
            raise RuntimeError("adjudication component nonpass")
    return {
        "identity": identity,
        "manifest_sha256": sha(path),
        "decision_sha256": sha(ROOT / "results/phase_g/g2_1/decision.json"),
        "method": "complete42 exact-variance Bernstein; G21 prospective64x4096",
        "original_G23_V0": "PHASE_G_BLOCKED",
    }


def log(s):
    print(time.strftime("%Y-%m-%dT%H:%M:%S"), s, flush=True)


def dump(path, value):
    write_json(
        path,
        json.loads(json.dumps(value, default=lambda x: np.asarray(x).tolist(), allow_nan=False)),
    )


def parent():
    if sha(PARENT) != PARENT_SHA or sha(old.v.CODEC) != old.v.EXPECTED[old.v.CODEC]:
        raise RuntimeError("accepted parent/codec mismatch")
    with PARENT.open("rb") as stream:
        state = pickle.load(stream)
    if state["step"] != 100 or tree_hash(state["params"]) != INIT:
        raise RuntimeError("parent optimizer/model identity mismatch")
    return jax.tree.map(jnp.asarray, state["params"])


def preserved():
    files = json.loads(
        (
            ROOT / "results/phase_g/preflight_g01_v1/f571ab40659d/historical_baseline.json"
        ).read_text()
    )["files"]
    for p, h in files.items():
        if sha(p) != h:
            raise RuntimeError(f"historical mutation:{p}")
    for sub in ("g1", "g1_1", "g1_1/teacher_v0"):
        path = ROOT / f"results/phase_g/{sub}/MANIFEST.json"
        manifest = json.loads(path.read_text())
        for p, h in manifest["files"].items():
            if sha(ROOT / p) != h:
                raise RuntimeError(f"accepted manifest mismatch:{p}")
        files[str(path)] = sha(path)
    receipt = old.OUT / "DELIVERY_CHECKS.json"
    files[str(receipt)] = sha(receipt)
    return files


def freeze():
    if IDENTITY.exists():
        raise RuntimeError("identity already frozen; audit it, never rewrite")
    p = parent()
    files = preserved()
    source = {
        str(path.relative_to(ROOT)): sha(path)
        for folder in ("src/adze_t", "scripts")
        for path in sorted((ROOT / folder).rglob("*.py"))
    }
    baseline = {"files": files, "accepted_G_commit": "7120e7de6e491e4028c92e8dabe31f3b92de7d70"}
    dump(OUT / "PRESERVATION_BASELINE.json", baseline)
    runtime = {
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "XLA_FLAGS": os.environ.get("XLA_FLAGS", ""),
        "prng_impl": str(getattr(jax.config, "jax_default_prng_impl")),
        "packages": {
            x: importlib.metadata.version(x) for x in ("jax", "jaxlib", "numpy", "extro-torx")
        },
        "backend": jax.default_backend(),
        "x64": bool(jax.config.read("jax_enable_x64")),
        "torx_direct_url": json.loads(
            importlib.metadata.distribution("extro-torx").read_text("direct_url.json") or "{}"
        ),
    }
    binding = {
        "version": "PHASE_G23_V1",
        "adjudication": verified_adjudication(),
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "source": source,
        "protocols": {
            str(p.relative_to(ROOT)): sha(p)
            for p in (OUT / "G2_PROTOCOL.md", G3 / "G3_PROTOCOL.md")
        },
        "parent_sha256": PARENT_SHA,
        "initial_parameter_tree": tree_hash(p),
        "codec_sha256": sha(old.v.CODEC),
        "config": asdict(CFG),
        "runtime": runtime,
        "update_mask": tree_hash(stochastic_model_update_mask(p)),
        "datasets": json.loads((old.v.OUT / "dataset_identity.json").read_text()),
        "preservation_baseline_sha256": sha(OUT / "PRESERVATION_BASELINE.json"),
    }
    identity = hashlib.sha256(json.dumps(binding, sort_keys=True).encode()).hexdigest()
    dump(IDENTITY, {"identity": identity, "binding": binding})
    log(f"FROZEN identity {identity}; commit this before preflight/training")


def audit():
    record = json.loads(IDENTITY.read_text())
    b = record["binding"]
    if b.get("version") != "PHASE_G23_V1" or b.get("adjudication") != verified_adjudication():
        raise RuntimeError("V1 adjudication/protocol mismatch")
    if hashlib.sha256(json.dumps(b, sort_keys=True).encode()).hexdigest() != record["identity"]:
        raise RuntimeError("protocol identity corrupt")
    for name, h in {**b["source"], **b["protocols"]}.items():
        if sha(ROOT / name) != h:
            raise RuntimeError(f"source/protocol mismatch:{name}")
    for name, version in b["runtime"]["packages"].items():
        if importlib.metadata.version(name) != version:
            raise RuntimeError("runtime package mismatch")
    direct = json.loads(
        importlib.metadata.distribution("extro-torx").read_text("direct_url.json") or "{}"
    )
    if (
        direct != b["runtime"]["torx_direct_url"]
        or direct.get("vcs_info", {}).get("commit_id") != "f1fc858ed950ecd41935d15c06d0ec7c5e0674ae"
    ):
        raise RuntimeError("Torx revision mismatch")
    if (
        jax.default_backend() != b["runtime"]["backend"]
        or bool(jax.config.read("jax_enable_x64")) != b["runtime"]["x64"]
    ):
        raise RuntimeError("backend mismatch")
    for name, current in {
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "XLA_FLAGS": os.environ.get("XLA_FLAGS", ""),
        "prng_impl": str(getattr(jax.config, "jax_default_prng_impl")),
    }.items():
        if current != b["runtime"][name]:
            raise RuntimeError("runtime identity mismatch:" + name)
    parent()
    return record["identity"]


def batch_for(data, step, seed):
    epoch, i = divmod(step, 57)
    ids = jax.random.permutation(jax.random.fold_in(named_key("dataset_order", seed), epoch), 1824)
    return old.v.take(data, ids[i * 32 : (i + 1) * 32])


def frozen_fingerprint(params, moments, mask):
    digest = hashlib.sha256()
    for x, m in zip(
        jax.tree.leaves((params, moments)), jax.tree.leaves((mask, (mask, mask))), strict=True
    ):
        if not bool(m):
            value = np.asarray(x)
            digest.update(value.dtype.str.encode())
            digest.update(str(value.shape).encode())
            digest.update(value.tobytes())
    return digest.hexdigest()


def summary(rows):
    skip = {
        "b",
        "ell",
        "capacity_cut",
        "prediction_b",
        "prediction_ell",
        "global_example_id",
        "extent_class_support",
        "extent_class_correct",
        "pack_m",
        "pack_k",
    }
    r: dict[str, Any] = {k: float(np.asarray(x).mean()) for k, x in rows.items() if k not in skip}
    for metric, support in (
        ("query_nll", "query_nll_support"),
        ("nonquery_nll", "nonquery_nll_support"),
        ("byte_nll", "byte_nll_support"),
    ):
        if support in rows:
            valid = np.asarray(rows[support]) > 0
            r[metric] = float(np.asarray(rows[metric])[valid].mean()) if valid.any() else None
    if "b_tp" in rows:
        tp = np.asarray(rows["b_tp"]).sum()
        fp = np.asarray(rows["b_fp"]).sum()
        fn = np.asarray(rows["b_fn"]).sum()
        r["active_boundary_f1"] = float(2 * tp / max(2 * tp + fp + fn, 1))
    if "extent_class_support" in rows:
        support = np.asarray(rows["extent_class_support"]).sum(0)
        correct = np.asarray(rows["extent_class_correct"]).sum(0)
        r["extent_class_support"] = support.tolist()
        r["extent_class_recall"] = [
            float(c / s) if s > 0 else None for c, s in zip(correct, support, strict=True)
        ]
    if "partial_support" in rows:
        support = np.asarray(rows["partial_support"]).sum()
        r["partial_extent_accuracy"] = (
            float(np.asarray(rows["partial_correct"]).sum() / support) if support > 0 else None
        )
    return r


def replay_metrics(params, data, b, ell, *, config=CFG):
    capacity = dynamic_capacity_routing(
        CommittedStructure(b, ell > 0, ell),
        M_max=config.packing.M_max,
        K=config.packing.K,
        max_extent=config.carrier.L_max,
    )
    route = capacity.routing
    ops = TorxOps.create(
        named_key("operator"), config=TorxOperatorConfig(operator_stochasticity=True, lambda_op=0.0)
    )
    executed, (hh, _, _, logits) = apply_observation(
        params, data["source_h"], data["prompt"], route.structure, ops, config=config
    )
    m = aligned_metrics(logits, ell, data["slots"], data["slot_mask"], data["q"])
    reach = reachability(route, config=config)
    return {
        **m,
        "h_mse": ((hh - data["target_h"]) ** 2).mean((1, 2)),
        "peer_reach": reach[jnp.arange(len(b)), data["q"], data["peer"]]
        & (ell[jnp.arange(len(b)), data["peer"]] > 0),
        "executed": executed,
        "b": b,
        "ell": ell,
        "capacity_cut": capacity.capacity_cut,
        "pack_m": route.metadata.carrier_to_m,
        "pack_k": route.metadata.carrier_to_k,
    }


REPLAY = jax.jit(replay_metrics)
INFER = {
    policy: jax.jit(
        lambda p, h, prompt, ids, key, op, policy=policy: inference(
            p, h, prompt, ids, key, op, 0, policy=policy, config=CFG
        )
    )
    for policy in ("B", "C")
}
OBJECTIVES = {
    policy: jax.jit(
        lambda p, b, key, op, policy=policy: objective(p, b, 0, key, op, policy=policy, config=CFG)
    )
    for policy in ("A", "B", "C")
}


def case_boundaries(name, b, data):
    q = data["q"]
    row = jnp.arange(len(q))
    if name == "cut":
        b = b.at[row, q].set(1)
        b = b.at[row, jnp.maximum(q - 1, 0)].set(jnp.where(q > 0, 1, b[row, jnp.maximum(q - 1, 0)]))
    elif name == "neutral":
        b = neutral_structure(len(b), CFG).c_b
    elif name == "shifted":
        b = jnp.concatenate(
            (jnp.roll(data["boundary"][:, :-1], 1, axis=1), jnp.ones((len(b), 1), jnp.int32)), 1
        )
    elif name == "teacher_boundary":
        b = data["boundary"]
    elif name == "overconnected":
        b = jnp.zeros_like(b).at[:, -1].set(1)
        b = b.at[row, jnp.maximum(q - 1, 0)].set(jnp.where(q > 0, 1, b[row, jnp.maximum(q - 1, 0)]))
    else:
        raise ValueError(name)
    return b.at[:, -1].set(1)


def evaluate(params, data, *, seed, arm, step, split, mode, root=0, control="policy"):
    identity = audit()
    model = tree_hash(params)
    checkpoint = RUN / f"seed{seed}/{arm}/snapshots/update_{step:05d}.pkl"
    with checkpoint.open("rb") as stream:
        saved_state = pickle.load(stream)
    if (
        saved_state["step"] != step
        or saved_state["campaign_identity"] != identity
        or saved_state["seed"] != seed
        or saved_state["arm"] != arm
        or tree_hash(saved_state["params"]) != model
    ):
        raise RuntimeError("evaluation parameters are not the claimed exact checkpoint")
    path = RUN / f"seed{seed}/{arm}/evaluation/{split}_{step:05d}_{mode}_{root:02d}_{control}.npz"
    meta = path.with_suffix(".json")
    expected = {
        "identity": identity,
        "model_tree": model,
        "seed": seed,
        "arm": arm,
        "step": step,
        "checkpoint_sha256": sha(checkpoint),
        "optimizer_tree": tree_hash(saved_state["moments"]),
        "split": split,
        "mode": mode,
        "root": root,
        "structure_root_key": np.asarray(
            jax.random.key_data(named_key("evaluation_structure", seed, root))
        ).tolist(),
        "operator_root_key": np.asarray(jax.random.key_data(named_key("operator", seed))).tolist(),
        "control": control,
        "ids_sha256": hashlib.sha256(np.asarray(data["global_example_id"]).tobytes()).hexdigest(),
        "dataset_sha256": json.loads(IDENTITY.read_text())["binding"]["datasets"]["ordered_sha256"][
            "protected_test" if split == "test" else "validation"
        ],
    }
    if meta.exists():
        saved = json.loads(meta.read_text())
        if saved["provenance"] != expected or sha(path) != saved["sha256"]:
            raise RuntimeError("mixed/stale evaluation shard")
        with np.load(path) as archive:
            cached = {k: archive[k] for k in archive.files}
        if not np.array_equal(cached["global_example_id"], np.asarray(data["global_example_id"])):
            raise RuntimeError("saved shard example identities mismatch")
        return cached
    if path.exists():
        orphan = path.with_name(
            path.stem
            + f"_uncommitted_{len(list(path.parent.glob(path.stem + '_uncommitted_*'))):03d}.npz"
        )
        path.rename(orphan)
        dump(
            orphan.with_suffix(".json"),
            {"not_completed": True, "provenance": expected, "sha256": sha(orphan)},
        )
    rows = []
    for start in range(0, len(data["q"]), 32):
        d = old.v.take(data, jnp.arange(start, min(start + 32, len(data["q"]))))
        policy = {"teacher": "A", "argmax": "B", "sample": "C"}[mode]
        key = named_key("evaluation_structure", seed, root)
        _, m = OBJECTIVES[policy](params, d, key, named_key("operator", seed))
        if not bool(m["executed"]) or not finite(m):
            raise RuntimeError("invalid/nonfinite eval policy")
        if control != "policy":
            if control in ("no_attention", "direct_decoder"):
                from adze_t.backends.mapping import torx_means_to_deterministic

                det = jax.tree.map(jnp.asarray, torx_means_to_deterministic(params))
                if control == "no_attention":
                    model_out = jax.jit(given_no_attention)(
                        det, d["source_h"], d["prompt"], m["b"], m["ell"]
                    )
                    logits = model_out["logits"]
                    carrier = model_out["h_hat"]
                else:
                    logits = jax.jit(old.v.decoder)(d["source_h"], m["ell"], det["decoder"])
                    carrier = d["source_h"]
                changed = {
                    **aligned_metrics(logits, m["ell"], d["slots"], d["slot_mask"], d["q"]),
                    "executed": jnp.asarray(True),
                    "b": m["b"],
                    "ell": m["ell"],
                    "mask_peer_reach": m["peer_reach"],
                    "peer_reach": jnp.zeros_like(m["peer_reach"]),
                    "h_mse": ((carrier - d["target_h"]) ** 2).mean((1, 2)),
                }
            else:
                b = case_boundaries(control, m["b"], d)
                changed = REPLAY(params, d, b, m["ell"])
            if not bool(changed["executed"]) or not finite(changed):
                raise RuntimeError("invalid eval intervention")
            m = {**m, **changed}
            active = d["length"][:, :-1] > 0
            chosen = m["b"][:, :-1]
            labels = d["boundary"][:, :-1]
            m.update(
                {
                    "b_accuracy": ((chosen == labels) & active).sum(1)
                    / jnp.maximum(active.sum(1), 1),
                    "b_tp": ((chosen == 1) & (labels == 1) & active).sum(1),
                    "b_fp": ((chosen == 1) & (labels == 0) & active).sum(1),
                    "b_fn": ((chosen == 0) & (labels == 1) & active).sum(1),
                    "structure_exact": jnp.all(m["b"] == d["boundary"], 1)
                    & jnp.all(m["ell"] == d["length"], 1),
                    "base_log_prob": m["log_prob"],
                    "log_prob": jnp.zeros_like(m["log_prob"]),
                    "intervened": jnp.ones(len(chosen), bool),
                }
            )
        rows.append({k: np.asarray(x) for k, x in m.items() if k != "executed"})
    result = {k: np.concatenate([r[k] for r in rows]) for k in rows[0]}
    result["global_example_id"] = np.asarray(data["global_example_id"])
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with tmp.open("wb") as stream:
        save_arrays: Any = np.savez_compressed
        save_arrays(stream, **result)
    tmp.replace(path)
    dump(meta, {"provenance": expected, "sha256": sha(path), "summary": summary(result)})
    return result


def mean_roots(rows):
    # Structural actions remain in per-root shards; averages are diagnostics,
    # never a new hard policy or voted/averaged-logit decoder.
    result = {k: np.mean([r[k] for r in rows], axis=0) for k in rows[0]}
    for metric, support in (
        ("query_nll", "query_nll_support"),
        ("nonquery_nll", "nonquery_nll_support"),
        ("byte_nll", "byte_nll_support"),
    ):
        if support in result:
            weights = np.sum([r[support] for r in rows], axis=0)
            result[metric] = np.sum([r[metric] * r[support] for r in rows], axis=0) / np.maximum(
                weights, 1
            )
    return result


def given_no_attention(params, h, prompt, b, ell):
    return old.v.forward(params, h, prompt, b, ell, no_attention=True)


def validation(params, data, seed, arm, step):
    modes = ("teacher",) if arm == "A" else (("argmax",) if arm == "B" else ("argmax", "sample"))
    result = {}
    for mode in modes:
        roots = range(4) if mode == "sample" else range(1)
        r = [
            evaluate(
                params,
                data,
                seed=seed,
                arm=arm,
                step=step,
                split="validation",
                mode=mode,
                root=root,
            )
            for root in roots
        ]
        cut = [
            evaluate(
                params,
                data,
                seed=seed,
                arm=arm,
                step=step,
                split="validation",
                mode=mode,
                root=root,
                control="cut",
            )
            for root in roots
        ]
        result[mode] = {"summary": summary(mean_roots(r)), "cut": summary(mean_roots(cut))}
    # Same weights under full teacher control: reference gap, not copied A model.
    teacher = evaluate(
        params, data, seed=seed, arm=arm, step=step, split="validation", mode="teacher"
    )
    result["teacher_reference"] = summary(teacher)
    dump(
        OUT / f"seed{seed}/{arm}/validation_{step:05d}.json",
        {"identity": audit(), "model_tree": tree_hash(params), "step": step, "modes": result},
    )
    return result


def guards(s):
    baseline = json.loads((old.OUT / "validation_100_A.json").read_text())["summary"]
    return (
        s["nonquery_accuracy"] >= baseline["nonquery_accuracy"] - 0.02
        and s["nonquery_loss"] <= 1.05 * baseline["nonquery_nll"]
    )


def select(seed, arm, cap):
    candidates = []
    mode = {"A": "teacher", "B": "argmax", "C": "sample"}[arm]
    for step in (0, 250, 500, 1000, 2000):
        if step > cap:
            continue
        path = OUT / f"seed{seed}/{arm}/validation_{step:05d}.json"
        if not path.exists():
            continue
        record = json.loads(path.read_text())
        snapshot = RUN / f"seed{seed}/{arm}/snapshots/update_{step:05d}.pkl"
        receipt = json.loads((RUN / f"seed{seed}/{arm}/receipts/{step:05d}.json").read_text())
        if (
            record["identity"] != audit()
            or record["step"] != step
            or sha(snapshot) != receipt["state_sha256"]
        ):
            raise RuntimeError("candidate checkpoint/validation provenance mismatch")
        with snapshot.open("rb") as stream:
            candidate_state = pickle.load(stream)
        if (
            tree_hash(candidate_state["params"]) != record["model_tree"]
            or candidate_state["step"] != step
            or candidate_state["campaign_identity"] != record["identity"]
        ):
            raise RuntimeError("candidate validation does not belong to snapshot means")
        s = record["modes"][mode]["summary"]
        candidates.append(
            {
                "step": step,
                "eligible": bool(guards(s) and s["valid"] >= 0.99),
                "query_loss": s["query_loss"],
                "query_accuracy": s["query_accuracy"],
                "model_tree": record["model_tree"],
                "optimizer_tree": tree_hash(candidate_state["moments"]),
            }
        )
    eligible = [r for r in candidates if r["eligible"]]
    if not eligible:
        return {"selected": None, "candidates": candidates}
    best = min(r["query_loss"] for r in eligible)
    tied = [r for r in eligible if abs(r["query_loss"] - best) <= 1e-8]
    chosen = min(tied, key=lambda r: (-r["query_accuracy"], r["step"]))
    path = RUN / f"seed{seed}/{arm}/snapshots/update_{chosen['step']:05d}.pkl"
    return {
        "selected": {
            "step": chosen["step"],
            "path": str(path.relative_to(ROOT)),
            "sha256": sha(path),
            "mode": mode,
            "model_tree": chosen["model_tree"],
            "optimizer_tree": chosen["optimizer_tree"],
        },
        "candidates": candidates,
    }


def train_arm(seed, arm, cap, train, val, initial, identity):
    segment_started = time.monotonic()
    journal = RollingJournal(RUN / f"seed{seed}/{arm}", identity + f":seed{seed}:{arm}")
    state = journal.recover()
    zeros = (adamw_init(initial), adamw_init(initial))
    mask = stochastic_model_update_mask(initial)
    frozen = frozen_fingerprint(initial, zeros, mask)
    if state is None:
        state = {
            "params": initial,
            "moments": zeros,
            "step": 0,
            "identity": journal.identity,
            "campaign_identity": identity,
            "seed": seed,
            "arm": arm,
        }
        journal.append(state, {"initial_tree": INIT, "fresh_Adam": True}, snapshot=True)
    if frozen_fingerprint(state["params"], state["moments"], mask) != frozen:
        raise RuntimeError("resumed frozen state differs")
    if state["step"] > cap:
        raise RuntimeError("resume beyond declared budget")
    segment_start_step = state["step"]
    key = named_key("training_structure", seed)
    op = named_key("operator", seed)
    stepfn = jax.jit(lambda p, m, s, b: train_step(p, m, s, b, key, op, policy=arm, config=CFG))
    for current in range(state["step"], cap + 1):
        if current in CHECKPOINTS:
            log(f"seed{seed} {arm} validation {current}")
            validation(state["params"], val, seed, arm, current)
        if current == cap:
            break
        if shutil.disk_usage(RUN).free < 2 * 1024**3:
            raise RuntimeError("critical disk capacity; exact durable state preserved")
        started = time.monotonic()
        p, m, s, metrics = stepfn(
            state["params"], state["moments"], jnp.asarray(current), batch_for(train, current, seed)
        )
        jax.block_until_ready(s)
        if (
            int(s) != current + 1
            or not bool(metrics["executed"])
            or not finite((p, m, metrics))
            or frozen_fingerprint(p, m, mask) != frozen
        ):
            raise RuntimeError(f"numerical/contract failure {seed}/{arm}/{current + 1}")
        state = {**state, "params": p, "moments": m, "step": int(s)}
        values = summary(metrics)
        values["frozen_fingerprint"] = frozen
        values["frozen_state_checked_unchanged"] = True
        values["seconds"] = time.monotonic() - started
        values["peak_rss_mib"] = float(
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            / (1024**2 if platform.system() == "Darwin" else 1024)
        )
        journal.append(state, values, snapshot=int(s) in CHECKPOINTS)
        if int(s) % 10 == 0:
            log(
                f"seed{seed} {arm} update {int(s)}/{cap}: loss={float(metrics['loss']):.5f} query={values['query_accuracy']:.4f}, seconds={values['seconds']:.2f}"
            )
    directory = OUT / f"seed{seed}/{arm}/runtime_segments"
    dump(
        directory / f"segment_{len(list(directory.glob('*.json'))):03d}.json",
        {
            "identity": identity,
            "start_step": segment_start_step,
            "end_step": state["step"],
            "declared_cap": cap,
            "wall_seconds": time.monotonic() - segment_started,
            "model_applications_per_update": 2,
            "physical_block_forward_applications_per_update": 24,
            "resumed_without_optimizer_reset": True,
        },
    )
    return state


def bootstrap(a, b, lengths, contrast):
    a = np.asarray(a)
    b = np.asarray(b)
    groups = [np.flatnonzero(lengths == n) for n in np.unique(lengths)]
    keys = jax.random.split(named_key("bootstrap:" + contrast), len(groups))
    idx = np.concatenate(
        [
            g[np.asarray(jax.random.randint(k, (2000, len(g)), 0, len(g)))]
            for g, k in zip(groups, keys, strict=True)
        ],
        1,
    )
    delta = (a[idx] - b[idx]).mean(1)
    denom = b[idx].mean(1)
    relative = a[idx].mean(1) / denom - 1 if np.all(denom > 0) else None
    return {
        "delta": float((a - b).mean()),
        "ci95": np.percentile(delta, [2.5, 97.5]).tolist(),
        "relative": float(a.mean() / b.mean() - 1) if b.mean() > 0 else None,
        "relative_ci95": np.percentile(relative, [2.5, 97.5]).tolist()
        if relative is not None
        else None,
        "support": len(a),
        "replicates": 2000,
    }


def functional(rows, cut, data, contrast):
    s = summary(rows)
    c = summary(cut)
    lengths = np.asarray(data["length"].sum(1))
    stats = {
        k: bootstrap(rows[k], cut[k], lengths, contrast + ":" + k)
        for k in ("query_accuracy", "query_loss", "query_nll")
    }
    checks = {
        "boundary": s["b_accuracy"] >= 0.95,
        "length": s["length_accuracy"] >= 0.98,
        "valid": s["valid"] >= 0.99,
        "reach": s["peer_reach"] >= 0.95,
        "accuracy": stats["query_accuracy"]["delta"] >= 0.10,
        "query_loss": s["query_loss"] <= 0.9 * c["query_loss"],
        "nll": s["query_nll"] is not None
        and c["query_nll"] is not None
        and s["query_nll"] <= 0.9 * c["query_nll"],
        "nll_support": bool(
            np.all(rows["query_nll_support"] > 0) & np.all(cut["query_nll_support"] > 0)
        ),
        "uncertainty": stats["query_accuracy"]["ci95"][0] > 0
        and stats["query_loss"]["ci95"][1] < 0
        and stats["query_nll"]["ci95"][1] < 0,
        "competence": guards(s),
        "chance_margin": s["query_accuracy"] >= 1 / 28 + 0.10,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "summary": s,
        "cut": c,
        "paired": stats,
    }


def g2_decision(val):
    results = {}
    selections = {}
    for arm in ("A", "B", "C"):
        selected = select(0, arm, 1000)
        selections[arm] = selected
        if selected["selected"] is None:
            continue
        s = selected["selected"]
        p = old.load_pickle(ROOT / s["path"])["params"]
        for mode in ("argmax",) if arm == "B" else (("argmax", "sample") if arm == "C" else ()):
            roots = range(4) if mode == "sample" else range(1)
            rows = mean_roots(
                [
                    evaluate(
                        p,
                        val,
                        seed=0,
                        arm=arm,
                        step=s["step"],
                        split="validation",
                        mode=mode,
                        root=r,
                    )
                    for r in roots
                ]
            )
            cut = mean_roots(
                [
                    evaluate(
                        p,
                        val,
                        seed=0,
                        arm=arm,
                        step=s["step"],
                        split="validation",
                        mode=mode,
                        root=r,
                        control="cut",
                    )
                    for r in roots
                ]
            )
            results[arm + ":" + mode] = functional(rows, cut, val, "G2:" + arm + ":" + mode)
    proceed = any(r["passed"] for r in results.values())
    record = {
        "identity": audit(),
        "selections": selections,
        "modes": results,
        "G3_eligible": proceed,
    }
    dump(OUT / "G2_DECISION.json", record)
    return proceed


def extension():
    values = {}
    for step in (500, 1000):
        records = []
        for arm, mode in (("B", "argmax"), ("C", "sample")):
            records.append(
                json.loads((OUT / f"seed0/{arm}/validation_{step:05d}.json").read_text())["modes"][
                    mode
                ]["summary"]
            )
        values[str(step)] = {
            "accuracy": float(np.mean([r["query_accuracy"] for r in records])),
            "loss": float(np.mean([r["query_loss"] for r in records])),
            "guards": all(guards(r) for r in records),
        }
    choices = [select(0, arm, 1000)["selected"] for arm in ("B", "C")]
    frontier = all(choice is not None and choice["step"] == 1000 for choice in choices)
    material = (
        values["1000"]["accuracy"] - values["500"]["accuracy"] >= 0.01
        and values["1000"]["loss"] <= 0.98 * values["500"]["loss"]
    )
    cap = 2000 if frontier and material and values["1000"]["guards"] else 1000
    record = {
        "identity": audit(),
        "frontier": frontier,
        "material": material,
        "operands": values,
        "cap": cap,
    }
    dump(G3 / "EXTENSION_DECISION.json", record)
    return cap


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage", choices=("freeze", "preflight", "campaign", "science", "close"), required=True
    )
    args = parser.parse_args()
    RUN.mkdir(parents=True, exist_ok=True)
    with (RUN / "campaign.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.stage == "freeze":
            freeze()
            return
        identity = audit()
        attempts = list(RUN.glob("launch_*.json"))
        dump(
            RUN / f"launch_{len(attempts):03d}.json",
            {
                "identity": identity,
                "stage": args.stage,
                "source_head": subprocess.check_output(
                    ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
                ).strip(),
                "command": sys.argv,
                "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "counted_update_state_not_reset": True,
            },
        )
        initial = parent()
        codec = old.load_pickle(old.v.CODEC)
        train, val = old.prepare(codec)
        if args.stage == "preflight":
            from preflight_phase_g23_v1 import run

            run(initial, codec, identity, train, val)
            return
        if args.stage == "campaign":
            pre = json.loads((OUT / "PREFLIGHT.json").read_text())
            if pre["identity"] != identity or not pre["passed"]:
                raise RuntimeError("no matching successful preflight")
            g2_path = OUT / "G2_DECISION.json"
            if g2_path.exists():
                g2 = json.loads(g2_path.read_text())
                if g2["identity"] != identity:
                    raise RuntimeError("stale G2 decision")
                proceed = g2["G3_eligible"]
            else:
                for arm in ("A", "B", "C"):
                    train_arm(0, arm, 1000, train, val, initial, identity)
                proceed = g2_decision(val)
            if not proceed:
                log("G2 nonpass: no dependent G3 training/test exposure")
                return
            extension_path = G3 / "EXTENSION_DECISION.json"
            if extension_path.exists():
                saved = json.loads(extension_path.read_text())
                if saved["identity"] != identity:
                    raise RuntimeError("stale extension decision")
                cap = saved["cap"]
            else:
                cap = extension()
            if cap == 2000:
                for arm in ("A", "B", "C"):
                    train_arm(0, arm, cap, train, val, initial, identity)
            for seed in (1, 2):
                for arm in ("A", "B", "C"):
                    train_arm(seed, arm, cap, train, val, initial, identity)
            selected = {
                f"seed{seed}:{arm}": select(seed, arm, cap)
                for seed in (0, 1, 2)
                for arm in ("A", "B", "C")
            }
            dump(
                G3 / "SELECTION.json",
                {
                    "identity": identity,
                    "cap": cap,
                    "selected": selected,
                    "protected_test_exposed": False,
                },
            )
            return
        from science_phase_g23_v1 import run, close

        if args.stage == "science":
            run(identity, codec, val, train)
        else:
            close(identity)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        RUN.mkdir(parents=True, exist_ok=True)
        number = len(list(RUN.glob("failure_*.json")))
        dump(
            RUN / f"failure_{number:03d}.json",
            {
                "exception": repr(exc),
                "traceback": traceback.format_exc(),
                "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "no_reinitialization": True,
            },
        )
        raise
