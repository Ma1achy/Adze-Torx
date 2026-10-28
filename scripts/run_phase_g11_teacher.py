"""One authorized G11 teacher100 run; no test-set or other-arm execution."""

import argparse
from dataclasses import asdict
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import time

import jax
import jax.numpy as jnp
import numpy as np

import run_phase_g11_validation as v
from run_phase_g import load_pickle
from run_phase_g0 import tree_hash
from adze_t.backends.mapping import torx_means_to_deterministic
from adze_t.objectives import adamw_init
from adze_t.phase_g0 import with_frozen_g0_codec
from adze_t.phase_g11_task import make_examples, observation_control
from adze_t.phase_g11_training import LOSS_VERSION, teacher_objective, train_step, ce
from adze_t.phase_g11_records import (
    Journal,
    sha,
    write_json,
    finite,
    paired_bootstrap,
    nonquery_guard,
)
from adze_t.training import stochastic_model_update_mask

ROOT = v.ROOT
OUT = ROOT / "results/phase_g/g1_1/teacher_v0"
RUN = ROOT / "results/runs/phase_g/g1_1/teacher_v0"
INIT = "84ab204d1cea2a1833cbbdeadc7993879ad29a3c4c354d4960c221c951daaeeb"
CHECKPOINTS = (0, 10, 25, 50, 100)
CFG = v.CFG


def log(message):
    print(time.strftime("%Y-%m-%dT%H:%M:%S"), message, flush=True)


def dump(name, obj):
    write_json(
        OUT / name,
        json.loads(json.dumps(obj, default=lambda x: np.asarray(x).tolist(), allow_nan=False)),
    )


def audit():
    for path, expected in v.EXPECTED.items():
        if sha(path) != expected:
            raise RuntimeError(f"artifact mismatch:{path}")
    if (
        sha(v.OUT / "BOUNDED_TEACHER_PROTOCOL.md")
        != "35cbd1fed3b0490b73b83693c7246b01d5d41057295046b95d4022e514d8e29c"
    ):
        raise RuntimeError("frozen protocol changed")
    if (
        sha(v.OUT / "G11_RESULTS.md")
        != "e78c40c9cbe64a9343104fcf4b89ec0ba95228d4a4a62dca3b9176af583b2ce5"
    ):
        raise RuntimeError("task evidence changed")
    baseline = json.loads(
        (
            ROOT / "results/phase_g/preflight_g01_v1/f571ab40659d/historical_baseline.json"
        ).read_text()
    )["files"]
    for path, expected in baseline.items():
        if sha(path) != expected:
            raise RuntimeError(f"protected historical mismatch:{path}")
    for milestone, archive_name in (
        ("g1", "diagnostic_v1_archive"),
        ("g1_1", "validation_v0_archive"),
    ):
        manifest_path = ROOT / f"results/phase_g/{milestone}/MANIFEST.json"
        manifest = json.loads(manifest_path.read_text())
        archive = (
            Path("/Users/malachy/src/Adze-Torx")
            / f"results/runs/phase_g/{milestone}/{archive_name}"
        )
        for name, expected in manifest["files"].items():
            if sha(ROOT / name) != expected:
                raise RuntimeError(f"historical manifest mismatch:{name}")
            candidates = list(archive.rglob(Path(name).name))
            if not any(sha(p) == expected for p in candidates):
                raise RuntimeError(f"archive mismatch:{name}")
        if not any(sha(p) == sha(manifest_path) for p in archive.rglob("MANIFEST.json")):
            raise RuntimeError("archive manifest mismatch")
    codec = load_pickle(v.CODEC)
    # The artifact stores a codec directly, not a previous training state.
    params = with_frozen_g0_codec(load_pickle(v.F1)["params"], codec)
    params = jax.tree.map(jnp.asarray, params)
    if tree_hash(params) != INIT:
        raise RuntimeError("fresh initialization mismatch")
    mask = stochastic_model_update_mask(params)
    if not any(bool(x) for x in jax.tree.leaves(mask["decoder"])):
        raise RuntimeError("decoder must remain trainable")
    source = {
        str(p.relative_to(ROOT)): sha(p)
        for folder in ("src/adze_t", "scripts")
        for p in sorted((ROOT / folder).rglob("*.py"))
    }
    source["clarification"] = sha(OUT / "PROSPECTIVE_CLARIFICATION.md")
    protocol = {
        "bounded_protocol_sha256": sha(v.OUT / "BOUNDED_TEACHER_PROTOCOL.md"),
        "task_manifest_sha256": sha(v.OUT / "MANIFEST.json"),
        "initial_tree": INIT,
        "codec_sha256": sha(v.CODEC),
        "f1_sha256": sha(v.F1),
        "source": source,
        "config": asdict(CFG),
        "loss": LOSS_VERSION,
        "optimizer": {
            "lr": 0.0003,
            "weight_decay": 0.01,
            "clip": 1.0,
            "batch_size": 32,
            "updates": 100,
            "fresh_step": 0,
        },
        "roots": {
            "shuffle": 1160,
            "operator": 1146,
            "accuracy_bootstrap": 1161,
            "nll_bootstrap": 1162,
        },
        "datasets": json.loads((v.OUT / "dataset_identity.json").read_text()),
        "update_mask": tree_hash(mask),
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "packages": {
                x: importlib.metadata.version(x) for x in ("jax", "jaxlib", "numpy", "extro-torx")
            },
            "backend": jax.default_backend(),
            "x64": bool(jax.config.read("jax_enable_x64")),
        },
    }
    identity = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
    dump("protocol_binding.json", {"sha256": identity, "protocol": protocol})
    dump(
        "source_snapshot.json",
        {
            "commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip(),
            "source": source,
            "historical_files_verified": len(baseline),
        },
    )
    return params, codec, identity


def frozen_hash(params, moments=None):
    mask = stochastic_model_update_mask(params)
    tree = params if moments is None else (params, moments)
    masks = mask if moments is None else (mask, (mask, mask))
    return tree_hash(jax.tree.map(lambda p, m: jnp.where(m, jnp.zeros_like(p), p), tree, masks))


def prepare(codec):
    identity = json.loads((v.OUT / "dataset_identity.json").read_text())
    encode = jax.jit(v.encode_fn)
    result = []
    for name, seed, n, offset in (("train", 1140, 32, 0), ("validation", 1141, 16, 10000)):
        data = make_examples(seed, n, id_offset=offset)
        if v.dataset_hash(data) != identity["ordered_sha256"][name]:
            raise RuntimeError("dataset changed")
        hs = []
        ht = []
        for start in range(0, len(data["q"]), 32):
            d = v.take(data, jnp.arange(start, min(start + 32, len(data["q"]))))
            hs.append(np.asarray(encode(codec, d["source"], d["source_mask"])))
            ht.append(np.asarray(encode(codec, d["target"], d["target_mask"])))
        data = {
            **data,
            "source_h": jnp.asarray(np.concatenate(hs)),
            "target_h": jnp.asarray(np.concatenate(ht)),
            "slots": jnp.pad(data["target"], ((0, 0), (0, 64))).reshape(-1, 32, 4),
            "slot_mask": jnp.arange(4)[None, None, :] < data["length"][:, :, None],
        }
        result.append(data)
        log(f"cached {name}: {len(data['q'])}, source/target frozen separately")
    return result


def summary(metrics):
    result = {
        k: float(np.asarray(x).mean())
        for k, x in metrics.items()
        if k not in ("predicted_b", "predicted_l")
    }
    tp = np.asarray(metrics.get("boundary_tp", 0)).sum()
    fp = np.asarray(metrics.get("boundary_fp", 0)).sum()
    fn = np.asarray(metrics.get("boundary_fn", 0)).sum()
    result["boundary_f1"] = float(2 * tp / max(2 * tp + fp + fn, 1))
    if "nonquery_count" in metrics:
        counts = np.asarray(metrics["nonquery_count"])
        result["nonquery_byte_weighted_accuracy"] = float(
            np.sum(np.asarray(metrics["nonquery_accuracy"]) * counts) / counts.sum()
        )
        result["nonquery_byte_weighted_nll"] = float(
            np.sum(np.asarray(metrics["nonquery_nll"]) * counts) / counts.sum()
        )
    return result


def evaluate(params, data, step, cut=False, *, protocol):
    fn = jax.jit(lambda p, b: teacher_objective(p, b, 0, cut=cut, config=CFG))
    rows = []
    for start in range(0, len(data["q"]), 32):
        _, m = fn(params, v.take(data, jnp.arange(start, min(start + 32, len(data["q"])))))
        if not bool(m["executed"]) or not finite(m):
            raise RuntimeError("invalid validation")
        rows.append({k: np.asarray(x) for k, x in m.items() if k != "executed"})
    metrics = {k: np.concatenate([x[k] for x in rows]) for k in rows[0]}
    dump(
        f"validation_{step:03d}_{'B' if cut else 'A'}.json",
        {
            "step": step,
            "protocol": protocol,
            "model_tree": tree_hash(params),
            "validation_dataset_sha256": json.loads((v.OUT / "dataset_identity.json").read_text())[
                "ordered_sha256"
            ]["validation"],
            "case": "B" if cut else "A",
            "summary": summary(metrics),
            "per_example": metrics,
            "global_example_id": data["global_example_id"],
        },
    )
    return metrics


def batch_for(data, update):
    epoch, index = divmod(update, 57)
    permutation = jax.random.permutation(jax.random.fold_in(jax.random.key(1160), epoch), 1824)
    return v.take(data, permutation[index * 32 : (index + 1) * 32])


def flow(params, codec, data, stage):
    ids = jax.random.permutation(jax.random.key(1143), len(data["q"]))[:64]
    # Original task dictionary only; cached loss tensors must not enter controls.
    raw = {k: x for k, x in data.items() if k not in ("source_h", "target_h", "slots", "slot_mask")}
    selected = v.take(raw, ids)
    det = torx_means_to_deterministic(params)
    run = jax.jit(v.forward, static_argnames=("no_attention", "torx"))
    encode = jax.jit(v.encode_fn)
    decode = jax.jit(v.decoder)
    checks = v.flow_check(det, codec, selected, encode, decode, run)
    extended = v.extended_checks(
        {"fresh_F1_G01": det}, params, codec, selected, encode, decode, run
    )
    dump(
        f"{stage}_flow.json",
        {
            "actual_model_tree": tree_hash(params),
            "legacy_helper_alias": "fresh_F1_G01 (not a claim about trained weights)",
            "selected_ids": selected["global_example_id"],
            "flow": checks,
            "extended": extended,
        },
    )
    if not all(checks["checks"].values()) or not extended["passed"]:
        raise RuntimeError("new-task flow gate failed")


def preflight(params, codec, identity, train, val):
    path = OUT / "preflight.json"
    if path.exists():
        saved = json.loads(path.read_text())
        if saved["protocol"] != identity or not saved["passed"]:
            raise RuntimeError("bad preflight record")
        return
    b = batch_for(train, 0)
    moments = (adamw_init(params), adamw_init(params))
    gradfn = jax.jit(
        jax.value_and_grad(lambda p: teacher_objective(p, b, 0, config=CFG), has_aux=True)
    )
    log("new-task full-model first gradient")
    (loss, metrics), grads = gradfn(params)
    jax.block_until_ready(loss)
    if not finite((loss, metrics, grads)) or not bool(metrics["executed"]):
        raise RuntimeError("first gradient failed")

    def norm(t):
        return float(jnp.sqrt(sum(jnp.sum(x * x) for x in jax.tree.leaves(t))))

    first = {
        "loss": float(loss),
        "raw_gradient_norm": norm(grads),
        "subsystems": {k: norm(g) for k, g in grads.items()},
        "metrics": summary(metrics),
    }
    dump("first_gradient.json", first)
    del gradfn, grads
    stepfn = jax.jit(lambda p, m, s, b: train_step(p, m, s, b, config=CFG))
    log("discarded full-model prospective AdamW update")
    p, m, s, metric = stepfn(params, moments, jnp.asarray(0), b)
    jax.block_until_ready(s)
    frozen = frozen_hash(params)
    if (
        int(s) != 1
        or not finite((p, m, metric))
        or frozen_hash(p) != frozen
        or tree_hash(p) == INIT
    ):
        raise RuntimeError("prospective update/frozen gate")
    if tree_hash(p["decoder"]) == tree_hash(params["decoder"]):
        raise RuntimeError("decoder did not update")
    mask = stochastic_model_update_mask(params)
    if any(
        not np.array_equal(
            np.asarray(a)[~np.broadcast_to(np.asarray(e), np.shape(a))],
            np.asarray(z)[~np.broadcast_to(np.asarray(e), np.shape(z))],
        )
        for a, z, e in zip(
            jax.tree.leaves(m), jax.tree.leaves(moments), jax.tree.leaves((mask, mask)), strict=True
        )
    ):
        raise RuntimeError("frozen moment update")
    trial = Journal(RUN / "discarded_trial", identity)
    if trial.recover() is None:
        trial.append(
            {"params": params, "moments": moments, "step": 0, "protocol": identity},
            {"counted": False},
        )
        trial.append(
            {"params": p, "moments": m, "step": 1, "protocol": identity}, {"counted": False}
        )
    reloaded = trial.recover()
    assert reloaded is not None
    if tree_hash((reloaded["params"], reloaded["moments"])) != tree_hash((p, m)):
        raise RuntimeError("persistence roundtrip")
    bad = {**b, "boundary": b["boundary"].at[:, -1].set(0)}
    rejected = stepfn(params, moments, jnp.asarray(0), bad)
    if (
        int(rejected[2]) != 0
        or tree_hash(rejected[:2]) != tree_hash((params, moments))
        or bool(rejected[3]["executed"])
    ):
        raise RuntimeError("invalid structure advanced optimizer")
    dump(
        "prospective_update.json",
        {
            "counted": False,
            "step": int(s),
            "initial_tree": INIT,
            "trial_tree": tree_hash(p),
            "moments": tree_hash(m),
            "frozen_tree": frozen,
            "rejected_update_unchanged": True,
            "persistence_roundtrip": True,
            "metrics": summary(metric),
        },
    )
    log("new-task negative-control, reachability, eager/JIT and Torx parity gates")
    flow(params, codec, val, "preflight")
    dump(
        "preflight.json",
        {
            "protocol": identity,
            "passed": True,
            "counted_updates": 0,
            "frozen_tree": frozen,
            "external_codec_sha256": sha(v.CODEC),
        },
    )


def train_run(params, codec, identity, train, val):
    preflight(params, codec, identity, train, val)
    journal = Journal(RUN / "official", identity)
    state = journal.recover()
    if state is None:
        state = {
            "params": params,
            "moments": (adamw_init(params), adamw_init(params)),
            "step": 0,
            "protocol": identity,
        }
        journal.append(state, {"initialization": True})
    frozen = frozen_hash(params)
    frozen_state = frozen_hash(params, (adamw_init(params), adamw_init(params)))
    stepfn = jax.jit(lambda p, m, s, b: train_step(p, m, s, b, config=CFG))
    for current in range(state["step"], 101):
        if current in CHECKPOINTS:
            for cut in (False, True):
                path = OUT / f"validation_{current:03d}_{'B' if cut else 'A'}.json"
                if not path.exists():
                    log(f"validation {current}, {'B' if cut else 'A'}")
                    evaluate(state["params"], val, current, cut, protocol=identity)
                record = json.loads(path.read_text())
                if (
                    record["protocol"] != identity
                    or record["model_tree"] != tree_hash(state["params"])
                    or record["step"] != current
                ):
                    raise RuntimeError("stale/mixed validation evidence")
            receipt = json.loads((journal.root / "receipts" / f"{current:03d}.json").read_text())
            dump(f"checkpoint_{current:03d}.json", receipt)
        if current == 100:
            break
        started = time.monotonic()
        p, m, s, metrics = stepfn(
            state["params"], state["moments"], jnp.asarray(current), batch_for(train, current)
        )
        jax.block_until_ready(s)
        if (
            int(s) != current + 1
            or not bool(metrics["executed"])
            or not finite((p, m, metrics))
            or frozen_hash(p) != frozen
            or frozen_hash(p, m) != frozen_state
        ):
            raise RuntimeError(f"training failed at attempted update {current + 1}")
        state = {"params": p, "moments": m, "step": int(s), "protocol": identity}
        journal.append(state, summary(metrics))
        log(
            f"completed update {int(s)}/100 loss={float(metrics['loss']):.6f} query_acc={float(jnp.mean(metrics['query_accuracy'])):.4f} seconds={time.monotonic() - started:.1f}"
        )
    return state


def endpoint(state, codec, identity, val):
    if state["step"] != 100:
        raise RuntimeError("fixed endpoint unavailable")
    params = state["params"]
    a = json.loads((OUT / "validation_100_A.json").read_text())
    b = json.loads((OUT / "validation_100_B.json").read_text())
    baseline = json.loads((OUT / "validation_000_A.json").read_text())
    for record in (a, b):
        if (
            record["protocol"] != identity
            or record["model_tree"] != tree_hash(params)
            or record["global_example_id"] != np.asarray(val["global_example_id"]).tolist()
        ):
            raise RuntimeError("endpoint comparison provenance/pairing mismatch")
    if baseline["protocol"] != identity or baseline["model_tree"] != INIT:
        raise RuntimeError("baseline provenance mismatch")
    am = {k: np.asarray(x) for k, x in a["per_example"].items()}
    bm = {k: np.asarray(x) for k, x in b["per_example"].items()}
    lengths = np.asarray(val["length"].sum(1))
    accuracy = paired_bootstrap(am["query_accuracy"], bm["query_accuracy"], lengths, seed=1161)
    nll = paired_bootstrap(am["query_nll"], bm["query_nll"], lengths, seed=1162)
    guard = nonquery_guard(a["summary"], baseline["summary"])
    checks = {
        "accuracy_effect": accuracy["delta"] >= 0.10,
        "nll_effect": a["summary"]["query_nll"] <= 0.9 * b["summary"]["query_nll"],
        "accuracy_ci": accuracy["delta_ci95"][0] > 0,
        "nll_ci": nll["delta_ci95"][1] < 0,
        "above_chance": a["summary"]["query_accuracy"] >= 1 / 28 + 0.10,
        "nonquery_accuracy": guard["accuracy_pass"],
        "nonquery_nll": guard["nll_pass"],
    }
    log("endpoint trained-weight exact negative controls")
    flow(params, codec, val, "endpoint")
    det = torx_means_to_deterministic(params)
    encode = jax.jit(v.encode_fn)
    decode = jax.jit(v.decoder)
    run = jax.jit(v.forward, static_argnames=("no_attention", "torx"))
    nuisance = jax.random.randint(jax.random.key(1145), (912, 16, 3), 5, 33)
    controls = {}
    for case in ("C", "D_substitute", "D_outside", "E_model", "E_codec"):
        metrics = []
        for start in range(0, 912, 32):
            ids = jnp.arange(start, min(start + 32, 912))
            d = v.take(val, ids)
            source = (
                d["source"]
                if case.startswith("E_")
                else observation_control(d, case, nuisance[ids])
            )
            h = encode(codec, source, d["source_mask"])
            logits = (
                decode(h, d["length"], codec["decoder"] if case == "E_codec" else det["decoder"])
                if case.startswith("E_")
                else run(det, h, d["prompt"], d["boundary"], d["length"])["logits"]
            )
            qlog = v.query(logits, d["q"])
            label = d["slots"][jnp.arange(len(ids)), d["q"], 1:]
            metrics.append(
                {
                    "query_accuracy": np.asarray((jnp.argmax(qlog, -1) == label).mean(1)),
                    "query_nll": np.asarray(ce(qlog, label).mean(1)),
                }
            )
        result = {k: np.concatenate([x[k] for x in metrics]) for k in metrics[0]}
        code = {"C": 1, "D_substitute": 2, "D_outside": 4, "E_model": 3, "E_codec": 3}[case]
        controls[case] = {
            "summary": summary(result),
            "per_example": result,
            "accuracy_pair": paired_bootstrap(
                am["query_accuracy"], result["query_accuracy"], lengths, seed=1161, code=code
            ),
            "nll_pair": paired_bootstrap(
                am["query_nll"], result["query_nll"], lengths, seed=1162, code=code
            ),
        }
    dump("endpoint_controls.json", controls)
    gap = np.asarray(val["peer"] - val["q"])
    q = np.asarray(val["q"])
    peer = np.asarray(val["peer"])
    outside = (
        (np.asarray(val["boundary"]) == 1)
        & (np.asarray(val["length"]) > 0)
        & (np.arange(32)[None, :] != peer[:, None])
    )
    groups = {
        "gap": gap,
        "length": lengths,
        "outside_competitor": outside.any(1).astype(int),
        "outside_full_competitor": (outside & (np.asarray(val["length"]) == 4)).any(1).astype(int),
        "fixed_neutral_reachable": (q // 8 == peer // 8).astype(int),
        "final_partial": (lengths % 4 != 0).astype(int),
    }
    strata = {
        name: {
            str(value): {
                "support": int(np.sum(index == value)),
                "A": summary({k: x[index == value] for k, x in am.items()}),
                "B": summary({k: x[index == value] for k, x in bm.items()}),
            }
            for value in np.unique(index)
        }
        for name, index in groups.items()
    }
    dump("endpoint_strata.json", {"strata": strata, "partial_query_support": 0})
    dump(
        "endpoint_decision.json",
        {
            "protocol": identity,
            "step": 100,
            "model_tree": tree_hash(params),
            "accuracy": accuracy,
            "nll": nll,
            "nonquery_guard": guard,
            "checks": checks,
            "bounded_necessity_pass": all(checks.values()),
            "interpretation": "bounded trained future-peer use under teacher structure only"
            if all(checks.values())
            else "healthy bounded endpoint nonpass/unresolved; no extension authorized",
            "protected_test_exposed": False,
        },
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage", choices=("audit", "preflight", "train", "endpoint"), required=True
    )
    args = parser.parse_args()
    params, codec, identity = audit()
    log(f"protocol={identity}; source/parents/history verified")
    if args.stage == "audit":
        return
    train, val = prepare(codec)
    if args.stage == "preflight":
        preflight(params, codec, identity, train, val)
    elif args.stage == "train":
        train_run(params, codec, identity, train, val)
    else:
        state = Journal(RUN / "official", identity).recover()
        if state is None:
            raise RuntimeError("no official state")
        endpoint(state, codec, identity, val)


if __name__ == "__main__":
    main()
