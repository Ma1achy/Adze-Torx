"""G.0 representation diagnosis and separately versioned codec remediation.

No G policy-arm training entry point exists here. All corpus randomness is JAX.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import pickle
import platform
import subprocess
import time
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from adze_t.config import REFERENCE_SMALL_V0 as CONFIG
from adze_t.model import apply_target_codec
from adze_t.objectives import adamw_init
from adze_t.phase_g import generated_structure_truth
from adze_t.phase_g0 import CAPACITY_POLICY, CODEC_VERSION, codec_subtree, codec_update

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/phase_g/g0"
RUN = ROOT / "results/runs/phase_g/g0/v1"
PARENT = ROOT / "results/phase_b/checkpoints/target_codec_b1.pkl"
F1 = ROOT / "results/runs/phase_f/f1/denoise_v1/one_step/init0_stoch0.pkl"
EXPECTED_PARENT = "55f057f78f795f1585d7aac28fa7ff37846d4c1098507ede375e86b21c4f8ff6"
EXPECTED_F1 = "520d3723d3a3187aa08ff3f467f434d43b579cdc4b0ce667727dfad96e83cf57"
MAX_UPDATES = 5000
CHECKPOINTS = (0, 100, 250, 500, 1000, 2000, 3000, 4000, 5000)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        pickle.dump(jax.device_get(value), stream, protocol=pickle.HIGHEST_PROTOCOL)
    temporary.replace(path)


def load(path: Path) -> Any:
    with path.open("rb") as stream:
        return jax.tree.map(jnp.asarray, pickle.load(stream))  # noqa: S301


def data(per_length: int, seed: int) -> tuple[jax.Array, jax.Array]:
    lengths = jnp.repeat(jnp.arange(8, 65), per_length)
    values = jax.random.randint(jax.random.key(seed), (len(lengths), 64), 1, 33)
    mask = jnp.arange(64)[None, :] < lengths[:, None]
    return jnp.where(mask, values, 0), mask


def protocol() -> dict[str, Any]:
    if digest(PARENT) != EXPECTED_PARENT or digest(F1) != EXPECTED_F1:
        raise RuntimeError("accepted codec/F1 hash mismatch")
    sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    if subprocess.check_output(["git", "diff", "--name-only", "HEAD"], cwd=ROOT, text=True).strip():
        raise RuntimeError("G.0 source must be committed before execution")
    return {
        "version": CODEC_VERSION,
        "source_commit": sha,
        "parent_codec_sha256": EXPECTED_PARENT,
        "f1_sha256": EXPECTED_F1,
        "configuration": asdict(CONFIG),
        "capacity_policy": CAPACITY_POLICY,
        "data": {
            "probe_train_per_length": 64,
            "validation_per_length": 16,
            "test_per_length": 32,
            "seeds": [960, 961, 962],
            "width": 64,
            "lengths": [8, 64],
            "byte_domain": [1, 32],
            "training_per_length": 256,
            "training_seed": 963,
        },
        "probe": {
            "family": "per-carrier ridge one-hot heads",
            "ridge": 0.01,
            "inputs": "h0 and fixed carrier/slot identity only",
            "decision": "decoder_only only if every validation probe gate passes",
        },
        "optimizer": {
            "name": "AdamW",
            "lr": 3e-4,
            "weight_decay": 0.01,
            "clip": 1.0,
            "batch_size": 32,
            "fresh_moments": True,
        },
        "loss": {"byte_ce": 1, "content_boundary_ce": 1, "extent_ce": 1},
        "max_updates": MAX_UPDATES,
        "checkpoints": CHECKPOINTS,
        "selection": "first scheduled validation checkpoint passing all gates; test once",
        "thresholds": {
            "aggregate_byte": 0.99,
            "every_length_byte": 0.98,
            "signature": 0.99,
            "numerical_capacity_failures": 0,
        },
        "runtime": {
            "python": platform.python_version(),
            "jax": jax.__version__,
            "backend": jax.default_backend(),
            "devices": list(map(str, jax.devices())),
        },
    }


def oracle() -> None:
    from adze_t.phase_g0_oracle import conformance

    manifest = protocol()
    rows = []
    for parameter in ("uniform", "asymmetric", "peaked"):
        for loss in (
            "inactive",
            "reactivation",
            "zero",
            "missing",
            "extra",
            "direct_dependencies",
            "baseline",
        ):
            row = conformance(parameter, loss)
            rows.append(row)
            print(
                json.dumps({"oracle": parameter, "loss": loss, "accepted": row["accepted"]}),
                flush=True,
            )
    write(
        OUT / "capacity_oracle.json",
        {
            "protocol": manifest,
            "rows": rows,
            "status": "G0_CAPACITY_ORACLE_PASS"
            if all(x["accepted"] for x in rows)
            else "G0_CAPACITY_ORACLE_FAIL",
        },
    )


def chunks(params: Any, values: jax.Array, masks: jax.Array) -> dict[str, np.ndarray]:
    evaluate = jax.jit(lambda p, x, m: apply_target_codec(p, x, m, config=CONFIG))
    rows: dict[str, list[np.ndarray]] = {}
    for start in range(0, len(values), 16):
        x, m = values[start : start + 16], masks[start : start + 16]
        # Pad the final batch to avoid recompilation, but never score padding.
        n = len(x)
        x = jnp.pad(x, ((0, 16 - n), (0, 0)))
        m = jnp.pad(m, ((0, 16 - n), (0, 0)))
        result = evaluate(params, x, m)
        target = result["target"]
        teacher = target["teacher"]
        logits = result["codec_logits"]
        predicted = jnp.argmax(logits, -1)
        predicted_length = jnp.argmax(target["l_logits"], -1)
        truth = generated_structure_truth(teacher.slot_bytes, teacher.length, K=CONFIG.packing.K)
        signature = generated_structure_truth(predicted, predicted_length, K=CONFIG.packing.K)
        logp = jax.nn.log_softmax(logits, -1)
        ce = -jnp.take_along_axis(logp, teacher.slot_bytes[..., None], -1)[..., 0]
        fields = {
            "h0": target["h0"],
            "bytes": teacher.slot_bytes,
            "mask": teacher.slot_mask,
            "length": teacher.length,
            "boundary": truth.c_b,
            "predicted": predicted,
            "predicted_length": predicted_length,
            "signature": signature.c_b,
            "direct_boundary": jnp.argmax(target["b_logits"], -1),
            "ce": ce,
            "finite": jnp.all(jnp.isfinite(logits), axis=(1, 2, 3))
            & jnp.all(jnp.isfinite(target["h0"]), axis=(1, 2)),
            "overflow": teacher.capacity_overflow,
        }
        for name, array in fields.items():
            rows.setdefault(name, []).append(np.asarray(array[:n]))
    return {name: np.concatenate(arrays) for name, arrays in rows.items()}


def metrics(rows: dict[str, np.ndarray]) -> dict[str, Any]:
    mask, truth = rows["mask"], rows["bytes"]
    correct = rows["predicted"] == truth
    lengths = mask.sum((1, 2))
    reports = []
    for n in range(8, 65):
        selected = lengths == n
        active_mask = mask[selected]
        partial = (rows["length"][selected] > 0) & (rows["length"][selected] < 4)
        partial_count = int(partial.sum())
        reports.append(
            {
                "length": n,
                "examples": int(selected.sum()),
                "byte_accuracy": float(correct[selected][active_mask].mean()),
                "byte_nll": float(rows["ce"][selected][active_mask].mean()),
                "exact_sequence_accuracy": float(
                    (correct[selected] | ~active_mask).all((1, 2)).mean()
                ),
                "signature_accuracy": float(
                    np.concatenate(
                        (
                            rows["signature"][selected, :-1] == rows["boundary"][selected, :-1],
                            rows["predicted_length"][selected] == rows["length"][selected],
                        ),
                        axis=1,
                    ).mean()
                ),
                "length_accuracy": float(
                    (rows["predicted_length"][selected] == rows["length"][selected]).mean()
                ),
                "exact_emitted_length_accuracy": float(
                    (rows["predicted_length"][selected].sum(1) == n).mean()
                ),
                "partial_carrier_support": partial_count,
                "partial_length_accuracy": None
                if not partial_count
                else float(
                    (
                        rows["predicted_length"][selected][partial]
                        == rows["length"][selected][partial]
                    ).mean()
                ),
                "partial_byte_accuracy": None
                if not partial_count
                else float(correct[selected][partial][active_mask[partial]].mean()),
            }
        )
    aggregate = float(correct[mask].mean())
    signature = float(np.mean([r["signature_accuracy"] for r in reports]))
    failures = int((~rows["finite"]).sum() + rows["overflow"].sum())
    return {
        "byte_accuracy": aggregate,
        "byte_nll": float(rows["ce"][mask].mean()),
        "exact_sequence_accuracy": float((correct | ~mask).all((1, 2)).mean()),
        "signature_accuracy": signature,
        "numerical_capacity_failures": failures,
        "passed": aggregate >= 0.99
        and min(r["byte_accuracy"] for r in reports) >= 0.98
        and signature >= 0.99
        and failures == 0,
        "by_length": reports,
    }


def probe_fit(train: dict[str, np.ndarray], valid: dict[str, np.ndarray]) -> dict[str, Any]:
    """Independent controlled linear probes; no true length enters features."""
    prediction = np.zeros_like(valid["bytes"])
    length_prediction = np.zeros_like(valid["length"])
    boundary_prediction = np.zeros_like(valid["boundary"])
    for c in range(CONFIG.carrier.C):
        x, y = train["h0"][:, c].astype(np.float64), valid["h0"][:, c].astype(np.float64)
        mean, std = x.mean(0), np.maximum(x.std(0), 1e-6)
        x = np.column_stack(((x - mean) / std, np.ones(len(x))))
        y = np.column_stack(((y - mean) / std, np.ones(len(y))))

        def fit(labels, classes, support):
            a = x[support]
            z = np.eye(classes)[labels[support]]
            weights = np.linalg.solve(a.T @ a + 0.01 * np.eye(a.shape[1]), a.T @ z)
            return np.argmax(y @ weights, axis=-1)

        for slot in range(4):
            prediction[:, c, slot] = fit(train["bytes"][:, c, slot], 256, train["mask"][:, c, slot])
        length_prediction[:, c] = fit(train["length"][:, c], 5, np.ones(len(x), bool))
        boundary_prediction[:, c] = fit(train["boundary"][:, c], 2, np.ones(len(x), bool))
    probe_rows = {**valid, "predicted": prediction, "predicted_length": length_prediction}
    probe_rows["signature"] = boundary_prediction
    result = metrics(probe_rows)
    # Probe CE is not probabilistically calibrated, so never misreport old CE as probe NLL.
    result.pop("byte_nll")
    for row in result["by_length"]:
        row.pop("byte_nll")
    result["interpretation"] = (
        "achievable linear recoverability, not a proof of information-theoretic sufficiency/loss"
    )
    return result


def diagnose() -> None:
    manifest = protocol()
    params = codec_subtree(load(PARENT))
    train = chunks(params, *data(64, 960))
    valid = chunks(params, *data(16, 961))
    baseline = metrics(valid)
    probes = probe_fit(train, valid)
    # Destroy h0 while retaining site identity: tests position/length-prior confounds.
    control_train = {**train, "h0": np.zeros_like(train["h0"])}
    control_valid = {**valid, "h0": np.zeros_like(valid["h0"])}
    controls = probe_fit(control_train, control_valid)
    decoder_only = probes["passed"] and all(
        r["length_accuracy"] >= 0.99
        and (r["partial_length_accuracy"] is None or r["partial_length_accuracy"] >= 0.99)
        for r in probes["by_length"]
    )
    result = {
        "protocol": manifest,
        "baseline": baseline,
        "h0_only_probes": probes,
        "destroyed_h0_control": controls,
        "selected_training": "decoder_only" if decoder_only else "joint_target_codec",
        "rationale": "decoder-only permitted only by all held-out byte/signature/extent probe gates; otherwise joint training is conservative, not proof of irrecoverable encoder loss",
    }
    write(OUT / "representation_audit.json", result)
    print(
        json.dumps(
            {
                "stage": "diagnose",
                "training": result["selected_training"],
                "baseline_byte": baseline["byte_accuracy"],
                "probe_byte": probes["byte_accuracy"],
                "probe_signature": probes["signature_accuracy"],
            }
        ),
        flush=True,
    )


def train() -> None:
    manifest = protocol()
    audit = json.loads((OUT / "representation_audit.json").read_text())
    if audit["protocol"] != json.loads(json.dumps(manifest)):
        raise RuntimeError("diagnostic/source protocol mismatch")
    decoder_only = audit["selected_training"] == "decoder_only"
    state_path = RUN / "state.pkl"
    if state_path.exists():
        with state_path.open("rb") as stream:
            state = pickle.load(stream)  # noqa: S301
        if state["protocol"] != manifest or state["decoder_only"] != decoder_only:
            raise RuntimeError("resume protocol mismatch")
        params, moments = jax.tree.map(jnp.asarray, (state["params"], state["moments"]))
        start = int(state["step"])
    else:
        params = codec_subtree(load(PARENT))
        moments = (adamw_init(params), adamw_init(params))
        start = 0
    values, masks = data(256, 963)
    evaluate_values, evaluate_masks = data(16, 961)
    update = jax.jit(codec_update, static_argnames=("decoder_only", "config"))
    started = time.monotonic()
    for step in range(start, MAX_UPDATES + 1):
        if step == start and state_path.exists():
            checkpoint = RUN / f"codec_{step:05d}.pkl"
            evidence = OUT / f"validation_{step:05d}.json"
            if not checkpoint.exists() or not evidence.exists():
                raise RuntimeError("resume lacks matching durable checkpoint/evidence")
            record = json.loads(evidence.read_text())
            if record["checkpoint_sha256"] != digest(checkpoint) or record[
                "protocol_sha256"
            ] != canonical(manifest):
                raise RuntimeError("resume checkpoint/evidence identity mismatch")
            if record["validation"]["passed"]:
                raise RuntimeError(
                    "validation-selected run must finish/reconcile test and freeze, not resume optimizer"
                )
            continue
        if step > start:
            indices = jax.random.randint(
                jax.random.fold_in(jax.random.key(964), step), (32,), 0, len(values)
            )
            params, moments, current = update(
                params,
                moments,
                jnp.asarray(step),
                values[indices],
                masks[indices],
                decoder_only=decoder_only,
                config=CONFIG,
            )
            if not bool(current["finite"]) or not all(
                bool(jnp.all(jnp.isfinite(x))) for x in jax.tree.leaves((params, moments))
            ):
                raise RuntimeError(f"nonfinite update {step}; last durable state preserved")
        if step in CHECKPOINTS:
            score = metrics(chunks(params, evaluate_values, evaluate_masks))
            state = {
                "protocol": manifest,
                "decoder_only": decoder_only,
                "step": step,
                "params": params,
                "moments": moments,
            }
            checkpoint = RUN / f"codec_{step:05d}.pkl"
            if checkpoint.exists():
                raise RuntimeError(
                    "refusing checkpoint overwrite; reconcile existing durable evidence first"
                )
            save(checkpoint, state)
            record = {
                "step": step,
                "checkpoint": str(checkpoint.relative_to(ROOT)),
                "checkpoint_sha256": digest(checkpoint),
                "protocol_sha256": canonical(manifest),
                "validation": score,
                "elapsed_seconds": time.monotonic() - started,
            }
            write(OUT / f"validation_{step:05d}.json", record)
            save(state_path, state)
            print(
                json.dumps(
                    {
                        "step": step,
                        "accuracy": score["byte_accuracy"],
                        "signature": score["signature_accuracy"],
                        "minimum_length": min(r["byte_accuracy"] for r in score["by_length"]),
                        "passed": score["passed"],
                    }
                ),
                flush=True,
            )
            if score["passed"]:
                write(OUT / "selection.json", {**record, "rule": manifest["selection"]})
                heldout = metrics(chunks(params, *data(32, 962)))
                write(
                    OUT / "heldout_compatibility.json",
                    {
                        "selection_sha256": digest(OUT / "selection.json"),
                        "protocol": manifest,
                        "test": heldout,
                        "status": "G0_CODEC_COMPATIBILITY_PASS"
                        if heldout["passed"]
                        else "G0_CODEC_COMPATIBILITY_BLOCKED",
                    },
                )
                if heldout["passed"]:
                    frozen = RUN / "frozen_codec.pkl"
                    if frozen.exists():
                        raise RuntimeError("frozen candidate already exists")
                    save(frozen, params)
                    write(
                        OUT / "frozen_codec.json",
                        {
                            "version": CODEC_VERSION,
                            "path": str(frozen.relative_to(ROOT)),
                            "sha256": digest(frozen),
                            "protocol_sha256": canonical(manifest),
                            "selection_sha256": digest(OUT / "selection.json"),
                            "read_only_after_freeze": True,
                        },
                    )
                return
    write(
        OUT / "training_endpoint.json",
        {
            "status": "G0_BUDGET_EXHAUSTED_NO_PASS",
            "updates": MAX_UPDATES,
            "protocol_sha256": canonical(manifest),
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("diagnose", "train", "oracle"), required=True)
    args = parser.parse_args()
    if args.stage == "diagnose":
        diagnose()
    elif args.stage == "train":
        train()
    else:
        oracle()
