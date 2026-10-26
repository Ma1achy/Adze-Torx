"""Prospective G.0.1 frozen length probes, minimal codec repair and evidence."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import pickle
import platform
import subprocess

import jax
import jax.numpy as jnp
import numpy as np

from run_phase_g0 import chunks, data, digest, load, metrics, save, tree_hash, write
from adze_t.backends.torx import TorxOperatorConfig, TorxOps
from adze_t.config import REFERENCE_SMALL_V0 as CONFIG
from adze_t.denoise import as_sanitized_model_target_analysis, make_sanitized_target_analysis
from adze_t.model import apply_model
from adze_t.objectives import adamw_init
from adze_t.phase_f_1 import make_initial_corruption
from adze_t.phase_g import generated_byte_metrics, neutral_structure
from adze_t.phase_g0 import CAPACITY_POLICY, with_frozen_g0_codec
from adze_t.phase_g01 import (
    LEVELS,
    VERSION,
    candidate_step,
    decode_inferred_extents,
    extent_report,
    initialize_probe,
    probe_apply,
    probe_step,
    structurally_acceptable,
)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/phase_g/g0_1"
RUN = ROOT / "results/runs/phase_g/g0_1/v1"
PARENT = ROOT / "results/runs/phase_g/g0/v1/frozen_codec.pkl"
PARENT_HASH = "0f4a54e221729b14c6e75d30bed84bd566182a490f975c76d574e540f936adab"
F1 = ROOT / "results/runs/phase_f/f1/denoise_v1/one_step/init0_stoch0.pkl"
F1_HASH = "520d3723d3a3187aa08ff3f467f434d43b579cdc4b0ce667727dfad96e83cf57"
CHECKPOINTS = (0, 100, 250, 500, 1000, 2000, 5000)


def manifest():
    if digest(PARENT) != PARENT_HASH or digest(F1) != F1_HASH:
        raise RuntimeError("accepted parent identity mismatch")
    if subprocess.check_output(["git", "diff", "--name-only", "HEAD"], cwd=ROOT, text=True).strip():
        raise RuntimeError("commit G.0.1 source before execution")
    return {
        "version": VERSION,
        "source": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "parent_codec_sha256": PARENT_HASH,
        "f1_sha256": F1_HASH,
        "configuration": asdict(CONFIG),
        "capacity_policy": CAPACITY_POLICY,
        "protocol_file_sha256": digest(OUT / "PROTOCOL.md"),
        "data": {
            "probe_train": [128, 1010],
            "validation": [32, 1011],
            "candidate_train": [256, 1012],
            "test": [64, 1013],
        },
        "probe_steps": 800,
        "candidate_checkpoints": list(CHECKPOINTS),
        "runtime": {
            "python": platform.python_version(),
            "jax": jax.__version__,
            "backend": jax.default_backend(),
        },
    }


def features(params, count, seed):
    values, masks = data(count, seed)
    rows = chunks(params, values, masks)
    h = jnp.asarray(rows["h0"])
    noise_root = jax.random.fold_in(jax.random.key(1014), seed)
    stages = {"clean": np.asarray(h)}
    for nu in LEVELS:
        noisy, _ = make_initial_corruption(
            h,
            jnp.full((len(h),), nu),
            noise_root,
            jnp.arange(len(h), dtype=jnp.uint32),
            optimizer_step=0,
        )
        stages[str(nu)] = np.asarray(noisy)
    return rows, stages


def normalization(train, valid):
    mean = train.mean((0, 1), keepdims=True)
    scale = np.maximum(train.std((0, 1), keepdims=True), 1e-4)
    local_train, local_valid = (train - mean) / scale, (valid - mean) / scale
    flat = train.reshape(len(train), -1)
    mean_g = flat.mean(0, keepdims=True)
    scale_g = np.maximum(flat.std(0, keepdims=True), 1e-4)
    return tuple(
        map(
            jnp.asarray,
            (
                local_train,
                local_valid,
                (flat - mean_g) / scale_g,
                (valid.reshape(len(valid), -1) - mean_g) / scale_g,
            ),
        )
    ), {"local_mean": mean, "local_scale": scale, "global_mean": mean_g, "global_scale": scale_g}


def fit_probe(train, valid, truth_train, truth_valid, mlp, index):
    (lt, lv, gt, gv), norm = normalization(train, valid)
    params = initialize_probe(
        jax.random.fold_in(jax.random.key(1015), index), lt.shape[-1], gt.shape[-1], mlp
    )
    moments = (adamw_init(params), adamw_init(params))
    update = jax.jit(probe_step, static_argnames=("mlp",))
    labels = jnp.asarray(truth_train)
    for step in range(1, 801):
        ids = jax.random.randint(
            jax.random.fold_in(jax.random.key(1017 + index), step), (128,), 0, len(train)
        )
        params, moments, loss, grad_norm = update(
            params, moments, step, lt[ids], gt[ids], labels[ids], mlp
        )
        if not bool(jnp.isfinite(loss) & jnp.isfinite(grad_norm)):
            raise RuntimeError("nonfinite diagnostic probe")
    local, glob = probe_apply(params, lv, gv, mlp)
    predictions = np.asarray(jnp.argmax(local, -1))
    direct = np.column_stack(
        [np.asarray(jnp.argmax(glob[:, a:b], -1)) for a, b in ((0, 57), (57, 74), (74, 78))]
    )
    save(
        RUN / "probes" / f"probe_{index:02d}.pkl",
        {"params": params, "normalization": norm, "mlp": mlp},
    )
    return extent_report(predictions, truth_valid, direct)


def real_g_input_control(params, stages, truth):
    installed = with_frozen_g0_codec(load(F1)["params"], params)
    ids = np.asarray([n * 32 + j for n in range(57) for j in range(4)])
    neutral = neutral_structure(16, CONFIG)
    if neutral.length is None:
        raise AssertionError("G predictor control requires explicit neutral lengths")
    sanitized = make_sanitized_target_analysis(
        neutral.c_b, neutral.length, config=CONFIG, target_width=64
    )
    target = as_sanitized_model_target_analysis(sanitized)
    ops = TorxOps.create(
        jax.random.key(1016), config=TorxOperatorConfig(operator_stochasticity=True, lambda_op=0.0)
    )

    @jax.jit
    def apply(h, nu):
        out = apply_model(
            installed,
            jnp.full((16, 1), 0xF1, jnp.int32),
            jnp.ones((16, 1), bool),
            sanitized.target_bytes_placeholder,
            sanitized.target_mask_placeholder,
            config=CONFIG,
            ops=ops,
            target_analysis=target,
            carrier_h_input=h,
            noise_level=nu,
            actual_s_index=0,
            denoise_condition_index=0,
            committed_c_b=neutral.c_b,
            committed_length=neutral.length,
            committed_activity=neutral.activity,
            observed_b=neutral.c_b,
            observed_l=neutral.length,
        )
        return out["current_carrier_input"], out["prediction"][2]

    reports = {}
    for nu in LEVELS:
        source = stages[str(nu)][ids]
        predictions = []
        for start in range(0, len(source), 16):
            h = jnp.asarray(source[start : start + 16])
            n = len(h)
            h = jnp.pad(h, ((0, 16 - n), (0, 0), (0, 0)))
            observed, logits = apply(h, jnp.full((16,), nu))
            if not bool(jnp.array_equal(h, observed)):
                raise RuntimeError("actual G input differs from declared h_nu")
            predictions.append(np.asarray(jnp.argmax(logits[:n], -1)))
        reports[str(nu)] = {
            "input_bitwise_equal_to_corrupted_h": True,
            "examples": len(ids),
            "existing_G_prediction_head": extent_report(np.concatenate(predictions), truth[ids]),
            "input_protocol": "constant prompt, neutral observed/routing structure, sanitized inert targets, explicit h_nu",
        }
    return reports


def diagnose():
    protocol = manifest()
    if (OUT / "diagnosis.json").exists():
        raise RuntimeError("preserve existing G.0.1 diagnosis")
    params = load(PARENT)
    train, train_stages = features(params, 128, 1010)
    val, val_stages = features(params, 32, 1011)
    truth_train, truth_val = train["length"], val["length"]
    local_prior = np.stack(
        [np.bincount(truth_train[:, c], minlength=5).argmax() for c in range(32)]
    )
    majority = np.bincount(truth_train.reshape(-1), minlength=5).argmax()
    baselines = {
        "global_majority": extent_report(np.full_like(truth_val, majority), truth_val),
        "position_length_prior": extent_report(
            np.broadcast_to(local_prior, truth_val.shape), truth_val
        ),
        "teacher_upper_bound": extent_report(truth_val, truth_val),
    }
    from adze_t.phase_g01 import length_labels

    global_prior = np.asarray(length_labels(jnp.asarray(truth_train)))
    target_global = np.asarray(length_labels(jnp.asarray(truth_val)))
    baselines["global_majority_length_count_partial"] = {
        name: float((target_global[:, i] == np.bincount(global_prior[:, i]).argmax()).mean())
        for i, name in enumerate(("total_length", "emitting_count", "partial_class"))
    }
    reports = {}
    for stage_index, (name, features_train) in enumerate(train_stages.items()):
        features_val = val_stages[name]
        p = params["encoder"]["target_l"]
        existing = np.asarray(jnp.argmax(jnp.asarray(features_val) @ p["weight"] + p["bias"], -1))
        report = {"existing_codec_head": extent_report(existing, truth_val)}
        for kind_index, mlp in enumerate((False, True)):
            kind = "small_mlp" if mlp else "production_linear"
            report[kind] = fit_probe(
                features_train,
                features_val,
                truth_train,
                truth_val,
                mlp,
                stage_index * 2 + kind_index,
            )
            print(
                json.dumps(
                    {
                        "stage": name,
                        "probe": kind,
                        "exact_length": report[kind]["exact_total_length_accuracy"],
                        "partial": report[kind]["partial_extent_accuracy"],
                    }
                ),
                flush=True,
            )
        reports[name] = report
    actual = real_g_input_control(params, val_stages, truth_val)
    readout_only = structurally_acceptable(reports["clean"]["production_linear"]) and all(
        structurally_acceptable(reports[str(nu)]["production_linear"], noisy=True) for nu in LEVELS
    )
    result = {
        "protocol": protocol,
        "baselines": baselines,
        "stages": reports,
        "actual_G_input": {
            str(nu): {
                "same_information_as": str(nu),
                "probe_results_reference": f"stages/{nu}",
                **actual[str(nu)],
            }
            for nu in LEVELS
        },
        "candidate_mode": "readout_only" if readout_only else "joint_codec",
        "interpretation": "finite probe outcomes establish achieved recoverability, not proof of irreversible information loss",
    }
    write(OUT / "diagnosis.json", result)


def evaluate(params, count, seed):
    rows, stages = features(params, count, seed)
    structural = extent_report(rows["predicted_length"], rows["length"])
    boundary = rows["direct_boundary"].copy()
    boundary[:, -1] = 1
    structural["exact_complete_structure_accuracy"] = float(
        (
            (boundary == rows["boundary"]).all(1)
            & (rows["predicted_length"] == rows["length"]).all(1)
        ).mean()
    )
    known = metrics(rows)

    @jax.jit
    def own(h, truth_bytes, truth_mask):
        inferred, logits, _ = decode_inferred_extents(
            h, params["encoder"]["target_l"], params["decoder"]
        )
        return generated_byte_metrics(logits, inferred, truth_bytes, truth_mask)

    measurements = []
    for start in range(0, len(rows["h0"]), 16):
        measurements.append(
            own(
                jnp.asarray(rows["h0"][start : start + 16]),
                jnp.asarray(rows["bytes"][start : start + 16]),
                jnp.asarray(rows["mask"][start : start + 16]),
            )
        )
    combined = jax.tree.map(lambda *x: np.concatenate([np.asarray(a) for a in x]), *measurements)
    lengths = rows["length"].sum(1)
    own_result = {
        "byte_accuracy": float(combined.accuracy_per_example.mean()),
        "exact_sequence_accuracy": float(combined.exact_per_example.mean()),
        "length_aware_reconstruction_loss": float(combined.loss_per_example.mean()),
        "overlap_only_byte_nll": float(combined.overlap_nll_per_example.mean()),
        "missing_bytes": int(combined.missing_count.sum()),
        "extra_bytes": int(combined.extra_count.sum()),
        "by_length": [
            {
                "length": n,
                "byte_accuracy": float(combined.accuracy_per_example[lengths == n].mean()),
                "exact_sequence_accuracy": float(combined.exact_per_example[lengths == n].mean()),
                "missing_bytes": int(combined.missing_count[lengths == n].sum()),
                "extra_bytes": int(combined.extra_count[lengths == n].sum()),
            }
            for n in range(8, 65)
        ],
    }
    noisy = {}
    p = params["encoder"]["target_l"]
    for nu in LEVELS:
        predicted = np.asarray(
            jnp.argmax(jnp.asarray(stages[str(nu)]) @ p["weight"] + p["bias"], -1)
        )
        noisy[str(nu)] = extent_report(predicted, rows["length"])
    clean_pass = (
        known["passed"]
        and structurally_acceptable(structural)
        and structural["exact_complete_structure_accuracy"] >= 0.99
        and own_result["byte_accuracy"] >= 0.99
        and min(x["byte_accuracy"] for x in own_result["by_length"]) >= 0.98
    )
    input_ready = all(structurally_acceptable(noisy[str(nu)], noisy=True) for nu in LEVELS)
    return {
        "clean_pass": clean_pass,
        "G_input_ready": input_ready,
        "known_extent_reconstruction": known,
        "inferred_extent_reconstruction": own_result,
        "structure": structural,
        "corrupted_readout": noisy,
    }


def train():
    protocol = manifest()
    diagnostic = json.loads((OUT / "diagnosis.json").read_text())
    if diagnostic["protocol"] != json.loads(json.dumps(protocol)):
        raise RuntimeError("diagnosis/candidate source mismatch")
    if (OUT / "selection.json").exists():
        raise RuntimeError("selection already frozen; do not reselect/retrain after test")
    readout_only = diagnostic["candidate_mode"] == "readout_only"
    state_path = RUN / "state.pkl"
    start = 0
    if state_path.exists():
        with state_path.open("rb") as f:
            state = pickle.load(f)  # noqa: S301
        if state["protocol"] != protocol:
            raise RuntimeError("resume protocol mismatch")
        params, moments = jax.tree.map(jnp.asarray, (state["params"], state["moments"]))
        start = state["step"]
    else:
        params = load(PARENT)
        moments = (adamw_init(params), adamw_init(params))
    values, masks = data(256, 1012)
    update = jax.jit(candidate_step, static_argnames=("readout_only",))
    for step in range(start, 5001):
        if state_path.exists() and step == start:
            continue
        if step > 0:
            key = jax.random.fold_in(jax.random.key(1018), step)
            batch_key, noise_key = jax.random.split(key)
            ids = jax.random.randint(batch_key, (32,), 0, len(values))
            params, moments, result = update(
                params, moments, step, values[ids], masks[ids], noise_key, readout_only
            )
            if not bool(result["finite"]) or not all(
                bool(jnp.all(jnp.isfinite(x))) for x in jax.tree.leaves((params, moments))
            ):
                raise RuntimeError(f"nonfinite G.0.1 update {step}; last durable state retained")
        if step in CHECKPOINTS:
            score = evaluate(params, 32, 1011)
            path = RUN / f"codec_{step:05d}.pkl"
            if path.exists():
                raise RuntimeError("refusing checkpoint overwrite")
            state = {
                "params": params,
                "moments": moments,
                "step": step,
                "protocol": protocol,
                "readout_only": readout_only,
            }
            save(path, state)
            record = {
                "protocol": protocol,
                "step": step,
                "checkpoint": str(path.relative_to(ROOT)),
                "sha256": digest(path),
                "evaluation": score,
            }
            write(OUT / f"validation_{step:05d}.json", record)
            save(state_path, state)
            print(
                json.dumps(
                    {
                        "step": step,
                        "clean_pass": score["clean_pass"],
                        "G_input_ready": score["G_input_ready"],
                        "exact_length": score["structure"]["exact_total_length_accuracy"],
                        "partial": score["structure"]["partial_extent_accuracy"],
                        "byte": score["known_extent_reconstruction"]["byte_accuracy"],
                    }
                ),
                flush=True,
            )
            if score["clean_pass"] and score["G_input_ready"]:
                write(
                    OUT / "selection.json",
                    {
                        **record,
                        "rule": "first scheduled clean AND noisy-input acceptance pass, before test",
                    },
                )
                heldout = evaluate(params, 64, 1013)
                write(
                    OUT / "heldout.json",
                    {
                        "selection_sha256": digest(OUT / "selection.json"),
                        "protocol": protocol,
                        "evaluation": heldout,
                    },
                )
                if heldout["clean_pass"] and heldout["G_input_ready"]:
                    frozen = RUN / "frozen_codec.pkl"
                    save(frozen, params)
                    frozen.chmod(0o444)
                    write(
                        OUT / "frozen_codec.json",
                        {
                            "version": VERSION,
                            "path": str(frozen.relative_to(ROOT)),
                            "sha256": digest(frozen),
                            "parameter_tree_sha256": tree_hash(params),
                            "selection_sha256": digest(OUT / "selection.json"),
                            "G_input_ready": heldout["G_input_ready"],
                        },
                    )
                return
    write(
        OUT / "endpoint.json",
        {"protocol": protocol, "status": "G01_BUDGET_EXHAUSTED_NO_CLEAN_PASS", "step": 5000},
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("diagnose", "train"), required=True)
    args = parser.parse_args()
    if args.stage == "diagnose":
        diagnose()
    else:
        train()
