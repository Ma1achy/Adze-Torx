"""G.0.1-bound engineering preflight; never launches a full G campaign."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from functools import partial
import hashlib
import json
import shutil
from pathlib import Path
import subprocess
import sys

import jax
import jax.numpy as jnp
import numpy as np

from run_phase_g import (
    config,
    dataset,
    with_nu,
    take_batch,
    stochastic_root,
    _control_structures,
    _bootstrap_interval,
    load_pickle,
    save_pickle,
    serialise,
    tree_all_finite,
    implementation_sha,
)
from run_phase_g0 import tree_hash
from adze_t.backends.deterministic import DeterministicOps
from adze_t.backends.mapping import torx_means_to_deterministic
from adze_t.backends.torx import TorxOperatorConfig, TorxOps
from adze_t.denoise import make_sanitized_target_analysis, as_sanitized_model_target_analysis
from adze_t.masking import draft_attention_mask
from adze_t.model import apply_clean_target_teacher, apply_model
from adze_t.objectives import adamw_init
from adze_t.packing import pack_values
from adze_t.phase_g import (
    generated_two_pass_objective,
    generated_train_step,
    neutral_structure,
    evaluate_committed_replay,
)
from adze_t.phase_g0 import CAPACITY_POLICY, dynamic_capacity_routing, with_frozen_g0_codec
from adze_t.phase_g0_oracle import conformance
from adze_t.phase_f_1 import make_initial_corruption
from adze_t.training import stochastic_model_update_mask
from adze_t.unpool import unpool_values

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/phase_g/preflight_g01_v1"
RUN = ROOT / "results/runs/phase_g/preflight_g01_v1"
CODEC = ROOT / "results/runs/phase_g/g0_1/v1/frozen_codec.pkl"
CODEC_HASH = "213d71be58b4e3a0eb685330ff8f20ad0d2574184d59060890baa4a2eed586e7"
F1 = ROOT / "results/runs/phase_f/f1/denoise_v1/one_step/init0_stoch0.pkl"
F1_HASH = "520d3723d3a3187aa08ff3f467f434d43b579cdc4b0ce667727dfad96e83cf57"
INITIAL_TREE = "84ab204d1cea2a1833cbbdeadc7993879ad29a3c4c354d4960c221c951daaeeb"
PROTOCOL = "PHASE_G_G01_DYNAMIC_STRUCTURE_ENGINEERING_PREFLIGHT_V1"
CHECKPOINTS = (0, 10, 25, 50, 100)
ARMS = ("teacher", "argmax", "sample")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def encoded(value):
    return json.loads(json.dumps(value, sort_keys=True, default=serialise, allow_nan=False))


def record(path, value):
    """Immutable atomic result: matching resumes are idempotent, never overwrite."""
    value = encoded(value)
    path = Path(path)
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise RuntimeError(f"refusing evidence overwrite: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def save_current(path, state):
    """Checkpoint payload then hash-bound pointer; mismatch fails closed on resume."""
    save_pickle(path, state)
    receipt = {
        "protocol": state["protocol"],
        "sha256": sha(path),
        "step": state["step"],
        "updates": state["updates"],
        "arm": state["arm"],
    }
    pointer = path.with_suffix(".receipt.json")
    temporary = pointer.with_suffix(pointer.suffix + ".tmp")
    temporary.write_text(json.dumps(encoded(receipt), sort_keys=True) + "\n")
    temporary.replace(pointer)


def load_current(path, protocol, arm):
    receipt = json.loads(path.with_suffix(".receipt.json").read_text())
    if receipt["protocol"] != protocol or receipt["sha256"] != sha(path) or receipt["arm"] != arm:
        raise RuntimeError("latest-state provenance/checksum mismatch")
    state = load_pickle(path)
    if state["step"] != receipt["step"] or state["updates"] != receipt["updates"]:
        raise RuntimeError("latest-state update identity mismatch")
    return state


def manifest():
    if sha(CODEC) != CODEC_HASH or sha(F1) != F1_HASH:
        raise RuntimeError("checkpoint provenance mismatch")
    meta = json.loads((ROOT / "results/phase_g/g0_1/frozen_codec.json").read_text())
    if meta["sha256"] != CODEC_HASH or meta["path"] != str(CODEC.relative_to(ROOT)):
        raise RuntimeError("codec metadata mismatch")
    return {
        "protocol": PROTOCOL,
        "implementation": implementation_sha(),
        "codec_sha256": CODEC_HASH,
        "f1_sha256": F1_HASH,
        "initial_tree_sha256": INITIAL_TREE,
        "configuration": asdict(config()),
        "capacity_policy": CAPACITY_POLICY,
        "commit": "all nonterminal semantic b; all ell; terminal forced; no fixed-position commitment masks",
        "data": {
            "train": [1024, 950],
            "validation": [256, 951],
            "stochastic_seed": 952,
            "global_id_ranges": [[0, 1023], [1024, 1279]],
            "nu": [0.025, 0.25],
        },
        "optimizer": {
            "name": "AdamW",
            "lr": 3e-4,
            "weight_decay": 0.01,
            "clip": 1.0,
            "batch": 32,
            "fresh_moments": True,
            "parent_optimizer_step_ignored": int(load_pickle(F1)["step"]),
            "initial_absolute_step": 0,
        },
        "loss": {
            "common": "boundary CE + extent CE + .5 predictor h-MSE",
            "descendant": ".5 replay h-MSE + length-aware byte loss",
            "proposal": 0,
            "sample": "common + exactly-once local score-corrected descendant",
        },
        "checkpoints": list(CHECKPOINTS),
        "arms": list(ARMS),
        "L": 4,
        "Q": 3,
        "S": 1,
        "R": 0,
        "lambda_op": 0.0,
        "sample_temperature": 1.0,
        "occurrences": "two shared-parameter same-input applications, both actualS=0/condition=0; same nonstructure keys; NOT S2",
        "evaluation": "fixed nu vector, occurrence0, global IDs retained; paired chunk32; no parameter update",
        "necessity": {
            "controls": ["shifted_cuts", "permuted_extents", "joint", "fixed_neutral"],
            "accuracy_delta": 0.01,
            "relative_loss_reduction": 0.02,
            "paired_bootstrap": 2000,
            "percentiles": [2.5, 97.5],
            "accuracy_seeds": [9600, 9601, 9602, 9603],
            "loss_seeds": [9700, 9701, 9702, 9703],
        },
        "gradient_variance": {
            "fixed_batch": 32,
            "structure_roots": 4,
            "root_base": 1100,
            "scope": "diagnostic only; no loss/optimizer changes",
        },
    }


def initialized():
    value = with_frozen_g0_codec(load_pickle(F1)["params"], load_pickle(CODEC))
    if tree_hash(value) != INITIAL_TREE:
        raise RuntimeError("matched initialization mismatch")
    return value


def frozen_hash(params, moments=None):
    mask = stochastic_model_update_mask(params)
    selected = {}
    for (path, value), enabled in zip(
        jax.tree_util.tree_leaves_with_path(params), jax.tree.leaves(mask), strict=True
    ):
        if not bool(enabled):
            selected[jax.tree_util.keystr(path)] = value
    if moments is not None:
        for i, moment in enumerate(moments):
            for (path, value), enabled in zip(
                jax.tree_util.tree_leaves_with_path(moment), jax.tree.leaves(mask), strict=True
            ):
                if not bool(enabled):
                    selected[f"moment{i}:" + jax.tree_util.keystr(path)] = value
    return tree_hash(selected)


def data_split(train=False):
    return dataset(
        1024 if train else 256, 950 if train else 951, global_id_offset=0 if train else 1024
    )


def training_batch(update):
    data = data_split(True)
    order = jax.random.permutation(jax.random.fold_in(jax.random.key(950), update // 32), 1024)
    indices = order[(update * 32) % 1024 : (update * 32) % 1024 + 32]
    return with_nu(take_batch(data, indices), update)


def objective(params, batch, *, arm, connected=True):
    return generated_two_pass_objective(
        params,
        batch,
        stochastic_root("operator"),
        stochastic_root("diffusion"),
        stochastic_root("structure"),
        0,
        arm=arm,
        temperature=1.0 if arm == "sample" else None,
        apply_score_correction=connected,
        dynamic_capacity=True,
        config=config(),
    )


def gate_record(name, protocol, payload):
    record(OUT / (name + ".json"), {"protocol": protocol, **payload})


def audit():
    protocol = manifest()
    params = initialized()
    zeros = adamw_init(params)
    if any(bool(jnp.any(x != 0)) for x in jax.tree.leaves(zeros)):
        raise RuntimeError("Adam not fresh")
    train, val = data_split(True), data_split(False)

    def hashes(data):
        return [
            hashlib.sha256(np.asarray(row).tobytes() + np.asarray(mask).tobytes()).hexdigest()
            for row, mask in zip(data["target"], data["target_mask"], strict=True)
        ]

    tr, va = hashes(train), hashes(val)
    if set(tr) & set(va):
        raise RuntimeError("G train/validation overlap")
    gate_record(
        "provenance",
        protocol,
        {
            "passed": True,
            "complete_initialization": tree_hash(params),
            "all_arm_initializations": {arm: tree_hash(initialized()) for arm in ARMS},
            "fresh_moments": True,
            "train_ordered_sha256": hashlib.sha256("".join(tr).encode()).hexdigest(),
            "validation_ordered_sha256": hashlib.sha256("".join(va).encode()).hexdigest(),
            "codec_reference_reconstruction": "G.0.1 known accuracy99.469%, NLL.019432; oracle exact83.169%; these are not G model metrics",
        },
    )
    gate_record("historical_baseline", protocol, {"passed": True, "files": historical_inventory()})


def historical_inventory():
    common = (
        Path(
            subprocess.check_output(
                ["git", "rev-parse", "--git-common-dir"], cwd=ROOT, text=True
            ).strip()
        )
        .resolve()
        .parent
    )
    roots = []
    for project in (ROOT, common):
        roots += [project / "results/phase_b/checkpoints", project / "results/runs/phase_f"]
    roots += [
        ROOT / "results/runs/phase_g/g0/v1",
        ROOT / "results/runs/phase_g/g0_1/v1",
        common / "results/runs/phase_g/g0/frozen_v1_archive",
        common / "results/runs/phase_g/g0_1/frozen_v1_archive",
        ROOT / "results/phase_f",
        ROOT / "results/phase_g/g0",
        ROOT / "results/phase_g/g0_1",
    ]
    paths = {
        p.resolve()
        for r in roots
        if r.exists()
        for p in r.rglob("*")
        if p.is_file() and p.suffix in (".pkl", ".json", ".jsonl", ".md")
    }
    return {str(p): sha(p) for p in sorted(paths)}


def historical_check():
    protocol = manifest()
    require("historical_baseline", protocol)
    before = json.loads((OUT / "historical_baseline.json").read_text())["files"]
    after = historical_inventory()
    if before != after:
        raise RuntimeError("historical artifacts changed")
    gate_record(
        "historical_preservation",
        protocol,
        {"passed": True, "file_count": len(after), "bitwise_hashes_unchanged": True},
    )


def import_fixtures():
    protocol = manifest()
    common = (
        Path(
            subprocess.check_output(
                ["git", "rev-parse", "--git-common-dir"], cwd=ROOT, text=True
            ).strip()
        )
        .resolve()
        .parent
    )
    rows = []
    expected = {
        "copy": "c1b0916cd161a398c3933591ebd4e6e0ab2cc0560f1139befed1d0d5b6773eb1",
        "reverse": "1a1b3748d333e80037712568dc40762470a82cb95ab38de4d6c7bf709679bf6c",
    }
    for name, digest in expected.items():
        source = common / f"results/phase_b/checkpoints/{name}.pkl"
        destination = ROOT / f"results/phase_b/checkpoints/{name}.pkl"
        if sha(source) != digest:
            raise RuntimeError("historical fixture source changed")
        if destination.exists():
            if sha(destination) != digest:
                raise RuntimeError("refusing fixture overwrite")
        else:
            shutil.copy2(source, destination)
        if sha(destination) != digest:
            raise RuntimeError("fixture copy mismatch")
        rows.append({"source": str(source), "copy": str(destination), "sha256": digest})
    gate_record(
        "fixture_import",
        protocol,
        {
            "passed": True,
            "files": rows,
            "historical_reference": "results/phase_c/C3_FULL_PARITY.md",
        },
    )


def oracle():
    protocol = manifest()
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
            value = conformance(parameter, loss)
            rows.append(value)
    gate_record(
        "production_oracle",
        protocol,
        {
            "passed": all(x["accepted"] for x in rows),
            "rows": rows,
            "roots": 16,
            "samples_per_root": 4096,
            "seed": 975,
        },
    )
    if not all(x["accepted"] for x in rows):
        raise RuntimeError("production score oracle failed")


def interventions(params, batch):
    """Real replay traces; teacher only supplies losses/corruption OUTSIDE forward."""
    cfg = config()
    codec = load_pickle(CODEC)
    clean = apply_clean_target_teacher(codec, batch["target"], batch["target_mask"], config=cfg)
    h, _ = make_initial_corruption(
        clean["target"]["h0"],
        batch["nu"],
        stochastic_root("diffusion"),
        batch["global_example_id"],
        optimizer_step=0,
    )
    neutral = neutral_structure(len(h), cfg)
    assert neutral.length is not None
    sanitized = make_sanitized_target_analysis(
        neutral.c_b, neutral.length, config=cfg, target_width=64
    )
    ops = TorxOps.create(
        stochastic_root("operator"),
        config=TorxOperatorConfig(operator_stochasticity=True, lambda_op=0.0),
    )
    output = {}
    max_error = None
    for name, semantic in _control_structures(batch).items():
        routed = dynamic_capacity_routing(semantic, M_max=32, K=8, max_extent=4)
        if not bool(routed.routing.executed):
            raise RuntimeError(f"invalid intervention {name}")
        routing = routed.routing
        out = apply_model(
            params,
            batch["prompt"],
            batch["prompt_mask"],
            sanitized.target_bytes_placeholder,
            sanitized.target_mask_placeholder,
            target_analysis=as_sanitized_model_target_analysis(sanitized),
            carrier_h_input=h,
            noise_level=batch["nu"],
            actual_s_index=0,
            denoise_condition_index=0,
            committed_c_b=routing.structure.c_b,
            committed_activity=routing.structure.activity,
            committed_length=routing.structure.length,
            observed_b=neutral.c_b,
            observed_l=neutral.length,
            routing_metadata=routing.metadata,
            ops=ops,
            config=cfg,
            capture_diagnostics=True,
        )
        if name == "correct":
            changed = make_sanitized_target_analysis(
                neutral.c_b, neutral.length, config=cfg, target_width=64, placeholder_value=31
            )
            other = apply_model(
                params,
                batch["prompt"],
                batch["prompt_mask"],
                changed.target_bytes_placeholder,
                changed.target_mask_placeholder,
                target_analysis=as_sanitized_model_target_analysis(changed),
                carrier_h_input=h,
                noise_level=batch["nu"],
                actual_s_index=0,
                denoise_condition_index=0,
                committed_c_b=routing.structure.c_b,
                committed_activity=routing.structure.activity,
                committed_length=routing.structure.length,
                observed_b=neutral.c_b,
                observed_l=neutral.length,
                routing_metadata=routing.metadata,
                ops=ops,
                config=cfg,
                capture_diagnostics=True,
            )
            if not jax.tree_util.tree_all(
                jax.tree.map(
                    jnp.array_equal,
                    (out["prediction"], out["byte_logits"]),
                    (other["prediction"], other["byte_logits"]),
                )
            ):
                raise RuntimeError("sanitized-placeholder inference leakage")
            deterministic = apply_model(
                torx_means_to_deterministic(params),
                batch["prompt"],
                batch["prompt_mask"],
                sanitized.target_bytes_placeholder,
                sanitized.target_mask_placeholder,
                target_analysis=as_sanitized_model_target_analysis(sanitized),
                carrier_h_input=h,
                noise_level=batch["nu"],
                actual_s_index=0,
                denoise_condition_index=0,
                committed_c_b=routing.structure.c_b,
                committed_activity=routing.structure.activity,
                committed_length=routing.structure.length,
                observed_b=neutral.c_b,
                observed_l=neutral.length,
                routing_metadata=routing.metadata,
                ops=DeterministicOps(),
                config=cfg,
                capture_diagnostics=True,
            )
            max_error = max(
                float(jnp.max(jnp.abs(a - b)))
                for a, b in zip(
                    jax.tree.leaves((out["prediction"], out["byte_logits"])),
                    jax.tree.leaves((deterministic["prediction"], deterministic["byte_logits"])),
                    strict=True,
                )
            )
            if max_error > 1e-6:
                raise RuntimeError("full-model deterministic/Torx parity failed")
        metadata = routing.metadata
        identity = jnp.arange(cfg.carrier.C)[None, :, None] + jnp.zeros((len(h), cfg.carrier.C, 1))
        if not jnp.array_equal(
            unpool_values(pack_values(identity, metadata), metadata, C=32), identity
        ):
            raise RuntimeError("carrier identity lost")
        expected = out["current_carrier_input"] + ops.linear(
            out["unpooled_carrier"], params["carrier_out"], name="model.carrier_output"
        )
        if not jnp.array_equal(expected, out["pre_head_carrier"]):
            raise RuntimeError("carrier residual mismatch")
        packed_mask = draft_attention_mask(metadata.block_id, metadata.query_mask, metadata.kv_mask)
        positions = metadata.carrier_to_m * cfg.packing.K + metadata.carrier_to_k
        carrier_mask = packed_mask[
            jnp.arange(len(h))[:, None, None], positions[:, :, None], positions[:, None, :]
        ]
        if out["dit_aux"]["cycle_ffn_rms"].shape != (12,) or not bool(
            jnp.all(out["dit_aux"]["cycle_ffn_rms"] > 0)
        ):
            raise RuntimeError("full LxQ SwiGLU path not executed")
        output[name] = {
            "semantic_b": semantic.c_b,
            "ell": semantic.length,
            "capacity_cuts": routed.capacity_cut,
            "packed_to_carrier": metadata.packed_to_carrier,
            "attention_reachability": carrier_mask,
            "emission_mask": out["emit_mask"],
            "byte_logits": out["byte_logits"],
            "packed_carrier": out["packed_carrier"],
            "dit_diagnostics": out["dit_aux"],
        }
    baseline = output["correct"]
    return {
        name: {
            "packing_changed": not bool(
                jnp.array_equal(x["packed_to_carrier"], baseline["packed_to_carrier"])
            ),
            "attention_changed": not bool(
                jnp.array_equal(x["attention_reachability"], baseline["attention_reachability"])
            ),
            "emission_changed": not bool(
                jnp.array_equal(x["emission_mask"], baseline["emission_mask"])
            ),
            "byte_logit_rms_change": float(
                jnp.sqrt(jnp.mean((x["byte_logits"] - baseline["byte_logits"]) ** 2))
            ),
            "semantic_b": x["semantic_b"],
            "length": x["ell"],
            "capacity_cuts": x["capacity_cuts"],
            "attention_allowed_edges": jnp.sum(x["attention_reachability"], axis=(1, 2)),
            "dit_block_rms": x["dit_diagnostics"]["block_rms"],
            "dit_aux_keys": list(x["dit_diagnostics"]),
            "placeholder_mutation_invariant": True if name == "correct" else None,
            "full_model_parity_max_absolute_error": max_error if name == "correct" else None,
        }
        for name, x in output.items()
    }


def gradient_variance(params, arm):
    if arm != "sample":
        return {
            "roots": 4,
            "variance_squared_norm": 0.0,
            "deduced_exact_zero_by_deterministic_structure_policy": True,
        }
    batch = training_batch(0)

    grad = jax.jit(
        jax.grad(
            lambda p, b, root: generated_two_pass_objective(
                p,
                b,
                stochastic_root("operator"),
                stochastic_root("diffusion"),
                root,
                0,
                arm="sample",
                temperature=1.0,
                dynamic_capacity=True,
                config=config(),
            )[0]
        )
    )
    values = [grad(params, batch, jax.random.fold_in(jax.random.key(1100), i)) for i in range(4)]
    if not tree_all_finite(values):
        raise RuntimeError("nonfinite gradient-variance diagnostic")
    arrays = jax.tree.map(lambda *xs: jnp.stack(xs), *values)
    variance = sum(jnp.sum(jnp.var(x, axis=0, ddof=1)) for x in jax.tree.leaves(arrays))
    return {
        "roots": 4,
        "root_base": 1100,
        "variance_squared_norm": variance,
        "root_raw_norms": [
            jnp.sqrt(sum(jnp.sum(x * x) for x in jax.tree.leaves(v))) for v in values
        ],
        "data_noise_operator_keys_fixed": True,
        "diagnostic_only": True,
    }


def full_model_gate():
    protocol = manifest()
    require("provenance", protocol)
    require("production_oracle", protocol)
    params = initialized()
    batch = training_batch(0)
    moments = (adamw_init(params), adamw_init(params))
    before = frozen_hash(params, moments)
    grad_fn = jax.jit(
        jax.value_and_grad(
            lambda p, b, connected: objective(p, b, arm="sample", connected=connected), has_aux=True
        ),
        static_argnums=(2,),
    )
    (loss, metrics), connected = grad_fn(params, batch, True)
    (_, _), detached = grad_fn(params, batch, False)
    teacher_grad = jax.jit(
        jax.value_and_grad(lambda p, b: objective(p, b, arm="teacher"), has_aux=True)
    )
    (teacher_loss, _), teacher_gradient = teacher_grad(params, batch)
    if not tree_all_finite((teacher_loss, teacher_gradient)):
        raise RuntimeError("nonfinite teacher first gradient")
    eager_loss, eager_metric = objective(params, batch, arm="teacher")
    if not bool(jnp.allclose(eager_loss, teacher_loss, rtol=1e-5, atol=1e-6)):
        raise RuntimeError("eager/JIT teacher loss parity failed")

    def norm(tree):
        return float(jnp.sqrt(sum(jnp.sum(x * x) for x in jax.tree.leaves(tree))))

    difference = jax.tree.map(lambda a, b: a - b, connected, detached)
    if (
        not tree_all_finite((loss, connected, detached))
        or norm(difference["b_head"]) <= 0
        or norm(difference["l_head"]) <= 0
    ):
        raise RuntimeError("real downstream score-gradient connection failed")
    update = jax.jit(
        generated_train_step, static_argnames=("arm", "temperature", "dynamic_capacity", "config")
    )
    new, new_moments, step, result = update(
        params,
        moments,
        0,
        batch,
        stochastic_root("operator"),
        stochastic_root("diffusion"),
        stochastic_root("structure"),
        arm="teacher",
        dynamic_capacity=True,
        config=config(),
    )
    if (
        not bool(result["executed"])
        or int(step) != 1
        or not tree_all_finite((new, new_moments))
        or frozen_hash(new, new_moments) != before
    ):
        raise RuntimeError("prospective full-model update failed")
    if tree_hash(new) == tree_hash(params):
        raise RuntimeError("optimizer applied no change")
    trace = interventions(params, take_batch(batch, jnp.arange(2)))
    if (
        not trace["shifted_cuts"]["attention_changed"]
        or not trace["permuted_extents"]["emission_changed"]
    ):
        raise RuntimeError("real structural consumption not demonstrated")
    deterministic = torx_means_to_deterministic(params)
    codec = load_pickle(CODEC)
    target = apply_clean_target_teacher(
        codec, batch["target"][:2], batch["target_mask"][:2], config=config()
    )
    mean = apply_clean_target_teacher(
        deterministic,
        batch["target"][:2],
        batch["target_mask"][:2],
        config=config(),
        ops=DeterministicOps(),
    )
    if not jnp.array_equal(target["target"]["h0"], mean["target"]["h0"]):
        raise RuntimeError("codec overlay teacher mismatch")
    gate_record(
        "full_model",
        protocol,
        {
            "passed": True,
            "batch": 32,
            "sample_loss": loss,
            "teacher_trial_loss": result["loss"],
            "teacher_trial_raw_norm": result["grad_raw_norm"],
            "teacher_trial_clipped_gradient_norm": jnp.minimum(result["grad_raw_norm"], 1.0),
            "teacher_trial_parameter_update_norm": norm(
                jax.tree.map(lambda a, b: a - b, new, params)
            ),
            "teacher_full_raw_norm": norm(teacher_gradient),
            "teacher_subsystem_norms": {
                name: norm(teacher_gradient[name])
                for name in (
                    "encoder",
                    "carrier_in",
                    "dit",
                    "h_head",
                    "b_head",
                    "l_head",
                    "decoder",
                )
            },
            "sample_subsystem_norms": {
                name: norm(connected[name])
                for name in (
                    "encoder",
                    "carrier_in",
                    "dit",
                    "h_head",
                    "b_head",
                    "l_head",
                    "decoder",
                )
            },
            "eager_jit_loss_absolute_error": jnp.abs(eager_loss - teacher_loss),
            "sample_full_raw_norm": norm(connected),
            "connected_detached_difference": norm(difference),
            "boundary_score_norm": norm(difference["b_head"]),
            "extent_score_norm": norm(difference["l_head"]),
            "trial_frozen_states_bitwise_equal": True,
            "trial_parameters_and_moments_finite": True,
            "trial_not_part_of_training": True,
            "sample_gradient_variance": gradient_variance(params, "sample"),
            "interventions": trace,
        },
    )


def require(name, protocol):
    path = OUT / (name + ".json")
    value = json.loads(path.read_text())
    if value.get("protocol") != protocol or not value.get("passed"):
        raise RuntimeError(f"missing/stale/failed gate: {name}")


@partial(jax.jit, static_argnames=("arm",))
def validation_apply(params, batch, *, arm):
    return objective(params, batch, arm=arm)


def evaluate(params, arm):
    data = with_nu(data_split(), 0)
    rows = []
    for start in range(0, 256, 32):
        loss, metric = validation_apply(
            params, take_batch(data, jnp.arange(start, start + 32)), arm=arm
        )
        if not bool(metric["executed"]):
            raise RuntimeError("invalid validation trajectory")
        rows.append({"loss": loss, **metric})
    result = {}
    for key in rows[0]:
        if key.startswith("example_"):
            result[key] = jnp.concatenate([r[key] for r in rows])
        elif key in (
            "missing_bytes",
            "extra_bytes",
            "overlap_byte_count",
            "overlap_example_support",
        ):
            result[key] = sum(r[key] for r in rows)
        elif key == "executed":
            result[key] = jnp.all(jnp.stack([r[key] for r in rows]))
        elif key == "error_code_max":
            result[key] = jnp.max(jnp.stack([r[key] for r in rows]))
        else:
            result[key] = sum(r[key] for r in rows) / len(rows)
    return result


def checkpoint(state, training):
    path = RUN / state["arm"] / f"update_{state['updates']:03d}.pkl"
    if path.exists():
        old = load_pickle(path)
        if old["protocol"] != state["protocol"] or tree_hash(
            (old["params"], old["moments"])
        ) != tree_hash((state["params"], state["moments"])):
            raise RuntimeError("checkpoint identity mismatch")
        if any(old[k] != state[k] for k in ("arm", "step", "updates")):
            raise RuntimeError("checkpoint step/arm mismatch")
    else:
        save_pickle(path, state)
    vals = {kind: evaluate(state["params"], kind) for kind in ("teacher", "argmax", "sample")}
    receipt = {
        "protocol": state["protocol"],
        "arm": state["arm"],
        "updates": state["updates"],
        "optimizer_absolute_step": state["step"],
        "path": str(path.relative_to(ROOT)),
        "sha256": sha(path),
        "training": training,
        "validation": vals,
    }
    if state["updates"] in (0, 100):
        receipt["gradient_variance"] = gradient_variance(state["params"], state["arm"])
    record(OUT / state["arm"] / f"update_{state['updates']:03d}.json", receipt)
    return receipt


def train_arm(arm):
    protocol = manifest()
    for name in ("provenance", "production_oracle", "full_model", "focused_tests"):
        require(name, protocol)
    if arm != "teacher":
        require("necessity", protocol)
    path = RUN / arm / "state.pkl"
    if path.exists():
        state = load_current(path, protocol, arm)
        if (
            state["protocol"] != protocol
            or state["arm"] != arm
            or state["step"] != state["updates"]
        ):
            raise RuntimeError("resume identity mismatch")
    else:
        p = initialized()
        state = {
            "params": p,
            "moments": (adamw_init(p), adamw_init(p)),
            "step": 0,
            "updates": 0,
            "arm": arm,
            "protocol": protocol,
        }
        checkpoint(state, None)
        save_current(path, state)
    frozen = frozen_hash(initialized(), (adamw_init(initialized()), adamw_init(initialized())))
    update = jax.jit(
        generated_train_step, static_argnames=("arm", "temperature", "dynamic_capacity", "config")
    )
    while state["updates"] < 100:
        index = state["updates"]
        p, m, step, metric = update(
            state["params"],
            state["moments"],
            state["step"],
            training_batch(index),
            stochastic_root("operator"),
            stochastic_root("diffusion"),
            stochastic_root("structure"),
            arm=arm,
            temperature=1.0 if arm == "sample" else None,
            dynamic_capacity=True,
            config=config(),
        )
        if (
            not bool(metric["executed"])
            or int(step) != index + 1
            or not tree_all_finite((p, m))
            or frozen_hash(p, m) != frozen
        ):
            failure = {
                "protocol": protocol,
                "completed_update": False,
                "attempt": index + 1,
                "metrics": metric,
            }
            # Preserve nonfinite failure evidence as tagged values, never repair
            # tensors or serialize nonstandard NaN JSON as successful results.
            safe = json.loads(
                json.dumps(failure, default=serialise),
                parse_constant=lambda value: {"nonfinite": value},
            )
            record(
                OUT / arm / f"failed_attempt_{index + 1:03d}.json",
                safe,
            )
            raise RuntimeError("failed G update; durable state unchanged")
        state = {**state, "params": p, "moments": m, "step": int(step), "updates": index + 1}
        if state["updates"] in CHECKPOINTS:
            checkpoint(state, metric)
            print(
                json.dumps(
                    {
                        "arm": arm,
                        "updates": index + 1,
                        "loss": float(metric["loss"]),
                        "byte_accuracy": float(metric["byte_accuracy"]),
                        "raw_norm": float(metric["grad_raw_norm"]),
                    }
                ),
                flush=True,
            )
        save_current(path, state)
    return state


def necessity():
    protocol = manifest()
    path = RUN / "teacher/state.pkl"
    state = load_pickle(path)
    if state["protocol"] != protocol or state["updates"] != 100:
        raise RuntimeError("teacher Arm A not complete")
    data = with_nu(data_split(), 0)
    results = {}
    controls = _control_structures(data)
    for name, semantic in controls.items():
        values = []
        apply = jax.jit(
            lambda p, b, s: evaluate_committed_replay(
                p,
                b,
                s,
                stochastic_root("operator"),
                stochastic_root("diffusion"),
                0,
                clean_params=load_pickle(CODEC),
                clean_ops=DeterministicOps(),
                dynamic_capacity=True,
                config=config(),
            )
        )
        for start in range(0, 256, 32):
            ids = jnp.arange(start, start + 32)
            metric, replay = apply(
                state["params"], take_batch(data, ids), jax.tree.map(lambda x: x[ids], semantic)
            )
            if not bool(replay.executed):
                raise RuntimeError("necessity control invalid")
            values.append(metric)
        joined = jax.tree.map(lambda *xs: jnp.concatenate(xs), *values)
        results[name] = {
            "accuracy": np.asarray(joined.accuracy_per_example),
            "loss": np.asarray(joined.loss_per_example),
            "exact": np.asarray(joined.exact_per_example),
            "overlap_nll": np.asarray(joined.overlap_nll_per_example),
        }
    comparisons = {}
    passed = True
    for i, name in enumerate(("shifted_cuts", "permuted_extents", "joint", "fixed_neutral")):
        accuracy = results["correct"]["accuracy"] - results[name]["accuracy"]
        relative = (results[name]["loss"] - results["correct"]["loss"]) / np.maximum(
            results[name]["loss"], 1e-12
        )
        aci = _bootstrap_interval(accuracy, 9600 + i)
        lci = _bootstrap_interval(relative, 9700 + i)
        ok = (
            float(accuracy.mean()) >= 0.01
            and float(relative.mean()) >= 0.02
            and aci[0] > 0
            and lci[0] > 0
        )
        passed &= ok
        comparisons[name] = {
            "accepted": ok,
            "accuracy_delta": accuracy.mean(),
            "relative_loss_reduction": relative.mean(),
            "accuracy_95CI": aci,
            "relative_loss_95CI": lci,
        }
    gate_record(
        "necessity",
        protocol,
        {
            "passed": passed,
            "teacher_checkpoint_sha256": sha(RUN / "teacher/update_100.pkl"),
            "comparisons": comparisons,
            "per_example": results,
            "global_ids": data["global_example_id"],
            "post_teacher_interventions": interventions(
                state["params"], take_batch(data, jnp.arange(2))
            ),
        },
    )
    if not passed:
        raise RuntimeError("STRUCTURAL_NECESSITY_GATE_FAILED: no generated arms permitted")


def focused_tests():
    protocol = manifest()
    command = [
        sys.executable,
        "-m",
        "pytest",
        "tests/test_phase_g_amended.py",
        "tests/test_phase_g_structure.py",
        "tests/test_phase_g0.py",
        "tests/test_phase_g_oracle.py",
        "-ra",
    ]
    outcome = subprocess.run(
        command,
        cwd=ROOT,
        env={**__import__("os").environ, "PYTHONPATH": "src"},
        capture_output=True,
        text=True,
        check=False,
    )
    gate_record(
        "focused_tests",
        protocol,
        {
            "passed": outcome.returncode == 0,
            "command": command,
            "exit_code": outcome.returncode,
            "stdout": outcome.stdout,
            "stderr": outcome.stderr,
        },
    )
    if outcome.returncode:
        raise RuntimeError("focused gate failed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        required=True,
        choices=(
            "audit",
            "oracle",
            "tests",
            "full-model",
            "teacher",
            "necessity",
            "argmax",
            "sample",
            "history",
            "import-fixtures",
        ),
    )
    arg = parser.parse_args().stage
    OUT = OUT / implementation_sha()[:12]
    if arg == "audit":
        audit()
    elif arg == "import-fixtures":
        import_fixtures()
    elif arg == "oracle":
        oracle()
    elif arg == "tests":
        focused_tests()
    elif arg == "full-model":
        full_model_gate()
    elif arg == "necessity":
        necessity()
    elif arg == "history":
        historical_check()
    else:
        train_arm(arg)
