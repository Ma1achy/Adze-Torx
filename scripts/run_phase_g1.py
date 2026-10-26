"""Inference-only G.1 path diagnostics. No optimizer or training entry point."""

from dataclasses import replace
import hashlib
import json
from pathlib import Path
import pickle
import platform
import subprocess
import sys

import jax
import jax.numpy as jnp
import numpy as np

from run_phase_g import config, dataset, with_nu, take_batch, stochastic_root, _control_structures
from adze_t.backends.deterministic import DeterministicOps
from adze_t.backends.mapping import torx_means_to_deterministic
from adze_t.backends.torx import TorxOps, TorxOperatorConfig
from adze_t.decoder import apply_decoder
from adze_t.denoise import make_sanitized_target_analysis, as_sanitized_model_target_analysis
from adze_t.masking import draft_attention_mask
from adze_t.model import apply_model, apply_clean_target_teacher
from adze_t.phase_f_1 import make_initial_corruption
from adze_t.phase_g import neutral_structure, generated_byte_metrics
from adze_t.phase_g0 import dynamic_capacity_routing
from adze_t.state import CommittedStructure

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/phase_g/g1"
RAW = ROOT / "results/runs/phase_g/g1/v1/diagnostics.npz"
CHECKPOINT = ROOT / "results/runs/phase_g/preflight_g01_v1/teacher/update_100.pkl"
CODEC = ROOT / "results/runs/phase_g/g0_1/v1/frozen_codec.pkl"
CFG = config()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, value):
    def encode(x):
        return np.asarray(x).tolist()

    if path.exists():
        raise RuntimeError(f"refusing diagnostic overwrite: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, default=encode, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


class DiagnosticOps(DeterministicOps):
    """Explicitly labelled ablations; production LearnedOps is untouched."""

    def __init__(self, variant="full"):
        self.variant = variant

    def linear(self, x, params, *, name="linear"):
        if self.variant == "zero_context" and name == "dit.conditioning_trunk":
            x = x.at[:, : CFG.model.d_ctx].set(0)
        value = super().linear(x, params, name=name)
        if self.variant == "no_attention" and name.startswith("dit.block_") and name.endswith(".o"):
            value = jnp.zeros_like(value)
        return value


def forward(params, h, nu, boundary, length, *, variant="full", torx=False):
    route = dynamic_capacity_routing(
        CommittedStructure(boundary, length > 0, length), M_max=32, K=8, max_extent=4
    ).routing
    neutral = neutral_structure(h.shape[0], CFG)
    assert neutral.length is not None
    sanitized = make_sanitized_target_analysis(
        neutral.c_b, neutral.length, config=CFG, target_width=64
    )
    ops = (
        TorxOps.create(stochastic_root("operator"), config=TorxOperatorConfig(lambda_op=0.0))
        if torx
        else DiagnosticOps(variant)
    )
    out = apply_model(
        params,
        jnp.full((h.shape[0], 1), 0xF1, jnp.int32),
        jnp.ones((h.shape[0], 1), bool),
        sanitized.target_bytes_placeholder,
        sanitized.target_mask_placeholder,
        config=CFG,
        ops=ops,
        target_analysis=as_sanitized_model_target_analysis(sanitized),
        carrier_h_input=h,
        noise_level=nu,
        actual_s_index=0,
        denoise_condition_index=0,
        committed_c_b=route.structure.c_b,
        committed_length=route.structure.length,
        committed_activity=route.structure.activity,
        observed_b=neutral.c_b,
        observed_l=neutral.length,
        routing_metadata=route.metadata,
    )
    m = route.metadata
    pos = m.carrier_to_m * 8 + m.carrier_to_k
    mask = draft_attention_mask(m.block_id, m.query_mask, m.kv_mask)
    reach = mask[jnp.arange(h.shape[0])[:, None, None], pos[:, :, None], pos[:, None, :]]
    return {
        "valid": route.executed,
        "h_hat": out["carrier"],
        "pre_head": out["pre_head_carrier"],
        "unpooled": out["unpooled_carrier"],
        "logits": out["byte_logits"],
        "context": out["context_global"],
        "reach": reach,
        "pack": m.packed_to_carrier,
        "unpool_m": m.carrier_to_m,
        "unpool_k": m.carrier_to_k,
        "query": m.query_mask,
        "kv": m.kv_mask,
        "mechanical_b": route.structure.c_b,
    }


def decode(h, length, decoder):
    return apply_decoder(h, length, decoder, CFG, DeterministicOps())[0]


def decode_local(h, length, decoder):
    """Diagnostic reset at every carrier; preserve its original embedding."""
    one = replace(CFG, carrier=replace(CFG.carrier, C=1))

    def single(x, ell, embed):
        params = {**decoder, "carrier_embed": embed[None]}
        return apply_decoder(x[:, None], ell[:, None], params, one, DeterministicOps())[0][:, 0]

    value = jax.vmap(single)(
        jnp.swapaxes(h, 0, 1), jnp.swapaxes(length, 0, 1), decoder["carrier_embed"]
    )
    return jnp.swapaxes(value, 0, 1)


def local_values(logits, slots, mask):
    nll = -jnp.take_along_axis(jax.nn.log_softmax(logits), slots[..., None], axis=-1)[..., 0]
    correct = (jnp.argmax(logits, -1) == slots) & mask
    return {
        "nll": np.asarray(jnp.sum(jnp.where(mask, nll, 0), -1)),
        "correct": np.asarray(jnp.sum(correct, -1)),
        "count": np.asarray(jnp.sum(mask, -1)),
        "exact": np.asarray(jnp.all(correct | ~mask, (1, 2))),
    }


def metrics(logits, length, slots, mask):
    m = generated_byte_metrics(logits, length, slots, mask)
    return {
        "example_mean_accuracy": float(jnp.mean(m.accuracy_per_example)),
        "exact_sequence": float(jnp.mean(m.exact_per_example)),
        "example_mean_nll": float(jnp.mean(m.overlap_nll_per_example)),
        "missing": int(jnp.sum(m.missing_count)),
        "extra": int(jnp.sum(m.extra_count)),
    }


def group(values, selection):
    support = int(np.sum(selection))
    count = int(np.sum(values["count"][selection]))
    return {
        "sites": support,
        "bytes": count,
        "accuracy": float(np.sum(values["correct"][selection]) / count) if count else None,
        "nll": float(np.sum(values["nll"][selection]) / count) if count else None,
    }


def fixtures():
    """Paired teacher-controlled instruments, not a production task change."""
    keys = jax.random.split(jax.random.key(1101), 6)
    n = 16
    q = jax.random.randint(keys[0], (n,), 1, 6)
    counts = 32 + 4 * (jnp.arange(n) % 8)
    background = jax.random.randint(keys[1], (n, 64), 1, 33)
    labels = jax.random.randint(keys[2], (2, n, 4), 1, 33)
    competitors = jax.random.randint(keys[3], (n, 4), 1, 33)
    b = jax.random.bernoulli(keys[4], 0.4, (n, 32)).astype(jnp.int32)
    b = b.at[jnp.arange(n), q - 1].set(1).at[jnp.arange(n), q].set(0)
    b = b.at[jnp.arange(n), q + 1].set(1).at[:, -1].set(1)
    wrong = b.at[jnp.arange(n), q].set(1)
    data = []
    for payload in labels:
        target = background.reshape(n, 16, 4)
        target = target.at[jnp.arange(n), q].set(payload)
        target = target.at[jnp.arange(n), q + 1].set(payload)
        target = target.at[jnp.arange(n), q - 1].set(competitors)
        target = target.reshape(n, 64)
        mask = jnp.arange(64)[None] < counts[:, None]
        data.append({"target": jnp.where(mask, target, 0), "target_mask": mask})
    return data, q, b, wrong


def main():
    if sha(CHECKPOINT) != "203926286c4f991b316e9f9d8cb8bee529861da5124f374279b82d143decd220":
        raise RuntimeError("teacher checkpoint mismatch")
    if sha(CODEC) != "213d71be58b4e3a0eb685330ff8f20ad0d2574184d59060890baa4a2eed586e7":
        raise RuntimeError("codec mismatch")
    if RAW.exists() or (OUT / "diagnosis.json").exists():
        raise RuntimeError("diagnostics already exist; no silent retry/overwrite")
    baseline_path = ROOT / "results/phase_g/preflight_g01_v1/f571ab40659d/historical_baseline.json"
    historic = json.loads(baseline_path.read_text())["files"]
    assert all(sha(p) == digest for p, digest in historic.items())
    frozen_hash = sha(CHECKPOINT)
    with CHECKPOINT.open("rb") as stream:
        state = pickle.load(stream)
    assert state["step"] == state["updates"] == 100 and state["arm"] == "teacher"
    params = jax.tree.map(jnp.asarray, torx_means_to_deterministic(state["params"]))
    with CODEC.open("rb") as stream:
        codec = jax.tree.map(jnp.asarray, pickle.load(stream))
    teacher = jax.jit(
        lambda target, mask: apply_clean_target_teacher(codec, target, mask, config=CFG)
    )
    run = jax.jit(forward, static_argnames=("variant", "torx"))
    dec = jax.jit(decode)
    local_dec = jax.jit(decode_local)
    data = with_nu(dataset(256, 951, global_id_offset=1024), 0)
    structures = _control_structures(data)
    raw, aggregates = {}, {}
    ablation_rows = {}
    parity = {}
    for start in range(0, 256, 8):
        batch = take_batch(data, jnp.arange(start, start + 8))
        clean = teacher(batch["target"], batch["target_mask"])
        h, _ = make_initial_corruption(
            clean["target"]["h0"],
            batch["nu"],
            stochastic_root("diffusion"),
            batch["global_example_id"],
            optimizer_step=0,
        )
        slots, mask, length = (
            clean["target"]["teacher"].slot_bytes,
            clean["target"]["teacher"].slot_mask,
            clean["target"]["teacher"].length,
        )
        outputs = {}
        for name in ("correct", "shifted_cuts"):
            b = structures[name].c_b[start : start + 8]
            out = run(params, h, batch["nu"], b, length)
            if not bool(out["valid"]):
                raise RuntimeError("invalid route")
            outputs[name] = out
            vals = local_values(out["logits"], slots, mask)
            for key, value in {
                **vals,
                **{
                    k: np.asarray(v)
                    for k, v in out.items()
                    if k not in ("valid", "logits", "pre_head", "h_hat", "unpooled", "context")
                },
            }.items():
                raw.setdefault(name + "_" + key, []).append(value)
            aggregates.setdefault(name, []).append(
                np.asarray(
                    generated_byte_metrics(out["logits"], length, slots, mask).accuracy_per_example
                )
            )
        base = outputs["correct"]
        shifted = outputs["shifted_cuts"]
        for key in ("unpooled", "h_hat", "logits"):
            axes = tuple(range(2, base[key].ndim))
            raw.setdefault(key + "_rms_difference", []).append(
                np.asarray(jnp.sqrt(jnp.mean((base[key] - shifted[key]) ** 2, axes)))
            )
        raw.setdefault("semantic_correct", []).append(
            np.asarray(structures["correct"].c_b[start : start + 8])
        )
        raw.setdefault("semantic_shifted", []).append(
            np.asarray(structures["shifted_cuts"].c_b[start : start + 8])
        )
        raw.setdefault("length", []).append(np.asarray(length))
        raw.setdefault("nu", []).append(np.asarray(batch["nu"]))
        variants = {
            "full": base["logits"],
            "direct_noisy_frozen_codec": dec(h, length, codec["decoder"]),
            "direct_clean_frozen_codec": dec(clean["target"]["h0"], length, codec["decoder"]),
            "residual_only": dec(
                DeterministicOps().linear(h, params["h_head"]), length, params["decoder"]
            ),
            "delta_only": dec(
                DeterministicOps().linear(base["pre_head"] - h, params["h_head"]),
                length,
                params["decoder"],
            ),
            "decoder_carrier_reset": local_dec(base["h_hat"], length, params["decoder"]),
            "direct_noisy_codec_carrier_reset": local_dec(h, length, codec["decoder"]),
        }
        for variant in ("no_attention", "zero_context"):
            variants[variant] = run(
                params,
                h,
                batch["nu"],
                structures["correct"].c_b[start : start + 8],
                length,
                variant=variant,
            )["logits"]
        for name, logits in variants.items():
            ablation_rows.setdefault(name, []).append(metrics(logits, length, slots, mask))
        if start == 0:
            parity["local_decoder_first_carrier_max_error"] = float(
                jnp.max(jnp.abs(variants["decoder_carrier_reset"][:, 0] - base["logits"][:, 0]))
            )
            torx = run(
                jax.tree.map(jnp.asarray, state["params"]),
                h,
                batch["nu"],
                structures["correct"].c_b[:8],
                length,
                torx=True,
            )
            parity["torx_mean_logits_max_error"] = float(
                jnp.max(jnp.abs(torx["logits"] - base["logits"]))
            )
            no_proposal = {**params, "proposal": jax.tree.map(jnp.zeros_like, params["proposal"])}
            proposal_out = run(no_proposal, h, batch["nu"], structures["correct"].c_b[:8], length)
            parity["zero_proposal_logits_max_error"] = float(
                jnp.max(jnp.abs(proposal_out["logits"] - base["logits"]))
            )
            changed_h = (
                base["h_hat"].at[:, 2].add(0.1 * jax.random.normal(jax.random.key(1102), (8, 64)))
            )
            delta = dec(changed_h, length, params["decoder"]) - base["logits"]
            parity["decoder_previous_h_perturbation_logit_rms_by_site"] = np.asarray(
                jnp.sqrt(jnp.mean(delta**2, (0, 2, 3)))
            )
            parity["context_spread_across_examples"] = float(
                jnp.max(jnp.abs(base["context"] - base["context"][:1]))
            )
        print(json.dumps({"original_examples": start + 8}), flush=True)
    arrays = {k: np.concatenate(v, 0) for k, v in raw.items()}
    saved = json.loads(
        (ROOT / "results/phase_g/preflight_g01_v1/f571ab40659d/necessity.json").read_text()
    )
    replication = {}
    for name in aggregates:
        accuracy = np.concatenate(aggregates[name])
        nll = arrays[name + "_nll"].sum(1) / arrays[name + "_count"].sum(1)
        error = max(
            np.max(np.abs(accuracy - np.asarray(saved["per_example"][name]["accuracy"]))),
            np.max(np.abs(nll - np.asarray(saved["per_example"][name]["loss"]))),
        )
        replication[name] = float(error)
        if error > 1e-4:
            raise RuntimeError("saved necessity replication mismatch")
    if (
        parity["torx_mean_logits_max_error"] > 1e-6
        or parity["local_decoder_first_carrier_max_error"] > 1e-6
        or parity["zero_proposal_logits_max_error"] != 0
    ):
        raise RuntimeError("diagnostic parity failure")
    reach, shifted = arrays["correct_reach"], arrays["shifted_cuts_reach"]
    lost = reach & ~shifted
    active = arrays["length"] > 0
    later = np.arange(32)[None, None, :] > np.arange(32)[None, :, None]
    lost_later = (lost & later).sum(-1)
    arrays["lost_later_peers"] = lost_later
    boundary = arrays["semantic_correct"]
    blocks = np.cumsum(np.pad(boundary[:, :-1], ((0, 0), (1, 0))), axis=1)
    block_length = (blocks[:, :, None] == blocks[:, None, :]).sum(-1)
    first = np.pad(boundary[:, :-1].astype(bool), ((0, 0), (1, 0)), constant_values=True)
    groups = {
        "all": np.ones((256, 32), bool),
        "lost_later_peer": lost_later > 0,
        "no_lost_later_peer": lost_later == 0,
        "first_in_block": first,
        "later_in_block": ~first,
        "original_uncorrupted": np.zeros((256, 32), bool),
    }
    for value in range(5):
        groups[f"extent_{value}"] = arrays["length"] == value
    for value in range(1, 9):
        groups[f"block_length_{value}"] = block_length == value
    for a, b in ((0, 1), (1, 4), (4, 8), (8, 16), (16, 32)):
        groups[f"absolute_site_{a}_{b}"] = np.broadcast_to(
            (np.arange(32) >= a) & (np.arange(32) < b), (256, 32)
        )
    for a, b in ((0, 1), (1, 4), (4, 8), (8, 32)):
        groups[f"block_ordinal_{a}_{b}"] = (blocks >= a) & (blocks < b)
    for a, b in ((0.025, 0.1), (0.1, 0.175), (0.175, 0.250001)):
        groups[f"nu_{a}_{b}"] = np.broadcast_to(
            ((arrays["nu"] >= a) & (arrays["nu"] < b))[:, None], (256, 32)
        )
    stratified = {}
    for name, selection in groups.items():
        stratified[name] = {
            s: group({k: arrays[s + "_" + k] for k in ("nll", "count", "correct")}, selection)
            for s in ("correct", "shifted_cuts")
        }
        seq = np.any(selection & active, axis=1)
        stratified[name]["sequence_support"] = int(seq.sum())
        stratified[name]["sequence_accuracy"] = {
            s: float(arrays[s + "_exact"][seq].mean()) if seq.any() else None
            for s in ("correct", "shifted_cuts")
        }
    counter = counterfactual(teacher, run, dec, params, codec, arrays)
    RAW.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(RAW, allow_pickle=False, **arrays)
    assert sha(CHECKPOINT) == frozen_hash and all(
        sha(p) == digest for p, digest in historic.items()
    )
    summary = {
        "checkpoint_sha256": frozen_hash,
        "codec_sha256": sha(CODEC),
        "parent_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "script_sha256": sha(__file__),
        "protocol_sha256": sha(OUT / "PROTOCOL.md"),
        "python": sys.version,
        "platform": platform.platform(),
        "jax": jax.__version__,
        "backend": jax.default_backend(),
        "replication_max_error": replication,
        "parity": parity,
        "strata": stratified,
        "ablations": {
            k: {field: float(np.mean([v[field] for v in rows])) for field in rows[0]}
            for k, rows in ablation_rows.items()
        },
        "connectivity": {
            "examples_changed": int(np.any(reach != shifted, (1, 2)).sum()),
            "emitting_sites": int(active.sum()),
            "emitting_sites_losing_future_paths": int(((lost_later > 0) & active).sum()),
            "emitted_bytes_losing_future_paths": int((arrays["length"] * (lost_later > 0)).sum()),
            "lost_edges": int(lost.sum()),
            "gained_edges": int((~reach & shifted).sum()),
            "any_earlier_active_path_lost": bool(np.any(lost & ~later & active[:, None, :])),
            "all_query_sites_active": bool(
                np.all(arrays["correct_query"].reshape(256, -1).sum(1) == 32)
            ),
            "semantic_edges_changed": int((boundary != arrays["semantic_shifted"]).sum()),
            "mechanical_edges_changed": int(
                (arrays["correct_mechanical_b"] != arrays["shifted_cuts_mechanical_b"]).sum()
            ),
            "identity_roundtrip": bool(
                np.all(
                    np.take_along_axis(
                        arrays["correct_pack"].reshape(256, -1),
                        arrays["correct_unpool_m"] * 8 + arrays["correct_unpool_k"],
                        axis=1,
                    )
                    == np.arange(32)
                )
            ),
        },
        "rms_difference": {
            k: float(np.mean(v)) for k, v in arrays.items() if k.endswith("rms_difference")
        },
        "counterfactual": counter,
        "raw_path": str(RAW.relative_to(ROOT)),
        "raw_sha256": sha(RAW),
        "historical_files_unchanged": len(historic),
        "checkpoint_and_optimizer_file_unchanged": True,
        "training_updates": 0,
    }
    dump(OUT / "diagnosis.json", summary)


def counterfactual(teacher, run, dec, params, codec, arrays):
    data, q, b, wrong = fixtures()
    outputs, representations, inputs = [], [], []
    results = []
    for fixture in data:
        clean = teacher(fixture["target"], fixture["target_mask"])
        h0 = clean["target"]["h0"]
        length = clean["target"]["teacher"].length
        ids = jnp.arange(16, dtype=jnp.uint32) + 2000
        nu = jnp.full((16,), 0.1)
        h, _ = make_initial_corruption(h0, nu, jax.random.key(1103), ids, optimizer_step=0)
        h = h.at[jnp.arange(16), q].set(0).at[jnp.arange(16), q + 1].set(h0[jnp.arange(16), q + 1])
        representations.append(np.asarray(h0))
        inputs.append(np.asarray(h))
        row = {}
        outs = {}
        for name, edges in (("correct", b), ("excluded_peer", wrong)):
            out = run(params, h, nu, edges, length)
            if not bool(out["valid"]):
                raise RuntimeError("invalid fixture routing")
            value = local_values(
                out["logits"],
                clean["target"]["teacher"].slot_bytes,
                clean["target"]["teacher"].slot_mask,
            )
            select = np.arange(32)[None, :] == np.asarray(q)[:, None]
            row[name] = group(value, select)
            outs[name] = out
        row["clean_codec"] = metrics(
            dec(h0, length, codec["decoder"]),
            length,
            clean["target"]["teacher"].slot_bytes,
            clean["target"]["teacher"].slot_mask,
        )
        results.append(row)
        outputs.append(outs)
    prefix = np.arange(32)[None, :] < np.asarray(q)[:, None]
    repr_diff = representations[0] - representations[1]
    stats = {
        "paired_fixtures": 16,
        "seed": 1101,
        "query_indices": np.asarray(q),
        "encoder_prefix_max_difference": float(np.max(np.abs(repr_diff[prefix]))),
        "encoder_later_rms_difference": float(np.sqrt(np.mean(repr_diff[~prefix] ** 2))),
        "input_prefix_max_difference": float(np.max(np.abs((inputs[0] - inputs[1])[prefix]))),
        "rows": results,
    }
    for name in ("correct", "excluded_peer"):
        a, c = outputs[0][name], outputs[1][name]
        stats[name] = {
            "target_h_rms_counterfactual_change": float(
                jnp.sqrt(
                    jnp.mean((a["h_hat"][jnp.arange(16), q] - c["h_hat"][jnp.arange(16), q]) ** 2)
                )
            ),
            "target_logits_rms_counterfactual_change": float(
                jnp.sqrt(
                    jnp.mean((a["logits"][jnp.arange(16), q] - c["logits"][jnp.arange(16), q]) ** 2)
                )
            ),
            "target_peer_reachable": np.asarray(a["reach"][jnp.arange(16), q, q + 1]),
        }
        arrays["fixture_" + name + "_target_logits_A"] = np.asarray(a["logits"][jnp.arange(16), q])
        arrays["fixture_" + name + "_target_logits_B"] = np.asarray(c["logits"][jnp.arange(16), q])
    if (
        stats["encoder_prefix_max_difference"] > 1e-6
        or stats["excluded_peer"]["target_logits_rms_counterfactual_change"] > 1e-6
    ):
        raise RuntimeError("unexpected counterfactual label bypass; inspect before interpretation")
    return stats


if __name__ == "__main__":
    main()
