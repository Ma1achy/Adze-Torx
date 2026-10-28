"""Validate G.1.1 information sets; intentionally no optimizer/training code."""

import hashlib
import json
from pathlib import Path
import pickle
import subprocess

import jax
import jax.numpy as jnp
import numpy as np

from run_phase_g import config
from run_phase_g0 import tree_hash
from adze_t.backends.deterministic import DeterministicOps
from adze_t.backends.mapping import torx_means_to_deterministic
from adze_t.backends.torx import TorxOps, TorxOperatorConfig
from adze_t.decoder import apply_decoder
from adze_t.denoise import make_sanitized_target_analysis, as_sanitized_model_target_analysis
from adze_t.masking import draft_attention_mask
from adze_t.model import apply_model, apply_clean_target_teacher
from adze_t.phase_g import neutral_structure, generated_byte_metrics
from adze_t.phase_g0 import dynamic_capacity_routing, with_frozen_g0_codec
from adze_t.phase_g11_task import (
    TASK_VERSION,
    make_examples,
    symbolic_structure,
    symbolic_target,
    relabel,
    observation_control,
    cut_after_query,
)
from adze_t.packing import pack_values
from adze_t.unpool import unpool_values
from adze_t.state import CommittedStructure

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/phase_g/g1_1"
CFG = config()
CODEC = ROOT / "results/runs/phase_g/g0_1/v1/frozen_codec.pkl"
F1 = ROOT / "results/runs/phase_f/f1/denoise_v1/one_step/init0_stoch0.pkl"
ARM = ROOT / "results/runs/phase_g/preflight_g01_v1/teacher/update_100.pkl"
EXPECTED = {
    CODEC: "213d71be58b4e3a0eb685330ff8f20ad0d2574184d59060890baa4a2eed586e7",
    F1: "520d3723d3a3187aa08ff3f467f434d43b579cdc4b0ce667727dfad96e83cf57",
    ARM: "203926286c4f991b316e9f9d8cb8bee529861da5124f374279b82d143decd220",
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, value):
    if path.exists():
        raise RuntimeError(f"refusing overwrite: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(value, default=lambda x: np.asarray(x).tolist(), indent=2, allow_nan=False)
        + "\n"
    )
    tmp.replace(path)


class FlowOps(DeterministicOps):
    def __init__(self, no_attention=False):
        self.no_attention = no_attention

    def linear(self, x, params, *, name="linear"):
        y = super().linear(x, params, name=name)
        if self.no_attention and name.startswith("dit.block_") and name.endswith(".o"):
            y = jnp.zeros_like(y)
        return y


def forward(params, h, prompt, b, ell, *, no_attention=False, torx=False, cfg=CFG):
    """Only observation h/prompt and explicit committed control enter the model."""
    route = dynamic_capacity_routing(
        CommittedStructure(b, ell > 0, ell),
        M_max=cfg.packing.M_max,
        K=cfg.packing.K,
        max_extent=cfg.carrier.L_max,
    ).routing
    neutral = neutral_structure(h.shape[0], cfg)
    assert neutral.length is not None
    safe = make_sanitized_target_analysis(
        neutral.c_b,
        neutral.length,
        config=cfg,
        target_width=min(64, cfg.carrier.C * cfg.carrier.L_max),
    )
    ops = (
        TorxOps.create(
            jax.random.key(1146),
            config=TorxOperatorConfig(operator_stochasticity=True, lambda_op=0.0),
        )
        if torx
        else FlowOps(no_attention)
    )

    def execute(_):
        out = apply_model(
            params,
            prompt,
            jnp.ones_like(prompt, bool),
            safe.target_bytes_placeholder,
            safe.target_mask_placeholder,
            config=cfg,
            ops=ops,
            target_analysis=as_sanitized_model_target_analysis(safe),
            carrier_h_input=h,
            noise_level=0.0,
            actual_s_index=0,
            denoise_condition_index=0,
            committed_c_b=route.structure.c_b,
            committed_length=route.structure.length,
            committed_activity=route.structure.activity,
            observed_b=neutral.c_b,
            observed_l=neutral.length,
            routing_metadata=route.metadata,
        )
        return out["carrier"], out["byte_logits"], out["context_global"], out["proposal"][0]

    def skip(_):
        return (
            jnp.zeros_like(h),
            jnp.zeros((*h.shape[:2], cfg.carrier.L_max, 256), h.dtype),
            jnp.zeros((len(h), cfg.model.d_ctx), h.dtype),
            jnp.zeros_like(h),
        )

    h_hat, logits, context, proposal = jax.lax.cond(route.executed, execute, skip, operand=None)
    m = route.metadata
    positions = jnp.maximum(m.carrier_to_m * cfg.packing.K + m.carrier_to_k, 0)
    mask = draft_attention_mask(m.block_id, m.query_mask, m.kv_mask)
    reach = mask[jnp.arange(len(h))[:, None, None], positions[:, :, None], positions[:, None, :]]
    return {
        "executed": route.executed,
        "h_hat": h_hat,
        "logits": logits,
        "context": context,
        "proposal": proposal,
        "reach": reach,
        "pack": m.packed_to_carrier,
        "m": m.carrier_to_m,
        "k": m.carrier_to_k,
    }


def take(data, indices):
    return jax.tree.map(lambda x: x[indices], data)


def dataset_hash(data):
    h = hashlib.sha256()
    for key in sorted(data):
        value = np.asarray(data[key])
        h.update(key.encode())
        h.update(str((value.shape, value.dtype.str)).encode())
        h.update(value.tobytes())
    return h.hexdigest()


def obs_hashes(data):
    return [hashlib.sha256(np.asarray(x).tobytes()).hexdigest() for x in data["source"]]


def query(logits, q):
    return logits[jnp.arange(len(q)), q, 1:]


def response(a, b):
    d = np.asarray(a) - np.asarray(b)
    return {"maximum": float(np.max(np.abs(d))), "rms": float(np.sqrt(np.mean(d * d)))}


def local_metric(logits, target, q):
    label = target.reshape(-1, 16, 4)[jnp.arange(len(q)), q, 1:]
    nll = -jnp.take_along_axis(jax.nn.log_softmax(logits), label[..., None], -1)[..., 0]
    correct = jnp.argmax(logits, -1) == label
    return {
        "accuracy": float(jnp.mean(correct)),
        "nll": float(jnp.mean(nll)),
        "exact": float(jnp.mean(jnp.all(correct, -1))),
    }


def encode_fn(codec, ids, mask):
    out = apply_clean_target_teacher(codec, ids, mask, config=CFG)
    return out["target"]["h0"]


def decoder(h, ell, p):
    return apply_decoder(h, ell, p, CFG, DeterministicOps())[0]


def codec_check(codec, data, encode, decode):
    accum = {
        k: []
        for k in (
            "accuracy",
            "nll",
            "exact",
            "count",
            "roles",
            "active",
            "pred_l",
            "true_l",
            "pred_b",
            "true_b",
            "retrieved",
            "peer_found",
        )
    }
    for start in range(0, len(data["q"]), 8):
        d = take(data, jnp.arange(start, start + 8))
        ht = encode(codec, d["target"], d["target_mask"])
        hs = encode(codec, d["source"], d["source_mask"])
        ell = d["length"]
        truth_slots = jnp.pad(d["target"], ((0, 0), (0, 64))).reshape(-1, 32, 4)
        true_mask = jnp.arange(4)[None, None] < ell[:, :, None]
        logits = decode(ht, ell, codec["decoder"])
        metric = generated_byte_metrics(logits, ell, truth_slots, true_mask)
        pl = jnp.argmax(DeterministicOps().categorical_logits(hs, codec["encoder"]["target_l"]), -1)
        source_logits = decode(hs, pl, codec["decoder"])
        source_pred = jnp.argmax(source_logits, -1)
        pb = ((source_pred[:, :, 0] == 1) & (pl > 0)).astype(jnp.int32).at[:, -1].set(1)
        roles = jnp.argmax(logits[:, :, 0], -1) == truth_slots[:, :, 0]
        candidate = (
            (source_pred[:, :, 0] == 1) & (pl > 0) & (jnp.arange(32)[None] > d["q"][:, None])
        )
        peer = jnp.argmax(candidate, -1)
        retrieved = source_pred[jnp.arange(8), peer, 1:] == truth_slots[jnp.arange(8), d["q"], 1:]
        values = (
            metric.accuracy_per_example,
            metric.overlap_nll_per_example,
            metric.exact_per_example,
            jnp.sum(ell, 1),
            roles,
            ell > 0,
            pl,
            ell,
            pb,
            d["boundary"],
            retrieved,
            jnp.any(candidate, -1),
        )
        for key, value in zip(accum, values, strict=True):
            accum[key].append(np.asarray(value))
    x = {k: np.concatenate(v) for k, v in accum.items()}
    length_rows = {
        str(n): {
            "examples": int((x["count"] == n).sum()),
            "accuracy": float(x["accuracy"][x["count"] == n].mean()),
            "nll": float(x["nll"][x["count"] == n].mean()),
            "exact": float(x["exact"][x["count"] == n].mean()),
        }
        for n in range(8, 65)
    }
    partial = (x["count"] % 4) > 0
    last = (x["count"] + 3) // 4 - 1
    partial_correct = (
        x["pred_l"][np.arange(len(last)), last] == x["true_l"][np.arange(len(last)), last]
    )
    active = x["active"]
    scores = {
        "byte_accuracy": float(x["accuracy"].mean()),
        "nll": float(x["nll"].mean()),
        "exact_sequence": float(x["exact"].mean()),
        "worst_length_accuracy": min(v["accuracy"] for v in length_rows.values()),
        "role_signature_recovery": float(x["roles"][active].mean()),
        "extent_accuracy": float((x["pred_l"] == x["true_l"]).mean()),
        "exact_total_length": float((x["pred_l"].sum(1) == x["true_l"].sum(1)).mean()),
        "exact_extent_vector": float((x["pred_l"] == x["true_l"]).all(1).mean()),
        "partial_extent_accuracy": float(partial_correct[partial].mean()),
        "active_boundary_accuracy": float((x["pred_b"] == x["true_b"])[active].mean()),
        "exact_boundary_vector": float((x["pred_b"] == x["true_b"]).all(1).mean()),
        "composed_reader_query_payload_accuracy": float(x["retrieved"].mean()),
        "composed_reader_end_found": float(x["peer_found"].mean()),
    }
    class_rows = {
        str(c): {
            "sites": int((x["true_l"] == c).sum()),
            "recall": float((x["pred_l"][x["true_l"] == c] == c).mean()),
        }
        for c in range(5)
    }
    gates = {
        k: scores[k] >= v
        for k, v in {
            "byte_accuracy": 0.99,
            "worst_length_accuracy": 0.98,
            "role_signature_recovery": 0.99,
            "extent_accuracy": 0.995,
            "exact_total_length": 0.99,
            "partial_extent_accuracy": 0.99,
            "active_boundary_accuracy": 0.99,
            "exact_boundary_vector": 0.95,
            "composed_reader_query_payload_accuracy": 0.98,
        }.items()
    }
    return {
        "scores": scores,
        "gates": gates,
        "length_buckets": length_rows,
        "extent_classes": class_rows,
        "note": "composed reader uses predicted extents and decoded ENDs, never teacher peer/length; it is an observability oracle, not an Adze decoder replacement",
    }


def flow_check(params, codec, data, encode, decode, run):
    q = data["q"]
    label = jax.random.randint(jax.random.key(1144), (len(q), 3), 5, 33)
    other = relabel(data, label)
    nuisance = jax.random.randint(jax.random.key(1145), (len(q), 16, 3), 5, 33)
    results = {}
    injection = []
    parity = []
    for start in range(0, len(q), 8):
        ids = jnp.arange(start, start + 8)
        batch = [take(d, ids) for d in (data, other)]
        n = nuisance[ids]
        h = [encode(codec, d["source"], d["source_mask"]) for d in batch]
        query_id, peer = batch[0]["q"], batch[0]["peer"]
        row = jnp.arange(8)
        prefix = jnp.arange(32)[None] < peer[:, None]
        prefix_error = float(jnp.max(jnp.where(prefix[:, :, None], jnp.abs(h[0] - h[1]), 0)))
        for case in ("A", "B", "C", "C_post", "D_outside", "D_substitute", "no_attention"):
            outputs = []
            for v, d in enumerate(batch):
                carrier = h[v]
                if case in ("C", "D_outside", "D_substitute"):
                    source = observation_control(d, case, n)
                    carrier = encode(codec, source, d["source_mask"])
                if case == "C_post":
                    carrier = carrier.at[row, peer].set(0)
                b = cut_after_query(d) if case == "B" else d["boundary"]
                output = run(
                    params,
                    carrier,
                    d["prompt"],
                    b,
                    d["length"],
                    no_attention=case == "no_attention",
                )
                if not bool(output["executed"]) or not bool(
                    jnp.all(jnp.isfinite(output["logits"]))
                ):
                    raise RuntimeError("invalid/nonfinite flow route")
                outputs.append(output)
            entry = {
                "prefix_encoder_error": prefix_error,
                "query_logits_response": response(
                    query(outputs[0]["logits"], query_id), query(outputs[1]["logits"], query_id)
                ),
                "query_h_response": response(
                    outputs[0]["h_hat"][row, query_id], outputs[1]["h_hat"][row, query_id]
                ),
                "context_response": response(outputs[0]["context"], outputs[1]["context"]),
                "proposal_response": response(outputs[0]["proposal"], outputs[1]["proposal"]),
                "peer_reachable": np.asarray(outputs[0]["reach"][row, query_id, peer]),
                "metrics_A": local_metric(
                    query(outputs[0]["logits"], query_id), batch[0]["target"], query_id
                ),
                "metrics_B": local_metric(
                    query(outputs[1]["logits"], query_id), batch[1]["target"], query_id
                ),
            }
            results.setdefault(case, []).append(entry)
        for name, decoder_params in (("E_codec", codec["decoder"]), ("E_model", params["decoder"])):
            a, b = [query(decode(x, batch[0]["length"], decoder_params), query_id) for x in h]
            results.setdefault(name, []).append(
                {
                    "query_logits_response": response(a, b),
                    "metrics_A": local_metric(a, batch[0]["target"], query_id),
                    "metrics_B": local_metric(b, batch[1]["target"], query_id),
                }
            )
        if start == 0:
            clean_query = [
                encode(codec, d["target"], d["target_mask"])[row, query_id] for d in batch
            ]
            neutral_source = observation_control(batch[0], "C", n)
            base = encode(codec, neutral_source, batch[0]["source_mask"])
            for case in ("A", "B", "no_attention", "E_codec", "E_model"):
                outputs = []
                states = []
                for vector in clean_query:
                    hi = base.at[row, peer].set(vector)
                    if case.startswith("E_"):
                        dp = codec["decoder"] if case == "E_codec" else params["decoder"]
                        outputs.append(query(decode(hi, batch[0]["length"], dp), query_id))
                    else:
                        b = cut_after_query(batch[0]) if case == "B" else batch[0]["boundary"]
                        value = run(
                            params,
                            hi,
                            batch[0]["prompt"],
                            b,
                            batch[0]["length"],
                            no_attention=case == "no_attention",
                        )
                        outputs.append(query(value["logits"], query_id))
                        states.append(value["h_hat"][row, query_id])
                per = np.sqrt(np.mean(np.asarray(outputs[0] - outputs[1]) ** 2, (1, 2)))
                injection.append(
                    {
                        "case": case,
                        "response": response(*outputs),
                        "per_example_logit_rms": per,
                        "h_response": response(*states) if states else None,
                    }
                )
            for case in ("A", "B", "C"):
                carrier = h[0] if case != "C" else base
                b = cut_after_query(batch[0]) if case == "B" else batch[0]["boundary"]
                eager = forward(params, carrier, batch[0]["prompt"], b, batch[0]["length"])
                compiled = run(params, carrier, batch[0]["prompt"], b, batch[0]["length"])
                parity.append(
                    {
                        "case": case,
                        "error": response(eager["logits"], compiled["logits"])["maximum"],
                    }
                )
    maximum = {
        name: max(r["query_logits_response"]["maximum"] for r in rows)
        for name, rows in results.items()
    }
    checks = {
        "A_peer_reachable": all(np.all(r["peer_reachable"]) for r in results["A"]),
        "B_peer_excluded": all(not np.any(r["peer_reachable"]) for r in results["B"]),
        "encoder_prefix_invariant": all(r["prefix_encoder_error"] <= 1e-6 for r in results["A"]),
        "negative_invariance": all(
            maximum[name] <= 1e-6
            for name in ("B", "C", "C_post", "D_substitute", "E_codec", "E_model", "no_attention")
        ),
        "oracle_positive_signal": all(
            np.all(r["per_example_logit_rms"] > 1e-6) for r in injection if r["case"] == "A"
        ),
        "oracle_blocked_invariance": all(
            r["response"]["maximum"] <= 1e-6 for r in injection if r["case"] != "A"
        ),
        "eager_jit": all(r["error"] <= 1e-5 for r in parity),
    }
    return {
        "checks": checks,
        "maximum_label_responses": maximum,
        "rows": results,
        "oracle_injection": injection,
        "eager_jit": parity,
    }


def main():
    for path, h in EXPECTED.items():
        if sha(path) != h:
            raise RuntimeError("checkpoint identity mismatch")
    history = json.loads(
        (
            ROOT / "results/phase_g/preflight_g01_v1/f571ab40659d/historical_baseline.json"
        ).read_text()
    )["files"]
    assert all(sha(p) == h for p, h in history.items())
    with CODEC.open("rb") as stream:
        codec = jax.tree.map(jnp.asarray, pickle.load(stream))
    with F1.open("rb") as stream:
        f1 = pickle.load(stream)
    with ARM.open("rb") as stream:
        arm = pickle.load(stream)
    initial = with_frozen_g0_codec(f1["params"], codec)
    initial_hash = tree_hash(initial)
    assert initial_hash == "84ab204d1cea2a1833cbbdeadc7993879ad29a3c4c354d4960c221c951daaeeb"
    models = {
        "fresh_F1_G01": jax.tree.map(jnp.asarray, torx_means_to_deterministic(initial)),
        "teacher100": jax.tree.map(jnp.asarray, torx_means_to_deterministic(arm["params"])),
    }
    datasets = {
        "train": make_examples(1140, 32),
        "validation": make_examples(1141, 16, id_offset=10000),
        "protected_test": make_examples(1142, 16, id_offset=20000),
    }
    sets = {k: set(obs_hashes(v)) for k, v in datasets.items()}
    overlap = {
        a + "_" + b: len(sets[a] & sets[b])
        for a, b in (
            ("train", "validation"),
            ("train", "protected_test"),
            ("validation", "protected_test"),
        )
    }
    grammar = {}
    for name, d in datasets.items():
        b, ell = symbolic_structure(d["source"])
        grammar[name] = bool(
            jnp.array_equal(b, d["boundary"])
            & jnp.array_equal(ell, d["length"])
            & jnp.array_equal(symbolic_target(d["source"], d["q"]), d["target"])
        )
    if any(overlap.values()) or not all(grammar.values()):
        raise RuntimeError("dataset contract failure")
    dump(
        OUT / "dataset_identity.json",
        {
            "version": TASK_VERSION,
            "ordered_sha256": {k: dataset_hash(v) for k, v in datasets.items()},
            "counts": {k: len(v["q"]) for k, v in datasets.items()},
            "overlap": overlap,
            "symbolic_identifiability": grammar,
            "protected_test_metrics_exposed": False,
        },
    )
    encode = jax.jit(encode_fn)
    decode = jax.jit(decoder)
    run = jax.jit(forward, static_argnames=("no_attention", "torx", "cfg"))
    compatibility = codec_check(codec, datasets["validation"], encode, decode)
    dump(OUT / "codec_identifiability.json", compatibility)
    selected = jax.random.permutation(jax.random.key(1143), 912)[:64]
    flow_data = take(datasets["validation"], selected)
    dump(
        OUT / "flow_selection.json",
        {
            "ids": flow_data["global_example_id"],
            "seed": 1143,
            "selection": "performance-independent permutation",
        },
    )
    flows = {}
    for name, p in models.items():
        flows[name] = flow_check(p, codec, flow_data, encode, decode, run)
        dump(OUT / ("flow_" + name + ".json"), flows[name])
        print(json.dumps({"model": name, "flow_checks": flows[name]["checks"]}), flush=True)
    other_checks = extended_checks(models, initial, codec, flow_data, encode, decode, run)
    dump(OUT / "extended_checks.json", other_checks)
    assert all(sha(p) == h for p, h in history.items()) and all(
        sha(p) == h for p, h in EXPECTED.items()
    )
    contract = all(all(v["checks"].values()) for v in flows.values()) and other_checks["passed"]
    verdict = (
        "G11_TASK_VALIDATED"
        if contract and all(compatibility["gates"].values())
        else "G11_TASK_UNRESOLVED"
        if contract
        else "G11_TASK_INVALID"
    )
    dump(
        OUT / "VERDICT.json",
        {
            "label": verdict,
            "flow_contract_passed": contract,
            "codec_and_readout_gates_passed": all(compatibility["gates"].values()),
            "source_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip(),
            "validator_sha256": sha(__file__),
            "task_source_sha256": sha(ROOT / "src/adze_t/phase_g11_task.py"),
            "protocol_sha256": sha(OUT / "VALIDATION_PROTOCOL.md"),
            "initial_tree": initial_hash,
            "historical_files_unchanged": len(history),
            "new_optimizer_updates": 0,
            "test_metrics_exposed": False,
            "bounded_teacher_training_executed": False,
        },
    )


def extended_checks(models, torx_params, codec, data, encode, decode, run):
    checks = {}
    evidence = {}
    d = take(data, jnp.arange(8))
    h = encode(codec, d["source"], d["source_mask"])
    a = run(models["fresh_F1_G01"], h, d["prompt"], d["boundary"], d["length"])
    t = run(
        jax.tree.map(jnp.asarray, torx_params),
        h,
        d["prompt"],
        d["boundary"],
        d["length"],
        torx=True,
    )
    evidence["torx_parity_error"] = response(a["logits"], t["logits"])["maximum"]
    checks["torx_parity"] = evidence["torx_parity_error"] <= 1e-6
    identity = jnp.broadcast_to(jnp.arange(32)[None, :, None], (8, 32, 1))
    route = dynamic_capacity_routing(
        CommittedStructure(d["boundary"], d["length"] > 0, d["length"]), M_max=32, K=8, max_extent=4
    )
    checks["identity"] = bool(
        jnp.array_equal(
            unpool_values(
                pack_values(identity, route.routing.metadata), route.routing.metadata, C=32
            ),
            identity,
        )
    )
    evidence["forced_capacity_cuts"] = int(jnp.sum(route.capacity_cut))
    semantic = CommittedStructure(
        jnp.zeros((1, 32), jnp.int32).at[:, -1].set(1),
        jnp.ones((1, 32), bool),
        jnp.full((1, 32), 4, jnp.int32),
    )
    stress = dynamic_capacity_routing(semantic, M_max=32, K=8, max_extent=4)
    compiled = jax.jit(lambda s: dynamic_capacity_routing(s, M_max=32, K=8, max_extent=4))(semantic)
    checks["capacity_eager_jit"] = bool(
        jax.tree_util.tree_all(jax.tree.map(jnp.array_equal, stress, compiled))
    )
    checks["capacity_separate"] = bool(
        jnp.array_equal(stress.semantic.c_b, semantic.c_b)
        & jnp.array_equal(jnp.where(stress.capacity_cut)[1], jnp.asarray([7, 15, 23]))
    )
    m = stress.routing.metadata
    pos = m.carrier_to_m * 8 + m.carrier_to_k
    mask = draft_attention_mask(m.block_id, m.query_mask, m.kv_mask)
    checks["forced_cut_can_block_future"] = bool(
        ~mask[0, pos[0, 0], pos[0, 9]] & mask[0, pos[0, 8], pos[0, 9]]
    )
    overflow = dynamic_capacity_routing(semantic, M_max=1, K=8, max_extent=4)
    checks["capacity_overflow_rejected"] = not bool(overflow.routing.executed)
    enum = []
    gaps = np.asarray(data["peer"] - data["q"])
    for index in (int(np.flatnonzero(gaps == 1)[0]), int(np.argmax(gaps))):
        base = take(data, jnp.full((28,), index))
        payload = base["source"].reshape(28, 16, 4)[jnp.arange(28), base["peer"], 1:]
        payload = payload.at[:, 0].set(jnp.arange(5, 33))
        varied = relabel(base, payload)
        hv = encode(codec, varied["source"], varied["source_mask"])
        peer_h = np.asarray(hv[jnp.arange(28), varied["peer"]])
        distinct = len({x.tobytes() for x in peer_h}) == 28
        controls = {}
        nuisance = jnp.broadcast_to(
            jax.random.randint(jax.random.fold_in(jax.random.key(1145), index), (1, 16, 3), 5, 33),
            (28, 16, 3),
        )
        for case in ("B", "C", "C_post", "D_substitute"):
            hc = hv
            if case in ("C", "D_substitute"):
                hc = encode(
                    codec, observation_control(varied, case, nuisance), varied["source_mask"]
                )
            if case == "C_post":
                hc = hc.at[jnp.arange(28), varied["peer"]].set(0)
            bc = cut_after_query(varied) if case == "B" else varied["boundary"]
            outputs = run(models["fresh_F1_G01"], hc, varied["prompt"], bc, varied["length"])
            controls[case] = np.asarray(query(outputs["logits"], varied["q"]))[:, 0]
        for case, dp in (
            ("E_codec", codec["decoder"]),
            ("E_model", models["fresh_F1_G01"]["decoder"]),
        ):
            controls[case] = np.asarray(query(decode(hv, varied["length"], dp), varied["q"]))[:, 0]
        control_scores = {}
        for case, logits in controls.items():
            control_scores[case] = {
                "max_logit_variation": float(np.max(np.abs(logits - logits[:1]))),
                "restricted_domain_accuracy": float(
                    np.mean(np.argmax(logits[:, 5:33], -1) + 5 == np.arange(5, 33))
                ),
                "raw_full_vocab_accuracy": float(
                    np.mean(np.argmax(logits, -1) == np.arange(5, 33))
                ),
            }
        enum.append(
            {
                "global_id": int(data["global_example_id"][index]),
                "query": int(data["q"][index]),
                "peer": int(data["peer"][index]),
                "distinct_peer_latents": distinct,
                "controls": control_scores,
            }
        )
    checks["alphabet_enumeration"] = all(
        x["distinct_peer_latents"]
        and all(
            c["max_logit_variation"] <= 1e-6
            and abs(c["restricted_domain_accuracy"] - 1 / 28) < 1e-12
            for c in x["controls"].values()
        )
        for x in enum
    )
    evidence["alphabet_enumeration"] = enum
    evidence["checks"] = checks
    evidence["passed"] = all(checks.values())
    return evidence


if __name__ == "__main__":
    main()
