"""Host-only integrity audit/reporting of completed, frozen G11 evidence.

No model execution, gradients, optimizer, dataset sampling or decision tuning.
"""

import argparse
import hashlib
import json
from pathlib import Path
import pickle

import jax
import numpy as np

from run_phase_g0 import tree_hash
from adze_t.phase_g11_records import sha, write_json
from adze_t.training import _CODEC_ENCODER_NAMES

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/phase_g/g1_1/teacher_v0"
RUN = ROOT / "results/runs/phase_g/g1_1/teacher_v0/official"


def read(name):
    return json.loads((OUT / name).read_text())


def verify():
    binding = read("protocol_binding.json")
    identity = binding["sha256"]
    assert (
        hashlib.sha256(json.dumps(binding["protocol"], sort_keys=True).encode()).hexdigest()
        == identity
    )
    protocol = binding["protocol"]
    for name, expected in protocol["source"].items():
        path = OUT / "PROSPECTIVE_CLARIFICATION.md" if name == "clarification" else ROOT / name
        assert sha(path) == expected, name
    history = json.loads(
        (
            ROOT / "results/phase_g/preflight_g01_v1/f571ab40659d/historical_baseline.json"
        ).read_text()
    )["files"]
    assert all(sha(p) == h for p, h in history.items())
    for milestone in ("g1", "g1_1"):
        manifest = json.loads((ROOT / f"results/phase_g/{milestone}/MANIFEST.json").read_text())
        assert all(sha(ROOT / p) == h for p, h in manifest["files"].items())
    paths = sorted((RUN / "receipts").glob("*.json"))
    assert len(paths) == 101
    parent = None
    initial = None
    rows = []
    payloads = {}
    mask = None
    for step, path in enumerate(paths):
        record = json.loads(path.read_text())
        payload = RUN / record["payload"]
        assert path.stem == f"{step:03d}" and record["step"] == step and record["completed"] is True
        assert record["protocol"] == identity and record["parent"] == parent
        assert sha(payload) == record["sha256"]
        with payload.open("rb") as stream:
            state = pickle.load(stream)
        assert state["step"] == step and state["protocol"] == identity
        assert all(
            np.isfinite(x).all() for x in jax.tree.leaves((state["params"], state["moments"]))
        )
        if step == 0:
            initial = state
            assert tree_hash(state["params"]) == protocol["initial_tree"]
            assert all(np.count_nonzero(x) == 0 for x in jax.tree.leaves(state["moments"]))

            def enabled(path, _):
                names = [getattr(k, "key", None) for k in path]
                return np.asarray(
                    "rho" not in names
                    and not (names[0] == "encoder" and names[1] in _CODEC_ENCODER_NAMES)
                )

            mask = jax.tree.map_with_path(enabled, state["params"])
            assert tree_hash(mask) == protocol["update_mask"]
        assert initial is not None and mask is not None
        for old, new, enabled_leaf in zip(
            jax.tree.leaves((initial["params"], initial["moments"])),
            jax.tree.leaves((state["params"], state["moments"])),
            jax.tree.leaves((mask, (mask, mask))),
            strict=True,
        ):
            assert old.shape == new.shape and old.dtype == new.dtype
            if not bool(enabled_leaf):
                assert np.array_equal(old, new)
        if step:
            assert record["metrics"]["executed"] == 1
            assert 0 <= record["metrics"]["applied_gradient_norm"] <= 1
            rows.append(
                {
                    "update": step,
                    "protocol": identity,
                    "payload_sha256": record["sha256"],
                    **record["metrics"],
                }
            )
        if step in (0, 10, 25, 50, 100):
            assert read(f"checkpoint_{step:03d}.json") == record
            for case in ("A", "B"):
                evidence = read(f"validation_{step:03d}_{case}.json")
                assert evidence["step"] == step and evidence["protocol"] == identity
                assert evidence["model_tree"] == tree_hash(state["params"])
                assert evidence["global_example_id"] == list(range(10000, 10912))
                assert (
                    evidence["validation_dataset_sha256"]
                    == protocol["datasets"]["ordered_sha256"]["validation"]
                )
            payloads[str(step)] = {
                "path": str(payload.relative_to(ROOT)),
                "sha256": record["sha256"],
                "model_tree": tree_hash(state["params"]),
            }
        parent = record["sha256"]
    # Materialized, deterministic view. Receipts remain the source of truth.
    text = "".join(json.dumps(row, sort_keys=True, allow_nan=False) + "\n" for row in rows)
    path = OUT / "training.jsonl"
    if path.exists():
        assert path.read_text() == text
    else:
        path.write_text(text)
    decision = read("endpoint_decision.json")
    flow = read("endpoint_flow.json")
    assert (
        decision["protocol"] == identity and decision["model_tree"] == payloads["100"]["model_tree"]
    )
    assert all(flow["flow"]["checks"].values()) and all(flow["extended"]["checks"].values())
    assert not decision["protected_test_exposed"]
    result = {
        "protocol": identity,
        "source_commit": read("source_snapshot.json")["commit"],
        "completed_updates": 100,
        "absolute_optimizer_step": 100,
        "verified_receipts": 101,
        "duplicate_or_missing_updates": 0,
        "all_parameter_and_Adam_states_finite": True,
        "initial_Adam_moments_zero": True,
        "frozen_parameter_and_moment_bits_unchanged_all_101_states": True,
        "historical_artifacts_unchanged": len(history),
        "G1_G11_manifests_unchanged": True,
        "frozen_execution_sources_unchanged": True,
        "checkpoints": payloads,
        "gradient_norm_range": [
            min(x["masked_raw_gradient_norm"] for x in rows),
            max(x["masked_raw_gradient_norm"] for x in rows),
        ],
        "reporter_sha256": sha(__file__),
    }
    write_json(OUT / "completion_integrity.json", result)
    print(json.dumps(result, indent=2))
    return result


def report(integrity):
    decision = read("endpoint_decision.json")
    a = read("validation_100_A.json")["summary"]
    b = read("validation_100_B.json")["summary"]
    base = read("validation_000_A.json")["summary"]
    controls = read("endpoint_controls.json")
    flow = read("endpoint_flow.json")
    lines = [
        "# G.1.1 — Authorized bounded teacher100 results",
        "",
        "## Outcome",
        "",
        (
            "All frozen bounded teacher-necessity criteria pass."
            if decision["bounded_necessity_pass"]
            else "Frozen bounded endpoint is a healthy nonpass/unresolved result."
        ),
        "Exactly100 new updates; endpoint100 fixed before execution. No extension or checkpoint selection. Protected-test examples/metrics remain unexposed.",
        "",
        "This establishes bounded trained use of future-peer attention under teacher-committed structure only. It does NOT establish superiority to fixed-neutral/overconnected routing, learned semantic commits, sampled score-gradient trainability, language modelling, convergence, or Phase-G completion.",
        "",
        "## Fixed endpoint and paired uncertainty",
        "",
        "| Metric | A correct teacher | B valid singleton cut |",
        "|---|---:|---:|",
        f"| Query payload byte accuracy | {100 * a['query_accuracy']:.6f}% | {100 * b['query_accuracy']:.6f}% |",
        f"| Query payload NLL (full256 vocabulary) | {a['query_nll']:.9f} | {b['query_nll']:.9f} |",
        f"| All3 query payload bytes exact | {100 * a['query_exact']:.6f}% | {100 * b['query_exact']:.6f}% |",
        f"| Whole packet byte accuracy | {100 * a['byte_accuracy']:.6f}% | {100 * b['byte_accuracy']:.6f}% |",
        f"| Whole packet exact | {100 * a['byte_exact']:.6f}% | {100 * b['byte_exact']:.6f}% |",
        "",
        f"A−B accuracy: {100 * decision['accuracy']['delta']:.6f}pp; paired95% CI {np.asarray(decision['accuracy']['delta_ci95']) * 100}pp.",
        f"NLL A−B: {decision['nll']['delta']:.9f}; paired95% CI {decision['nll']['delta_ci95']}.",
        f"Relative NLL change: {100 * decision['nll']['relative']:.6f}%; CI {np.asarray(decision['nll']['relative_ci95']) * 100}%.",
        "2000 whole-example paired percentile replicates, resampled16 per each57 length strata; seeds1161/1162, controlcode0. Query3 bytes are not independently resampled. NLL relative change is ratio of aggregate means.",
        "",
        "## Both nonquery guards (prospective AND)",
        "",
        f"Outside-query-carrier accuracy: {100 * base['nonquery_accuracy']:.6f}% → {100 * a['nonquery_accuracy']:.6f}% ({100 * (a['nonquery_accuracy'] - base['nonquery_accuracy']):+.6f}pp); required no loss >2pp.",
        f"Nonquery NLL: {base['nonquery_nll']:.9f} → {a['nonquery_nll']:.9f} ({100 * (a['nonquery_nll'] / base['nonquery_nll'] - 1):+.6f}%); required increase <=5%.",
        "Equal-example means are primary as declared before any new-task gradient. Byte-weighted and complementary nonquery definitions remain separately recorded, never substituted into the guard.",
        "",
        "## Checkpoint curves",
        "",
        "| Update | A query accuracy | A query NLL | B query accuracy | B query NLL | A nonquery accuracy | A whole packet accuracy |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for step in (0, 10, 25, 50, 100):
        x = read(f"validation_{step:03d}_A.json")["summary"]
        y = read(f"validation_{step:03d}_B.json")["summary"]
        lines.append(
            f"| {step} | {100 * x['query_accuracy']:.4f}% | {x['query_nll']:.6f} | {100 * y['query_accuracy']:.4f}% | {y['query_nll']:.6f} | {100 * x['nonquery_accuracy']:.4f}% | {100 * x['byte_accuracy']:.4f}% |"
        )
    lines += [
        "",
        "Nonquery accuracy declines from95.51% at50 to91.37% at100: a visible retrieval/reconstruction tradeoff, not hidden by the easy update0 guard. Whole-packet exactness is only3.62%; passing this query-focused bounded gate is not general reconstruction success. All100 batch loss components and gradient norms are in training.jsonl; full checkpoint validation is per-example, not just an averaged loss.",
        "",
        "## Negative controls",
        "",
        "| Control | Query accuracy | Query NLL |",
        "|---|---:|---:|",
    ]
    for name, result in controls.items():
        x = result["summary"]
        lines.append(f"| {name} | {100 * x['query_accuracy']:.6f}% | {x['query_nll']:.9f} |")
    lines += [
        "",
        "C erases peer before encoding; D_substitute replaces it with independent content; D_outside changes nuisance signatures; E_model/E_codec are causal direct decoding with oracle extents, not generation. C_post and no-attention controls are also tested in the fixed64-example flow audit.",
        f"Unavailable-information counterlabel responses: {flow['flow']['maximum_label_responses']}. All B/C/C_post/D_substitute/E/no-attention query-logit responses are exactly0. Full28-symbol negative slices remain invariant and score1/28 under the separately labelled valid-domain argmax. Model accuracy above always uses all256 logits.",
        "All correct query→peer paths are reachable and all cut paths excluded on the frozen64 examples. Pack/unpool identity, separate dynamic capacity cuts, overflow rejection, oracle injection, eager/JIT and deterministic/public-Torx parity pass. See endpoint_flow.json for full numerical traces; CIs for other controls are in endpoint_controls.json and cannot override A−B.",
        "",
        "## Prescribed stratification",
        "",
        "All57 lengths, query→peer gaps, outside/full outside competitors, fixed-neutral reachability and final-partial packets are in endpoint_strata.json with counts. Full/partial nonquery byte supports and summed losses/correct counts are retained per example. Queries are always full extent: partial-query support0, no invented partial-query quality score.",
        "",
        "| Stratum | Value | Count | A accuracy | B accuracy | A NLL | B NLL |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, groups in read("endpoint_strata.json")["strata"].items():
        for value, x in groups.items():
            lines.append(
                f"| {name} | {value} | {x['support']} | {100 * x['A']['query_accuracy']:.4f}% | {100 * x['B']['query_accuracy']:.4f}% | {x['A']['query_nll']:.6f} | {x['B']['query_nll']:.6f} |"
            )
    lines += [
        "",
        "## Initialization, health and provenance",
        "",
        f"Training source commit `{integrity['source_commit']}`; protocol binding `{integrity['protocol']}`. Frozen original protocol SHA35cbd1fed3b0490b73b83693c7246b01d5d41057295046b95d4022e514d8e29c. Prospective conjunctive clarification and both implementation preflights are preserved.",
        "F1 parent SHA520d3723d3a3187aa08ff3f467f434d43b579cdc4b0ce667727dfad96e83cf57; G01 external codec SHA213d71be58b4e3a0eb685330ff8f20ad0d2574184d59060890baa4a2eed586e7. Initial full tree84ab204d1cea2a1833cbbdeadc7993879ad29a3c4c354d4960c221c951daaeeb. Exact overlay means, fresh Adam0; no prior optimizer inheritance.",
        "Both shared-parameter S1 applications use condition0 and identical observation h: neutral predictor → teacher commit → replay. NOT S2 denoising. L4/Q3/S1/R0; lambda0; AdamW3e-4/wd.01/clip1/batch32; frozen declared loss weights. External codec and target encoder/rho remain unchanged; the output decoder is explicitly trainable.",
        f"New-task first raw full-model gradient norm {read('first_gradient.json')['raw_gradient_norm']:.9f}; clipped1. Prospective update finite and discarded. Official masked raw norm range {integrity['gradient_norm_range']}. All101 parameter/Adam states finite; frozen parameters AND moments bitwise unchanged at every durable state. {integrity['historical_artifacts_unchanged']} protected historical hashes and G1/G11 manifests still match.",
        "Per-example b/ell predictions, boundary counts/F1, per-class extent supports/correct counts, exact predicted structure, emission length, h-MSE, byte NLL/exactness and missing/extra counts are retained in each validation file. Teacher commitment correctness is oracle control, not a learned-commit claim. structural_churn denotes neutral→teacher changed-site fraction per replay, NOT temporal learned-policy churn.",
        "",
        "## Checkpoint payload identities (full means/Adam m/v)",
        "",
        "| Update / absolute Adam step | Payload SHA256 |",
        "|---:|---|",
    ]
    for step, p in integrity["checkpoints"].items():
        lines.append(f"| {step} | `{p['sha256']}` |")
    lines += [
        "",
        "completion_integrity.json retains exact paths and parameter-tree identities. Immutable journal receipts form a verified contiguous parent-hash chain,0..100; training.jsonl is a derived view. No failed attempt is a counted official update.",
        "",
        "## Execution correction and limitations",
        "",
        "The first endpoint driver failed on an eager NumPy/JAX indexing boundary after recovery; it did not change any state or fail a scientific criterion. ENDPOINT_ATTEMPT_1_FAILURE.json preserves it. A separately committed wrapper losslessly converts arrays and calls unchanged endpoint logic after validating source/parents/runtime and parameter bits. endpoint_array_conversion.json binds its exact source hash. No data, roots, thresholds, weights or update count changed; no extra training or selective rerun occurred.",
        "The first engineering preflight was also preserved before a pre-training reporting/identity hardening revision. Both trial updates are NOT COUNTED, never parents of official training. No hidden extension or rejected-task restart.",
        "",
        "## Validation and reproduction",
        "",
        "PYTHONPATH=src:scripts python scripts/run_phase_g11_teacher.py --stage preflight and --stage train executed under4689293 (hard cap100). This historical source snapshot is bound in protocol_binding.json; later reporting files must not be confused with a changed training protocol. Current evaluation reproduction: PYTHONPATH=src:scripts python scripts/evaluate_phase_g11_teacher.py. It verifies the frozen execution sources/checkpoint, uses no optimizer and does not expose protected test.",
        "Focused tests, whole-project regular/slow suites, format/lint/types/boundaries and explicit private-Torx scan are recorded in VALIDATION.json and validation_logs/. Check exact counts/status there; historical counts are not substituted.",
        "No main merge/push. Accepted F/G histories preserved. No argmax/sampled training, full G campaign, H/I science, codec/LM training or S4. Further experiments require separate authorization. STOP.",
        "",
    ]
    path = OUT / "TEACHER100_RESULTS.md"
    text = "\n".join(lines)
    if path.exists():
        assert path.read_text() == text
    else:
        path.write_text(text)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", action="store_true")
    args = parser.parse_args()
    integrity = verify()
    if args.report:
        report(integrity)


if __name__ == "__main__":
    main()
