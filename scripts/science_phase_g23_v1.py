"""Preregistered G3 test opening, paired interventions and axis decisions."""

import json

import jax
import jax.numpy as jnp
import numpy as np

import run_phase_g23_v1 as g
from preflight_phase_g23_v1 import negatives, gradient_variance
from adze_t.phase_g11_task import make_examples
from adze_t.phase_g11_records import sha
from adze_t.objectives import adamw_init
from adze_t.training import stochastic_model_update_mask


def test_data(codec):
    selection = g.G3 / "SELECTION.json"
    record = json.loads(selection.read_text())
    if record["identity"] != g.audit() or any(
        x["selected"] is None for x in record["selected"].values()
    ):
        raise RuntimeError("no complete frozen validation selection")
    from adze_t.phase_g2_storage import RollingJournal

    for seed in (0, 1, 2):
        for arm in ("A", "B", "C"):
            journal = RollingJournal(
                g.RUN / f"seed{seed}/{arm}", record["identity"] + f":seed{seed}:{arm}"
            )
            state = journal.recover()
            if state is None or state["step"] != record["cap"]:
                raise RuntimeError("scientific evaluation before complete matched training cap")
    g.dump(
        g.G3 / "TEST_OPENING.json",
        {
            "identity": g.audit(),
            "selection_sha256": sha(selection),
            "selection_complete_before_generation": True,
            "dataset_seed": 1142,
            "global_id_offset": 20000,
        },
    )
    data = make_examples(1142, 16, id_offset=20000)
    binding = json.loads(g.IDENTITY.read_text())["binding"]
    if g.old.v.dataset_hash(data) != binding["datasets"]["ordered_sha256"]["protected_test"]:
        raise RuntimeError("protected split mismatch")
    encode = jax.jit(g.old.v.encode_fn)
    source = []
    target = []
    for start in range(0, 912, 32):
        d = g.old.v.take(data, jnp.arange(start, min(start + 32, 912)))
        source.append(np.asarray(encode(codec, d["source"], d["source_mask"])))
        target.append(np.asarray(encode(codec, d["target"], d["target_mask"])))
    return {
        **data,
        "source_h": jnp.asarray(np.concatenate(source)),
        "target_h": jnp.asarray(np.concatenate(target)),
        "slots": jnp.pad(data["target"], ((0, 0), (0, 64))).reshape(-1, 32, 4),
        "slot_mask": jnp.arange(4)[None, None, :] < data["length"][:, :, None],
    }


def seed_interval(values):
    x = np.asarray(values)
    width = 4.302652729911275 * x.std(ddof=1) / np.sqrt(3)
    return {
        "mean": float(x.mean()),
        "sd": float(x.std(ddof=1)),
        "range": [float(x.min()), float(x.max())],
        "ci95": [float(x.mean() - width), float(x.mean() + width)],
        "n_training_seeds": 3,
        "df": 2,
    }


def contrast(a, b, data, name, mask=None):
    # a,b: three complete paired per-example dictionaries (already root-averaged).
    if mask is None:
        mask = np.ones(len(data["q"]), bool)
    if not np.any(mask):
        return {"outcome": "UNRESOLVED", "support": 0}
    lengths = np.asarray(data["length"].sum(1))[mask]
    result = {}
    for metric in ("query_accuracy", "query_loss", "query_nll"):
        av = np.stack([x[metric][mask] for x in a])
        bv = np.stack([x[metric][mask] for x in b])
        effects = (av - bv).mean(1)
        if metric != "query_accuracy" and np.any(bv.mean(1) <= 0):
            return {
                "outcome": "UNRESOLVED",
                "support": int(mask.sum()),
                "reason": "Relative loss/NLL undefined at zero comparator; no invented percentage effect",
            }
        relative = av.mean(1) / bv.mean(1) - 1 if np.all(bv.mean(1) > 0) else None
        result[metric] = {
            "example_bootstrap": g.bootstrap(av.mean(0), bv.mean(0), lengths, name + ":" + metric),
            "training_seed_absolute": seed_interval(effects),
            "training_seed_relative": seed_interval(relative) if relative is not None else None,
            "seed_deltas": effects.tolist(),
            "seed_relative": relative.tolist() if relative is not None else None,
        }
    acc = result["query_accuracy"]
    loss = result["query_loss"]
    nll = result["query_nll"]
    ac = acc["example_bootstrap"]
    lc = loss["example_bootstrap"]
    nc = nll["example_bootstrap"]
    benefit = (
        ac["delta"] >= 0.01
        and lc["relative"] <= -0.02
        and nc["relative"] <= -0.02
        and ac["ci95"][0] > 0
        and lc["ci95"][1] < 0
        and nc["ci95"][1] < 0
        and acc["training_seed_absolute"]["ci95"][0] > 0
        and loss["training_seed_relative"]["ci95"][1] < 0
        and nll["training_seed_relative"]["ci95"][1] < 0
    )
    negative = (
        ac["delta"] <= -0.01
        and lc["relative"] >= 0.02
        and nc["relative"] >= 0.02
        and ac["ci95"][1] < 0
        and lc["ci95"][0] > 0
        and nc["ci95"][0] > 0
        and acc["training_seed_absolute"]["ci95"][1] < 0
        and loss["training_seed_relative"]["ci95"][0] > 0
        and nll["training_seed_relative"]["ci95"][0] > 0
    )
    positive_count = sum(x > 0 for x in acc["seed_deltas"])
    negative_count = sum(x < 0 for x in acc["seed_deltas"])
    no_material_reverse = all(
        not (da <= -0.01 and dl >= 0.02)
        for da, dl in zip(acc["seed_deltas"], loss["seed_relative"], strict=True)
    )
    no_material_gain = all(
        not (da >= 0.01 and dl <= -0.02)
        for da, dl in zip(acc["seed_deltas"], loss["seed_relative"], strict=True)
    )
    equivalent = (
        ac["ci95"][0] > -0.01
        and ac["ci95"][1] < 0.01
        and acc["training_seed_absolute"]["ci95"][0] > -0.01
        and acc["training_seed_absolute"]["ci95"][1] < 0.01
    )
    for metric in (loss, nll):
        ci = metric["example_bootstrap"]["relative_ci95"]
        seedci = metric["training_seed_relative"]["ci95"]
        equivalent = (
            equivalent
            and ci is not None
            and ci[0] > -0.02
            and ci[1] < 0.02
            and seedci[0] > -0.02
            and seedci[1] < 0.02
        )
    outcome = (
        "BENEFIT"
        if benefit and positive_count >= 2 and no_material_reverse
        else (
            "NEGATIVE"
            if negative and negative_count >= 2 and no_material_gain
            else ("NEUTRAL" if equivalent else "UNRESOLVED")
        )
    )
    return {"outcome": outcome, "support": int(mask.sum()), "metrics": result}


def strata(data, rows):
    q = np.asarray(data["q"])
    peer = np.asarray(data["peer"])
    length = np.asarray(data["length"].sum(1))
    b = np.asarray(data["boundary"])
    groups = {
        "neutral_reaches": q // 8 == peer // 8,
        "gap": peer - q,
        "length": length,
        "partial": length % 4 != 0,
        "competitor": (
            (b == 1) & (np.asarray(data["length"]) > 0) & (np.arange(32)[None] != peer[:, None])
        ).any(1),
        "block_ordinal": np.sum((np.arange(32)[None] < q[:, None]) & (b == 1), 1),
        "query_block_length": peer - q + 1,
        "preceding_cut": np.where(q > 0, b[np.arange(len(q)), np.maximum(q - 1, 0)], 1),
    }
    return {
        name: {
            str(value): {
                "support": int(np.sum(index == value)),
                "summary": g.summary({k: x[index == value] for k, x in rows.items()}),
            }
            for value in np.unique(index)
        }
        for name, index in groups.items()
    }


def run(identity, codec, val, train=None):
    decision = json.loads((g.OUT / "G2_DECISION.json").read_text())
    if not decision["G3_eligible"]:
        raise RuntimeError("G3 remains gated off")
    selection = json.loads((g.G3 / "SELECTION.json").read_text())
    data = test_data(codec)
    all_rows = {}
    validation_rows = {}
    models = {}
    diagnostics = {}
    initial = g.parent()
    mask = stochastic_model_update_mask(initial)
    frozen = g.frozen_fingerprint(initial, (adamw_init(initial), adamw_init(initial)), mask)
    for seed in (0, 1, 2):
        for arm in ("A", "B", "C"):
            chosen = selection["selected"][f"seed{seed}:{arm}"]["selected"]
            if chosen is None:
                raise RuntimeError("missing eligible selected model")
            path = g.ROOT / chosen["path"]
            if sha(path) != chosen["sha256"]:
                raise RuntimeError("selected checkpoint changed")
            state = g.old.load_pickle(path)
            if state["step"] != chosen["step"] or state["campaign_identity"] != identity:
                raise RuntimeError("selected optimizer/protocol mismatch")
            params = state["params"]
            if g.frozen_fingerprint(params, state["moments"], mask) != frozen:
                raise RuntimeError("selected checkpoint frozen-parameter/moment mutation")
            step = state["step"]
            g.log(f"G3 seed{seed} {arm} selected{step}: fixed full-test interventions")
            prefix = f"seed{seed}:{arm}"
            val_cases = {}
            for control in ("policy", "cut", "neutral", "shifted", "overconnected"):
                val_cases[control] = g.evaluate(
                    params,
                    val,
                    seed=seed,
                    arm=arm,
                    step=step,
                    split="validation",
                    mode="argmax",
                    control=control,
                )
            validation_rows[prefix] = val_cases
            cases = {}
            for mode, controls in (
                ("teacher", ("policy", "cut")),
                (
                    "argmax",
                    (
                        "policy",
                        "cut",
                        "neutral",
                        "shifted",
                        "overconnected",
                        "teacher_boundary",
                        "no_attention",
                        "direct_decoder",
                    ),
                ),
            ):
                for control in controls:
                    rows = g.evaluate(
                        params,
                        data,
                        seed=seed,
                        arm=arm,
                        step=step,
                        split="test",
                        mode=mode,
                        control=control,
                    )
                    cases[mode + ":" + control] = rows
            samples = {control: [] for control in ("policy", "cut", "neutral")}
            for root in range(16):
                for control in samples:
                    samples[control].append(
                        g.evaluate(
                            params,
                            data,
                            seed=seed,
                            arm=arm,
                            step=step,
                            split="test",
                            mode="sample",
                            root=root,
                            control=control,
                        )
                    )
            for control, rows in samples.items():
                cases["sample:" + control] = g.mean_roots(rows)
            all_rows[prefix] = cases
            diagnostics[prefix] = {
                "checkpoint": chosen,
                "selected_validation": {
                    name: {"summary": g.summary(rows), "strata": strata(val, rows)}
                    for name, rows in val_cases.items()
                },
                "model_tree": g.tree_hash(params),
                "summaries": {case: g.summary(r) for case, r in cases.items()},
                "strata": {case: strata(data, r) for case, r in cases.items()},
                "sample_root_summary": {
                    control: {
                        metric: {
                            "mean": float(np.mean([np.mean(r[metric]) for r in roots])),
                            "sd": float(np.std([np.mean(r[metric]) for r in roots], ddof=1)),
                            "range": [
                                float(np.min([np.mean(r[metric]) for r in roots])),
                                float(np.max([np.mean(r[metric]) for r in roots])),
                            ],
                        }
                        for metric in (
                            "query_accuracy",
                            "query_loss",
                            "query_nll",
                            "length_accuracy",
                            "peer_reach",
                        )
                    }
                    for control, roots in samples.items()
                },
            }
            models[prefix] = {
                "model_tree": g.tree_hash(params),
                "optimizer_step": step,
                "payload_sha256": chosen["sha256"],
            }
            # Fixed validation controls are not new selection data.
            negatives(params, codec, val, "selected_" + prefix.replace(":", "_"))
            if arm == "C":
                if train is None:
                    raise RuntimeError("fixed training diagnostic batch absent")
                gradient_variance(params, train, seed, "selected")
            g.dump(g.G3 / f"{prefix.replace(':', '_')}_test.json", diagnostics[prefix])
    functions = {}
    for arm, mode in (("B", "argmax"), ("C", "argmax"), ("C", "sample")):
        records = []
        for seed in (0, 1, 2):
            cases = all_rows[f"seed{seed}:{arm}"]
            records.append(
                g.functional(
                    cases[mode + ":policy"], cases[mode + ":cut"], data, f"G3:{seed}:{arm}:{mode}"
                )
            )
        a = [all_rows[f"seed{s}:{arm}"][mode + ":policy"] for s in (0, 1, 2)]
        b = [all_rows[f"seed{s}:{arm}"][mode + ":cut"] for s in (0, 1, 2)]
        aggregate = g.functional(
            g.mean_roots(a), g.mean_roots(b), data, "G3:aggregate:" + arm + ":" + mode
        )
        seed_acc = seed_interval([r["paired"]["query_accuracy"]["delta"] for r in records])
        seed_loss = seed_interval([r["paired"]["query_loss"]["delta"] for r in records])
        no_reverse = all(
            not (
                r["paired"]["query_accuracy"]["delta"] <= -0.01
                and r["paired"]["query_loss"]["relative"] >= 0.02
            )
            for r in records
        )
        passed = (
            sum(r["passed"] for r in records) >= 2
            and aggregate["passed"]
            and seed_acc["ci95"][0] > 0
            and seed_loss["ci95"][1] < 0
            and no_reverse
        )
        functions[arm + ":" + mode] = {
            "outcome": "BENEFIT" if passed else "UNRESOLVED",
            "functional": passed,
            "per_seed": records,
            "aggregate": aggregate,
            "training_seed_accuracy": seed_acc,
            "training_seed_query_loss": seed_loss,
        }
    mask = (np.asarray(data["q"]) // 8) != (np.asarray(data["peer"]) // 8)
    semantic = {}
    for control in ("neutral", "shifted", "cut", "overconnected"):
        a = [all_rows[f"seed{s}:B"]["argmax:policy"] for s in (0, 1, 2)]
        b = [all_rows[f"seed{s}:B"]["argmax:" + control] for s in (0, 1, 2)]
        semantic[control] = {
            "overall": contrast(a, b, data, "semantic:" + control),
            "neutral_discordant": contrast(a, b, data, "semantic:" + control + ":discordant", mask),
        }
    semantic_outcome = "UNRESOLVED"
    if all(semantic[c]["overall"]["outcome"] == "BENEFIT" for c in ("neutral", "shifted")):
        semantic_outcome = "BENEFIT"
    elif all(
        semantic[c]["neutral_discordant"]["outcome"] == "BENEFIT" for c in ("neutral", "shifted")
    ) and all(semantic[c]["overall"]["outcome"] != "NEGATIVE" for c in ("neutral", "shifted")):
        semantic_outcome = "BENEFIT"
    elif all(semantic[c]["overall"]["outcome"] == "NEUTRAL" for c in ("neutral", "shifted")):
        semantic_outcome = "NEUTRAL"
    elif all(semantic[c]["overall"]["outcome"] == "NEGATIVE" for c in ("neutral", "shifted")):
        semantic_outcome = "NEGATIVE"
    sampling = contrast(
        [all_rows[f"seed{s}:C"]["argmax:policy"] for s in (0, 1, 2)],
        [all_rows[f"seed{s}:B"]["argmax:policy"] for s in (0, 1, 2)],
        data,
        "sampling_training",
    )
    validation_mask = np.asarray(val["q"] // 8 != val["peer"] // 8)
    validation_neutral = contrast(
        [validation_rows[f"seed{s}:B"]["policy"] for s in (0, 1, 2)],
        [validation_rows[f"seed{s}:B"]["neutral"] for s in (0, 1, 2)],
        val,
        "selected_validation_neutral",
        validation_mask,
    )
    axis1 = "BENEFIT" if any(x["functional"] for x in functions.values()) else "UNRESOLVED"
    overall = (
        "PHASE_G_GENERATED_STRUCTURE_FUNCTIONAL" if axis1 == "BENEFIT" else "PHASE_G_UNRESOLVED"
    )
    result = {
        "identity": identity,
        "axis1": axis1,
        "axis2": semantic_outcome,
        "axis3": sampling["outcome"],
        "overall": overall,
        "functionality": functions,
        "semantic": semantic,
        "sampling_training": sampling,
        "test_neutral_discordant_support": int(mask.sum()),
        "models": models,
        "selected_validation_neutral_discordant": validation_neutral,
        "selected_validation_intervals_are_descriptive": True,
        "protected_selection_sha256": sha(g.G3 / "SELECTION.json"),
        "scientific_scope": "reference-scale warm-start, fixed approved budget; not scratch, LM, H/I or general-scale superiority",
        "protected_test_evaluated": True,
    }
    g.dump(g.G3 / "DECISION.json", result)


def close(identity):
    g.preserved()
    g2 = (
        json.loads((g.OUT / "G2_DECISION.json").read_text())
        if (g.OUT / "G2_DECISION.json").exists()
        else None
    )
    decision = (
        json.loads((g.G3 / "DECISION.json").read_text())
        if (g.G3 / "DECISION.json").exists()
        else None
    )
    failures = list(g.RUN.glob("failure_*.json"))
    if decision is not None:
        outcome = decision["overall"]
        text = json.dumps(
            {
                k: decision[k]
                for k in (
                    "axis1",
                    "axis2",
                    "axis3",
                    "overall",
                    "test_neutral_discordant_support",
                    "models",
                )
            },
            indent=2,
        )
    else:
        outcome = "PHASE_G_BLOCKED" if failures and g2 is None else "PHASE_G_UNRESOLVED"
        text = json.dumps(
            {
                "G2": g2,
                "G3": "not executed because gating/closure evidence incomplete",
                "failure_records": [str(x.relative_to(g.ROOT)) for x in failures],
            },
            indent=2,
        )
    g.dump(
        g.G3 / "CLOSURE.json",
        {
            "identity": identity,
            "overall": outcome,
            "has_complete_G3_decision": decision is not None,
            "failures_retained": len(failures),
            "historical_preservation_passed": True,
            "H_I_Thermo_untouched": True,
            "S4_not_run": True,
        },
    )
    documents = (
        (g.OUT / "G2_RESULTS.md", "G2 — learned commitment", json.dumps(g2, indent=2)),
        (g.G3 / "G3_RESULTS.md", "G3 — scientific evaluation", text),
        (
            g.ROOT / "results/phase_g/G_FINAL_RESULTS_V1.md",
            "Phase G — current reference-scale closure",
            text,
        ),
    )
    for path, title, body in documents:
        content = f"# {title}\n\n{outcome}\n\n```json\n{body}\n```\n\nSee frozen G2/G3 protocols, selection records, per-seed/root/stratum evidence and validation ledger.\nEngineering correctness, generated functionality, semantic-routing advantage and sampling-training advantage are separate claims.\nNo test-based selection, architecture/task redesign, H/I/Thermo changes or S4.\nPhase H prerequisites: validated generated-structure interface/commit ownership, valid UNKNOWN encoding, repacking and shared-weight causal/global integration; no H science is authorized by this result.\n"
        if path.exists():
            if path.read_text() != content:
                raise RuntimeError("refuse rewriting closure evidence")
        else:
            path.write_text(content)
