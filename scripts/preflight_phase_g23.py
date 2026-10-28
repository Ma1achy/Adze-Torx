"""Actual G2 full-model preflight; all trial updates discarded."""

import json
from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np

import run_phase_g23 as g
from adze_t.backends.mapping import torx_means_to_deterministic
from adze_t.phase_g2 import inference, objective, train_step, named_key
from adze_t.phase_g2_oracle import conformance
from adze_t.phase_g11_task import relabel, observation_control
from adze_t.phase_g11_records import finite, sha
from adze_t.objectives import adamw_init
from adze_t.phase_g2_storage import RollingJournal
from adze_t.training import stochastic_model_update_mask
from adze_t.state import CommittedStructure, PredictedStructure
from adze_t.generated_structure import commit_predicted_structure_core, full_commit_masks
from adze_t.phase_g0 import dynamic_capacity_routing
from adze_t.packing import pack_values
from adze_t.unpool import unpool_values
from adze_t.phase_g2 import aligned_metrics


def norm(tree):
    return float(jnp.sqrt(sum(jnp.sum(x * x) for x in jax.tree.leaves(tree))))


def gradient_variance(params, train, seed, stage):
    """Sixteen stochastic structural occurrences on a fixed32-example batch."""
    ids = jax.random.permutation(named_key("gradient_diagnostic_examples"), len(train["q"]))[:32]
    batch = g.old.v.take(train, ids)
    op = named_key("operator", seed)
    fn = jax.jit(
        jax.grad(lambda p, key: objective(p, batch, 0, key, op, policy="C", config=g.CFG)[0])
    )
    mask = jax.tree.leaves(stochastic_model_update_mask(params))
    paths = [jax.tree_util.keystr(path) for path, _ in jax.tree_util.tree_leaves_with_path(params)]
    means = None
    squares = None
    norms = []
    for root in range(16):
        grad = fn(params, named_key("gradient_diagnostic", seed, root))
        if not finite(grad):
            raise RuntimeError("nonfinite structural gradient diagnostic")
        leaves = [
            np.asarray(x, dtype=np.float64)
            if bool(e)
            else np.zeros_like(np.asarray(x), dtype=np.float64)
            for x, e in zip(jax.tree.leaves(grad), mask, strict=True)
        ]
        norms.append(float(np.sqrt(sum(np.sum(x * x) for x in leaves))))
        if means is None:
            means = [x.copy() for x in leaves]
            squares = [np.zeros_like(x) for x in leaves]
        else:
            assert squares is not None
            for n, x in enumerate(leaves):
                delta = x - means[n]
                means[n] += delta / (root + 1)
                squares[n] += delta * (x - means[n])
    assert means is not None and squares is not None
    groups = {}
    for name in ("global", "dit", "b_head", "l_head", "decoder"):
        indices = [
            i
            for i, path in enumerate(paths)
            if name == "global" or path.startswith("['" + name + "']")
        ]
        groups[name] = {
            "mean_gradient_norm": float(np.sqrt(sum(np.sum(means[i] ** 2) for i in indices))),
            "summed_parameter_variance": float(sum(np.sum(squares[i] / 15) for i in indices)),
        }
    result = {
        "model_tree": g.tree_hash(params),
        "seed": seed,
        "roots": 16,
        "global_ids": batch["global_example_id"],
        "norms": norms,
        "groups": groups,
        "optimizer_updates": 0,
    }
    g.dump(g.OUT / f"{stage}_gradient_variance_seed{seed}.json", result)
    return result


def negatives(params, codec, val, stage):
    ids = jax.random.permutation(jax.random.key(1143), len(val["q"]))[:64]
    data = g.old.v.take(val, ids)
    payload = jax.random.randint(jax.random.key(1144), (64, 3), 5, 33)
    other = relabel(data, payload)
    encode = jax.jit(g.old.v.encode_fn)
    h0 = data["source_h"]
    h1 = encode(codec, other["source"], other["source_mask"])
    key = named_key("evaluation_structure", 0, 0)
    op = named_key("operator", 0)
    det = jax.tree.map(jnp.asarray, torx_means_to_deterministic(params))
    run = jax.jit(g.old.v.forward, static_argnames=("no_attention", "torx"))
    rows = {}
    for policy in ("B", "C"):
        fn = g.INFER[policy]
        base = fn(params, h0, data["prompt"], data["global_example_id"], key, op)
        alternate = fn(params, h1, data["prompt"], data["global_example_id"], key, op)
        if not bool(base.executed) or not bool(alternate.executed):
            raise RuntimeError("invalid generated flow")
        b = g.case_boundaries("cut", base.commit.committed.c_b, data)
        ell = base.commit.committed.length
        assert ell is not None
        a = run(det, h0, data["prompt"], b, ell)
        z = run(det, h1, data["prompt"], b, ell)
        row = jnp.arange(64)
        q = data["q"]
        error = float(jnp.max(jnp.abs(a["logits"][row, q, 1:] - z["logits"][row, q, 1:])))
        if error > 1e-6:
            raise RuntimeError("fixed-action physical cut label invariance failed")
        intervention = float(jnp.sqrt(jnp.mean((base.logits - a["logits"]) ** 2)))
        if intervention <= 1e-8:
            raise RuntimeError("generated topology has no real downstream effect")
        if bool(jnp.any(a["reach"][row, q, data["peer"]])):
            raise RuntimeError("cut failed to remove physical future-peer access")
        nuisance = jax.random.randint(jax.random.key(1145), (64, 16, 3), 5, 33)
        negative = {}
        for case in ("C", "D_substitute"):
            source0 = observation_control(data, case, nuisance)
            source1 = observation_control(other, case, nuisance)
            if not bool(jnp.array_equal(source0, source1)):
                raise RuntimeError("negative observations not label-independent")
            hs = [encode(codec, x, data["source_mask"]) for x in (source0, source1)]
            outputs = [
                fn(params, h, data["prompt"], data["global_example_id"], key, op) for h in hs
            ]
            delta = float(jnp.max(jnp.abs(outputs[0].logits - outputs[1].logits)))
            if delta > 1e-6:
                raise RuntimeError("whole-policy erased-information negative failed")
            negative[case] = delta
        # Deterministic/public-Torx and eager/JIT at real C32/L4/Q3.
        first = g.old.v.take(data, jnp.arange(8))
        small_h = first["source_h"]
        chosen = inference(
            params,
            small_h,
            first["prompt"],
            first["global_example_id"],
            key,
            op,
            0,
            policy=policy,
            config=g.CFG,
        )
        eager = g.old.v.forward(
            det,
            small_h,
            first["prompt"],
            chosen.commit.committed.c_b,
            chosen.commit.committed.length,
        )
        torx = run(
            params,
            small_h,
            first["prompt"],
            chosen.commit.committed.c_b,
            chosen.commit.committed.length,
            torx=True,
        )
        err = float(jnp.max(jnp.abs(eager["logits"] - torx["logits"])))
        if not bool(
            jnp.array_equal(chosen.carrier_to_m, eager["m"])
            & jnp.array_equal(chosen.carrier_to_k, eager["k"])
        ):
            raise RuntimeError("actual model packing differs from committed generated routing")
        if err > 1e-5:
            raise RuntimeError("actual generated deterministic/Torx parity failed")
        rows[policy] = {
            "fixed_action_physical_cut_response": error,
            "generated_vs_cut_logit_rms": intervention,
            "policy_boundary_logit_response": float(
                jnp.max(jnp.abs(base.b_logits - alternate.b_logits))
            ),
            "policy_extent_logit_response": float(
                jnp.max(jnp.abs(base.l_logits - alternate.l_logits))
            ),
            "whole_policy_negatives": negative,
            "recomputed_policy_b_changes": int(
                jnp.sum(base.commit.committed.c_b != alternate.commit.committed.c_b)
            ),
            "recomputed_policy_extent_changes": int(
                jnp.sum(base.commit.committed.length != alternate.commit.committed.length)
            ),
            "generated_eager_Torx_error": err,
            "global_ids": data["global_example_id"],
        }
    g.dump(g.OUT / f"{stage}_negative_controls.json", rows)
    return rows


def run(initial, codec, identity, train, val):
    destination = g.OUT / "PREFLIGHT.json"
    if destination.exists():
        p = json.loads(destination.read_text())
        if p["identity"] != identity or not p["passed"]:
            raise RuntimeError("incompatible preflight evidence")
        return
    g.preserved()
    neutral_reaches = np.asarray(val["q"] // 8 == val["peer"] // 8)
    if int(neutral_reaches.sum()) != 780 or int((~neutral_reaches).sum()) != 132:
        raise RuntimeError("preregistered neutral reachability strata mismatch")
    for filename in (
        "generated_structure.py",
        "phase_g0.py",
        "phase_g_oracle.py",
        "phase_g0_oracle.py",
        "train/score_bridge.py",
    ):
        original = json.loads((g.old.OUT / "protocol_binding.json").read_text())["protocol"][
            "source"
        ]
        relative = "src/adze_t/" + filename
        if sha(g.ROOT / relative) != original[relative]:
            raise RuntimeError("accepted action/bridge source changed")
    inherited = {}
    for path in (
        g.ROOT / "results/phase_g/gradient_oracle.json",
        g.ROOT / "results/phase_g/g0/capacity_oracle.json",
    ):
        data = json.loads(path.read_text())
        inherited[str(path.relative_to(g.ROOT))] = {
            "sha256": sha(path),
            "status": data["status"],
            "fixtures": len(data["rows"]),
        }
        if data["status"] not in ("G_GRADIENT_ORACLE_PASS", "G0_CAPACITY_ORACLE_PASS"):
            raise RuntimeError("accepted oracle is not a pass")
        if len(data["rows"]) != 21:
            raise RuntimeError("missing accepted oracle fixtures")
    oracle = []
    for name in ("uniform", "asymmetric", "peaked"):
        g.log("G2 actual query-length oracle " + name)
        result = conformance(name)
        oracle.append(result)
        g.dump(g.OUT / f"query_oracle_{name}.json", result)
        if not result["passed"]:
            raise RuntimeError("new-query exact/MC gradient oracle nonpass")
    b = g.batch_for(train, 0, 0)
    moments = (adamw_init(initial), adamw_init(initial))
    mask = stochastic_model_update_mask(initial)
    frozen = g.frozen_fingerprint(initial, moments, mask)
    evidence = {}
    empty = CommittedStructure(
        jnp.zeros_like(b["boundary"]).at[:, -1].set(1),
        jnp.zeros_like(b["length"], bool),
        jnp.zeros_like(b["length"]),
    )
    empty_out = jax.jit(
        lambda p: inference(
            p,
            b["source_h"],
            b["prompt"],
            b["global_example_id"],
            named_key("diagnostic_structure"),
            named_key("operator"),
            0,
            policy="A",
            teacher=empty,
            config=g.CFG,
        )
    )(initial)
    if not bool(empty_out.executed) or not finite(empty_out):
        raise RuntimeError("all-inactive generated emission graph failed")
    empty_metrics = aligned_metrics(
        empty_out.logits, empty.length, b["slots"], b["slot_mask"], b["q"]
    )
    if not finite(empty_metrics) or bool(jnp.any(empty_metrics["emitted_length"] != 0)):
        raise RuntimeError("zero emission not finite/exact")
    predicted = PredictedStructure(
        jnp.zeros((32, 32, 2)),
        jnp.broadcast_to(jnp.asarray([-20.0, -20.0, -20.0, -20.0, 20.0]), (32, 32, 5)),
    )
    react = commit_predicted_structure_core(
        empty,
        predicted,
        full_commit_masks((32, 32)),
        policy="argmax",
        max_extent=4,
        allow_reactivation=True,
    )
    route = dynamic_capacity_routing(react.committed, M_max=32, K=8, max_extent=4).routing
    values = jnp.broadcast_to(jnp.arange(32)[None, :, None], (32, 32, 1))
    if not bool(jnp.all(react.reactivated)) or not bool(
        jnp.array_equal(
            unpool_values(pack_values(values, route.metadata), route.metadata, C=32), values
        )
    ):
        raise RuntimeError("reactivation or persistent identity failed")
    for arm in ("A", "B", "C"):
        g.log("new full-model gradient/trial " + arm)
        key = named_key("training_structure", 0)
        op = named_key("operator", 0)
        fn = jax.jit(
            jax.value_and_grad(
                lambda p: objective(p, b, 0, key, op, policy=arm, config=g.CFG), has_aux=True
            )
        )
        (loss, metrics), grad = fn(initial)
        if not bool(metrics["executed"]) or not finite((loss, metrics, grad)):
            raise RuntimeError("nonfinite initial gradient")
        raw = norm(grad)
        subsystems = {k: norm(v) for k, v in grad.items()}
        connected_difference = None
        if arm == "C":
            detached = jax.jit(
                jax.grad(
                    lambda p: objective(
                        p, b, 0, key, op, policy=arm, config=g.CFG, detach_score=True
                    )[0]
                )
            )(initial)
            connected_difference = norm(jax.tree.map(lambda a, c: a - c, grad, detached))
            if connected_difference <= 1e-8:
                raise RuntimeError("score bridge disconnected from actual descendant")
            # Selected action LP is checked independently using model logits.
            out = inference(
                initial,
                b["source_h"],
                b["prompt"],
                b["global_example_id"],
                key,
                op,
                0,
                policy="C",
                config=g.CFG,
            )
            selected_b = out.commit.committed.c_b
            ell = out.commit.committed.length
            expected = g.old.ce(out.b_logits[:, :-1], selected_b[:, :-1]).sum(1) + g.old.ce(
                out.l_logits, ell
            ).sum(1)
            if not bool(
                jnp.allclose(
                    -expected, out.commit.sampled.descendant_log_prob, atol=1e-5, rtol=1e-6
                )
            ):
                raise RuntimeError("selected LP ownership mismatch")
        update = jax.jit(
            lambda p, m, s, b: train_step(p, m, s, b, key, op, policy=arm, config=g.CFG)
        )
        p, m, step, metric = update(initial, moments, jnp.asarray(0), b)
        if (
            int(step) != 1
            or not finite((p, m, metric))
            or g.frozen_fingerprint(p, m, mask) != frozen
            or g.tree_hash(p) == g.INIT
        ):
            raise RuntimeError("prospective update/frozen state gate")
        trial = RollingJournal(g.RUN / f"preflight/trial_{arm}", identity + ":trial:" + arm)
        state = trial.recover()
        if state is None:
            trial.append(
                {"params": initial, "moments": moments, "step": 0, "identity": trial.identity},
                {"executed": True, "counted": False},
                snapshot=True,
            )
            trial.append(
                {"params": p, "moments": m, "step": 1, "identity": trial.identity},
                {"executed": True, "counted": False},
                snapshot=True,
            )
        recovered = trial.recover()
        assert recovered is not None
        if g.tree_hash((recovered["params"], recovered["moments"])) != g.tree_hash((p, m)):
            raise RuntimeError("trial persistence did not roundtrip")
        if arm == "A":
            bad = {**b, "boundary": b["boundary"].at[:, -1].set(0)}
            invalid = update(initial, moments, jnp.asarray(0), bad)
            if int(invalid[2]) != 0 or g.tree_hash(invalid[:2]) != g.tree_hash((initial, moments)):
                raise RuntimeError("invalid update changed full optimizer state")
            tiny = replace(g.CFG, packing=replace(g.CFG.packing, M_max=1))
            overflow = jax.jit(
                lambda p, m: train_step(p, m, 0, b, key, op, policy="A", config=tiny)
            )(initial, moments)
            if int(overflow[2]) != 0 or g.tree_hash(overflow[:2]) != g.tree_hash(
                (initial, moments)
            ):
                raise RuntimeError("overflow updated full-model optimizer")
        evidence[arm] = {
            "loss": float(loss),
            "raw_gradient_norm": raw,
            "subsystem_norms": subsystems,
            "score_connected_detached_difference": connected_difference,
            "trial_mean_tree": g.tree_hash(p),
            "trial_moment_tree": g.tree_hash(m),
            "frozen_fingerprint": frozen,
            "trial_not_counted": True,
            "metrics": g.summary(metrics),
        }
        del fn, grad, update
    g.dump(g.OUT / "FIRST_GRADIENTS.json", evidence)
    variance = gradient_variance(initial, train, 0, "preflight")
    negative = negatives(initial, codec, val, "preflight")
    g.dump(
        destination,
        {
            "identity": identity,
            "passed": True,
            "accepted_oracles": inherited,
            "new_query_oracle_fixtures": 3,
            "full_model_arms": evidence,
            "negative_controls": negative,
            "counted_updates": 0,
            "gradient_variance": variance,
            "all_inactive_emission": True,
            "reactivation_identity": True,
            "full_model_invalid_and_overflow_rollback": True,
        },
    )
