"""Prospective, immutable G2.1 audit/calibration/confirmation. No neural training."""

import argparse
from dataclasses import replace
from functools import partial
import hashlib
import json
from pathlib import Path
import subprocess
import traceback
from typing import Literal, cast

import jax
import jax.numpy as jnp
import numpy as np

import run_phase_g23 as g
from adze_t.generated_structure import (
    CategoricalCommitFactor,
    SampledStructure,
    StructureCommitResult,
    StructureEstimatorOwner,
    StructureKeyContext,
    commit_predicted_structure_core,
    full_commit_masks,
    score_corrected_structure_loss,
)
from adze_t.phase_g0 import dynamic_capacity_routing
from adze_t.phase_g2 import named_key
from adze_t.phase_g2_oracle import descendant
from adze_t.phase_g21 import (
    FIXTURES,
    bernstein_radius,
    component_labels,
    finite_difference,
    from_vector,
    pack_parameters,
    tolerance,
    wilson,
)
from adze_t.phase_g_oracle import OracleActions, fixture_parameters
from adze_t.phase_g11_records import sha
from adze_t.state import CommittedStructure, PredictedStructure


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/phase_g/g2_1"
IDENTITY = OUT / "IDENTITY.json"


def freeze():
    if IDENTITY.exists():
        raise RuntimeError("adjudication identity already frozen")
    identity = g.audit()
    g.preserved()
    manifest = ROOT / "results/phase_g/g3/MANIFEST.json"
    old = json.loads(manifest.read_text())
    for p, h in old["files"].items():
        assert sha(ROOT / p) == h, p
    files = {p: sha(ROOT / p) for p in old["files"]}
    files[str(manifest.relative_to(ROOT))] = sha(manifest)
    for p in (
        "src/adze_t/phase_g21.py",
        "scripts/adjudicate_phase_g21.py",
        "tests/test_phase_g21.py",
        "results/phase_g/g2_1/G2_ORACLE_ADJUDICATION_PROTOCOL.md",
    ):
        files[p] = sha(ROOT / p)
    binding = {
        "base_identity": identity,
        "files": files,
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
    }
    digest = hashlib.sha256(json.dumps(binding, sort_keys=True).encode()).hexdigest()
    g.dump(IDENTITY, {"identity": digest, "binding": binding})
    print("FROZEN", digest, flush=True)


def audit_identity():
    r = json.loads(IDENTITY.read_text())
    assert (
        hashlib.sha256(json.dumps(r["binding"], sort_keys=True).encode()).hexdigest()
        == r["identity"]
    )
    assert g.audit() == r["binding"]["base_identity"]
    for p, h in r["binding"]["files"].items():
        assert sha(ROOT / p) == h, p
    return r["identity"]


def enumerated_commit(p, raw):
    b = jnp.concatenate((jnp.asarray(raw[:, :2]), jnp.ones((108, 1), jnp.int32)), 1)
    ell = jnp.asarray(raw[:, 2:])
    fb, fl = (
        CategoricalCommitFactor(p.boundary_logits[0]),
        CategoricalCommitFactor(p.extent_logits[0]),
    )
    blp = jnp.stack(
        [
            jax.vmap(
                lambda a: fb.log_probability(
                    {"logits": p.boundary_logits[i]}, a, None, {"temperature": 1.0}
                )
            )(b[:, i])
            for i in range(2)
        ],
        1,
    )
    blp = jnp.pad(blp, ((0, 0), (0, 1)))
    llp = jnp.stack(
        [
            jax.vmap(
                lambda a: fl.log_probability(
                    {"logits": p.extent_logits[i]}, a, None, {"temperature": 1.0}
                )
            )(ell[:, i])
            for i in range(3)
        ],
        1,
    )
    state = CommittedStructure(b, ell > 0, ell)
    zeros = jnp.zeros((108, 3), bool)
    return StructureCommitResult(
        state,
        state,
        SampledStructure(b, ell, blp, llp, (blp + llp).sum(1)),
        zeros,
        zeros,
        zeros,
        jnp.ones(108, bool),
        jnp.zeros(108, jnp.int32),
        None,
        "sample",
        StructureEstimatorOwner.LOCAL_SCORE_BRIDGE,
    )


def routed_actions(commit):
    r = dynamic_capacity_routing(commit.committed, M_max=3, K=2, max_extent=2)
    return OracleActions(r.routing.structure.c_b, commit.committed.length), r


def draw(p, key):
    n = 4096
    previous = CommittedStructure(
        jnp.broadcast_to(jnp.asarray([0, 0, 1]), (n, 3)),
        jnp.ones((n, 3), bool),
        jnp.ones((n, 3), jnp.int32),
    )
    pred = PredictedStructure(
        jnp.broadcast_to(jnp.pad(p.boundary_logits, ((0, 1), (0, 0))), (n, 3, 2)),
        jnp.broadcast_to(p.extent_logits, (n, 3, 3)),
    )
    context = StructureKeyContext(key, 0, jnp.arange(n, dtype=jnp.uint32), 0, 0, 0)
    c = commit_predicted_structure_core(
        previous,
        pred,
        full_commit_masks((n, 3)),
        policy="sample",
        max_extent=2,
        key_context=context,
        temperature=1.0,
        estimator_owner=StructureEstimatorOwner.LOCAL_SCORE_BRIDGE,
        allow_reactivation=True,
    )
    actions, route = routed_actions(c)
    loss = descendant(p, actions)
    keys = jnp.concatenate(
        (
            jax.random.key_data(context.keys("boundary", 2)).reshape(-1, 2),
            jax.random.key_data(context.keys("extent", 3)).reshape(-1, 2),
        )
    )
    return score_corrected_structure_loss(loss, c).mean(), (
        c.sampled.boundary,
        c.sampled.length,
        c.sampled.descendant_log_prob,
        c.sampled.boundary_log_prob[:, -1],
        route.routing.executed,
        keys,
    )


DRAW = jax.jit(jax.value_and_grad(draw, has_aux=True))


def run_roots(name, count, namespace, table):
    p = fixture_parameters(name)
    gradients, frequencies, words, allkeys = [], [], [], []
    maximum_lp, maximum_reconstruction = 0.0, 0.0
    for root in range(count):
        key = named_key(namespace + name, 0, root)
        (_, (b, ell, lp, terminal_lp, valid, keys)), gradient = DRAW(p, key)
        b, ell = np.asarray(b), np.asarray(ell)
        assert bool(valid) and np.all(np.asarray(terminal_lp) == 0)
        index = (((b[:, 0] * 2 + b[:, 1]) * 3 + ell[:, 0]) * 3 + ell[:, 1]) * 3 + ell[:, 2]
        histogram = np.bincount(index, minlength=108)
        reconstructed = histogram @ table["gradient"] / 4096
        got = pack_parameters(gradient)
        assert np.all(np.isfinite(got)) and np.all(
            np.abs(got - reconstructed) <= tolerance(reconstructed)
        )
        lp_error = float(np.max(np.abs(np.asarray(lp) - np.log(table["probability"][index]))))
        assert lp_error <= 2e-5
        maximum_lp = max(maximum_lp, lp_error)
        maximum_reconstruction = max(
            maximum_reconstruction, float(np.max(np.abs(got - reconstructed)))
        )
        gradients.append(got)
        frequencies.append(histogram)
        words.append(np.asarray(jax.random.key_data(key)))
        allkeys.append(np.asarray(keys).view(np.uint64).reshape(-1))
        tag = "original" if namespace == "query_oracle:" else "confirmation"
        g.dump(
            OUT / "roots" / tag / name / f"{root:02d}.json",
            {
                "root": root,
                "namespace": namespace + name,
                "root_key": words[-1],
                "gradient": got,
                "action_histogram": histogram,
                "selected_LP_max_error": lp_error,
                "gradient_reconstruction_max_error": float(np.max(np.abs(got - reconstructed))),
                "completed": True,
            },
        )
    return {
        "gradients": np.asarray(gradients),
        "histograms": np.asarray(frequencies),
        "root_keys": words,
        "maximum_selected_LP_error": maximum_lp,
        "maximum_gradient_reconstruction_error": maximum_reconstruction,
    }, np.concatenate(allkeys)


def exact_audit():
    records, mutation_detected, old_keys = {}, np.zeros(6, bool), []
    for name in FIXTURES:
        p = fixture_parameters(name)
        table = from_vector(pack_parameters(p))
        finite = finite_difference(pack_parameters(p))
        assert np.all(np.abs(finite - table["mean"]) <= 2e-8 + 1e-4 * np.abs(table["mean"]))
        c = enumerated_commit(p, table["actions"])
        a, routing = routed_actions(c)
        losses = np.asarray(descendant(p, a))
        assert bool(routing.routing.executed)
        assert np.all(np.abs(losses - table["loss"]) <= tolerance(table["loss"]))
        assert (
            np.max(np.abs(np.asarray(c.sampled.descendant_log_prob) - np.log(table["probability"])))
            <= 2e-5
        )
        probability = jnp.asarray(table["probability"])

        def expected(x, multiplier=1.0, baseline=0.0, omit=None, detach=False):
            commit = enumerated_commit(x, table["actions"])
            act, _ = routed_actions(commit)
            loss = descendant(x, act)
            lp = multiplier * commit.sampled.descendant_log_prob
            if omit == "boundary":
                lp = lp - commit.sampled.boundary_log_prob[:, 0]
            elif omit == "extent":
                lp = lp - commit.sampled.length_log_prob[:, 0]
            if detach:
                loss = jax.lax.stop_gradient(loss)
            commit = replace(commit, sampled=replace(commit.sampled, descendant_log_prob=lp))
            return jnp.sum(
                probability * (score_corrected_structure_loss(loss - baseline, commit) + baseline)
            )

        prod = pack_parameters(jax.grad(expected)(p))
        assert np.all(np.abs(prod - table["mean"]) <= tolerance(table["mean"]))
        for baseline in (0.0, 1.0, float(table["probability"] @ table["loss"])):
            got = pack_parameters(jax.grad(lambda x: expected(x, baseline=baseline))(p))
            assert np.all(np.abs(got - table["mean"]) <= tolerance(table["mean"]))
        for i, multiplier in enumerate((0.0, 2.0, -1.0)):
            wrong = pack_parameters(jax.grad(lambda x: expected(x, multiplier=multiplier))(p))
            mutation_detected[i] |= np.any(np.abs(wrong - table["mean"]) > tolerance(table["mean"]))
        wrong_path = pack_parameters(jax.grad(lambda x: expected(x, detach=True))(p))
        mutation_detected[3] |= np.any(
            np.abs(wrong_path - table["mean"]) > tolerance(table["mean"])
        )
        for i, kind in enumerate(("boundary", "extent"), 4):
            wrong = pack_parameters(jax.grad(lambda x: expected(x, omit=kind))(p))
            mutation_detected[i] |= np.any(np.abs(wrong - table["mean"]) > tolerance(table["mean"]))
        try:
            score_corrected_structure_loss(
                jnp.asarray(losses), replace(c, estimator_owner=StructureEstimatorOwner.NATIVE_TORX)
            )
        except ValueError:
            pass
        else:
            raise AssertionError("double ownership accepted")
        historical, keys = run_roots(name, 16, "query_oracle:", table)
        saved = json.loads((g.OUT / f"query_oracle_{name}.json").read_text())
        saved_mean = np.concatenate(
            [np.asarray(x["mean"]).reshape(-1) for x in saved["components"]]
        )
        roots = historical["gradients"]
        assert np.max(np.abs(roots.mean(0) - saved_mean)) <= 2e-6
        centered = roots - roots.mean(0)
        sd = roots.std(0, ddof=1)
        covariance = table["covariance"]
        den = np.sqrt(np.outer(table["variance"], table["variance"]))
        correlation = np.divide(covariance, den, out=np.zeros_like(covariance), where=den > 0)
        records[name] = {
            **historical,
            "independent_exact": table["mean"],
            "production_exact": prod,
            "finite_difference": finite,
            "probabilities": table["probability"],
            "action_rows": table["actions"],
            "per_draw_covariance": covariance,
            "component_correlation": correlation,
            "empirical_root_covariance": np.cov(roots, rowvar=False),
            "exact_covariance_eigenvalues": np.linalg.eigvalsh(covariance),
            "exact_covariance_rank": int(np.linalg.matrix_rank(covariance)),
            "root_SD": sd,
            "root_SE": sd / 4,
            "root_skew": np.divide((centered**3).mean(0), sd**3, out=np.zeros(14), where=sd > 0),
            "max_root_squared_deviation_fraction": np.divide(
                (centered**2).max(0),
                (centered**2).sum(0),
                out=np.zeros(14),
                where=(centered**2).sum(0) > 0,
            ),
            "per_draw_variance": table["variance"],
            "known_mean_SE_original": np.sqrt(table["variance"] / 65536),
            "empirical_to_true_variance_ratio": np.divide(
                (sd / 4) ** 2,
                table["variance"] / 65536,
                out=np.zeros(14),
                where=table["variance"] > 0,
            ),
            "numerical_loss_max_error": float(np.max(np.abs(losses - table["loss"]))),
        }
        old_keys.append(keys)
    assert np.all(mutation_detected)
    keys = np.concatenate(old_keys)
    assert len(np.unique(keys)) == len(keys)
    return {
        "passed": True,
        "components": component_labels(),
        "fixtures": records,
        "mutation_detection": mutation_detected,
        "historical_sample_key_count": len(keys),
        "original_nonpass_unchanged": True,
    }


@partial(jax.jit, static_argnames=("trials", "roots"))
def count_draws(key, conditional, *, trials, roots):
    remaining = jnp.full((256, roots), trials, jnp.int32)

    def step(n, pair):
        k, p = pair
        draw_count = jax.random.binomial(k, n=n, p=p, shape=n.shape).astype(jnp.int32)
        return n - draw_count, draw_count

    left, counts = jax.lax.scan(step, remaining, (jax.random.split(key, 107), conditional[:-1]))
    return jnp.moveaxis(jnp.concatenate((counts, left[None]), 0), 0, -1)


def calibration():
    joint_old = np.zeros(2048, bool)
    joint_old_independent = np.zeros(2048, bool)
    joint_new = np.zeros(2048, bool)
    rows = {}
    for name in FIXTURES:
        table = from_vector(pack_parameters(fixture_parameters(name)))
        probs = table["probability"]
        conditional = jnp.asarray(probs / np.cumsum(probs[::-1])[::-1])
        exact = table["mean"]
        saved = json.loads((g.OUT / f"query_oracle_{name}.json").read_text())
        old_target = np.concatenate(
            [np.asarray(x["exact"]).reshape(-1) for x in saved["components"]]
        )
        old_bad, old_independent_bad, new_bad, max_z, rares = [], [], [], [], []
        for chunk in range(8):
            old = np.asarray(
                count_draws(
                    named_key("g21_calibration_v1:old:" + name, 0, chunk),
                    conditional,
                    trials=4096,
                    roots=16,
                )
            )
            new = np.asarray(
                count_draws(
                    named_key("g21_calibration_v1:new:" + name, 0, chunk),
                    conditional,
                    trials=262144,
                    roots=1,
                )
            )[:, 0]
            assert (
                np.all(old.sum(-1) == 4096)
                and np.all(new.sum(-1) == 262144)
                and old.min() >= 0
                and new.min() >= 0
            )
            means = old @ table["gradient"] / 4096
            se = means.std(1, ddof=1) / 4
            err = np.abs(means.mean(1) - old_target)
            old_bad.append(err > np.where(se == 0, 1e-6, 4 * se))
            old_independent_bad.append(
                np.abs(means.mean(1) - exact) > np.where(se == 0, 1e-6, 4 * se)
            )
            max_z.append(
                np.divide(err, se, out=np.where(err <= 1e-6, 0.0, np.inf), where=se > 0).max(1)
            )
            radius = bernstein_radius(table["variance"], table["maximum"], 262144)
            new_bad.append(
                np.abs(new @ table["gradient"] / 262144 - exact) > radius + tolerance(exact)
            )
            rares.append(old[:, :, table["actions"][:, 2] == 2].sum((1, 2)))
        ob, nb = np.concatenate(old_bad), np.concatenate(new_bad)
        joint_old |= ob.any(1)
        joint_old_independent |= np.concatenate(old_independent_bad).any(1)
        joint_new |= nb.any(1)
        rows[name] = {
            "old_coordinate_rejections": ob.sum(0),
            "new_coordinate_rejections": nb.sum(0),
            "old_max_studentized_quantiles": [
                float(x) if np.isfinite(x) else "Infinity"
                for x in np.quantile(
                    np.concatenate(max_z), [0, 0.5, 0.9, 0.95, 0.99, 1], method="nearest"
                )
            ],
            "original_budget_rare_extent0_class2_counts": np.concatenate(rares),
        }
    count = int(joint_new.sum())
    passed = count / 2048 <= 0.01 and wilson(count, 2048)[1] <= 0.02
    return {
        "passed": passed,
        "replicates": 2048,
        "family_size": 42,
        "old_FWER": float(joint_old.mean()),
        "old_FWER_Wilson95": wilson(int(joint_old.sum()), 2048),
        "old_FWER_against_independent_float64": float(joint_old_independent.mean()),
        "new_FWER": float(joint_new.mean()),
        "new_FWER_Wilson95": wilson(count, 2048),
        "original_family_rejections": joint_old,
        "new_family_rejections": joint_new,
        "fixtures": rows,
    }


def confirmation():
    rows, keys = {}, []
    audit = json.loads((OUT / "audit.json").read_text())
    all_pass = True
    for name in FIXTURES:
        table = from_vector(pack_parameters(fixture_parameters(name)))
        result, k = run_roots(name, 64, "g21_confirmation_v1:", table)
        means = result["gradients"].mean(0)
        radius = bernstein_radius(table["variance"], table["maximum"], 262144)
        error = np.abs(means - table["mean"])
        accepted = error <= radius + tolerance(table["mean"])
        all_pass &= bool(np.all(accepted))
        rows[name] = {
            **result,
            "exact": table["mean"],
            "mean": means,
            "absolute_error": error,
            "relative_error": error / np.maximum(np.abs(table["mean"]), 1e-12),
            "root_SD": result["gradients"].std(0, ddof=1),
            "empirical_root_SE": result["gradients"].std(0, ddof=1) / 8,
            "known_variance_SE": np.sqrt(table["variance"] / 262144),
            "simultaneous_radius": radius,
            "simultaneous_interval": np.stack(
                (
                    means - radius - tolerance(table["mean"]),
                    means + radius + tolerance(table["mean"]),
                ),
                1,
            ),
            "sign_agreement": np.sign(means) == np.sign(table["mean"]),
            "accepted": accepted,
        }
        keys.append(k)
    original_keys = []
    for name in FIXTURES:
        for root in range(16):
            c = StructureKeyContext(
                named_key("query_oracle:" + name, 0, root),
                0,
                jnp.arange(4096, dtype=jnp.uint32),
                0,
                0,
                0,
            )
            original_keys.extend(
                [
                    np.asarray(
                        jax.random.key_data(c.keys(cast(Literal["boundary", "extent"], kind), n))
                    )
                    .view(np.uint64)
                    .reshape(-1)
                    for kind, n in (("boundary", 2), ("extent", 3))
                ]
            )
    combined = np.concatenate(keys + original_keys)
    assert len(np.unique(combined)) == len(combined)
    assert audit["passed"]
    return {
        "passed": all_pass,
        "family_size": 42,
        "roots_per_fixture": 64,
        "draws_per_root": 4096,
        "total_confirmation_draws": 786432,
        "new_and_original_keys_disjoint_unique": True,
        "combined_occurrence_keys": len(combined),
        "combined_key_SHA256": hashlib.sha256(combined.tobytes()).hexdigest(),
        "fixtures": rows,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        choices=("freeze", "audit", "calibration", "confirmation", "decision"),
        required=True,
    )
    stage = parser.parse_args().stage
    if stage == "freeze":
        freeze()
        return
    identity = audit_identity()
    path = OUT / (stage + ".json")
    if path.exists():
        raise RuntimeError(
            "immutable stage already executed; inspect saved evidence, do not reroll"
        )
    if stage == "decision":
        records = {
            s: json.loads((OUT / (s + ".json")).read_text())
            for s in ("audit", "calibration", "confirmation")
        }
        ok = all(r["passed"] and r["identity"] == identity for r in records.values())
        result = {
            "passed": ok,
            "label": "G21_GRADIENT_ESTIMATOR_CONFIRMED" if ok else "G21_UNRESOLVED",
            "original_G23_V0_status": "PHASE_G_BLOCKED",
            "historical_nonpass_waived": False,
        }
    else:
        result = {"audit": exact_audit, "calibration": calibration, "confirmation": confirmation}[
            stage
        ]()
    g.dump(path, {"identity": identity, **result})
    print(stage, "PASS" if result["passed"] else "NONPASS", flush=True)
    if not result["passed"]:
        raise RuntimeError("adjudication nonpass; stop dependent continuation")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        # Preserve the traceback; never manufacture a passed stage or completed update.
        errors = list(OUT.glob("ERROR_*.json"))
        g.dump(OUT / f"ERROR_{len(errors):03d}.json", {"traceback": traceback.format_exc()})
        raise
