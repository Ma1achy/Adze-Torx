"""Post-nonpass audit of the SAME frozen peaked MC draws; no gate override.

Does not rerun gradient MC, select a new seed, train a model or alter protocols.
"""

import json
import math

import jax
import jax.numpy as jnp
import numpy as np

import run_phase_g23 as g
from adze_t.generated_structure import (
    CategoricalCommitFactor,
    StructureCommitResult,
    SampledStructure,
    StructureEstimatorOwner,
    StructureKeyContext,
    commit_predicted_structure_core,
    full_commit_masks,
    score_corrected_structure_loss,
)
from adze_t.phase_g2 import named_key
from adze_t.phase_g2_oracle import action_lp, descendant
from adze_t.phase_g_oracle import fixture_parameters, enumerate_actions, OracleActions
from adze_t.phase_g0_oracle import independent_routes
from adze_t.state import CommittedStructure, PredictedStructure


def softmax(x):
    x = np.asarray(x, dtype=np.float64)
    v = np.exp(x - x.max(-1, keepdims=True))
    return v / v.sum(-1, keepdims=True)


def main():
    identity = g.audit()
    saved = json.loads((g.OUT / "query_oracle_peaked.json").read_text())
    p = fixture_parameters("peaked")
    a = enumerate_actions()
    routed = OracleActions(independent_routes(a.boundary.tolist(), 2), a.extent)
    n = 108
    previous = CommittedStructure(
        jnp.broadcast_to(jnp.asarray([0, 0, 1]), (n, 3)),
        jnp.ones((n, 3), bool),
        jnp.ones((n, 3), jnp.int32),
    )

    def production_lp(x):
        fb = CategoricalCommitFactor(x.boundary_logits[0])
        fl = CategoricalCommitFactor(x.extent_logits[0])
        blp = jnp.stack(
            [
                jax.vmap(
                    lambda action: fb.log_probability(
                        {"logits": x.boundary_logits[i]}, action, None, {"temperature": 1.0}
                    )
                )(a.boundary[:, i])
                for i in range(2)
            ],
            1,
        )
        blp = jnp.pad(blp, ((0, 0), (0, 1)))
        llp = jnp.stack(
            [
                jax.vmap(
                    lambda action: fl.log_probability(
                        {"logits": x.extent_logits[i]}, action, None, {"temperature": 1.0}
                    )
                )(a.extent[:, i])
                for i in range(3)
            ],
            1,
        )
        sample = SampledStructure(a.boundary, a.extent, blp, llp, (blp + llp).sum(1))
        return StructureCommitResult(
            previous,
            CommittedStructure(a.boundary, a.extent > 0, a.extent),
            sample,
            jnp.zeros((n, 3), bool),
            jnp.zeros((n, 3), bool),
            jnp.zeros((n, 3), bool),
            jnp.ones(n, bool),
            jnp.zeros(n, jnp.int32),
            None,
            "sample",
            StructureEstimatorOwner.LOCAL_SCORE_BRIDGE,
        )

    probability = jax.lax.stop_gradient(jnp.exp(action_lp(p, a)))
    exact = jax.grad(lambda x: jnp.sum(jnp.exp(action_lp(x, a)) * descendant(x, routed)))(p)
    production = jax.grad(
        lambda x: jnp.sum(
            probability * score_corrected_structure_loss(descendant(x, routed), production_lp(x))
        )
    )(p)
    exact_errors = [float(jnp.max(jnp.abs(x - y))) for x, y in zip(exact, production, strict=True)]
    lp_error = float(
        jnp.max(jnp.abs(production_lp(p).sampled.descendant_log_prob - action_lp(p, a)))
    )
    # Analytic per-action score contributions. Known probabilities allow an
    # independent true-variance diagnostic, NOT a replacement acceptance rule.
    pb = softmax(p.boundary_logits)
    pl = softmax(p.extent_logits)
    ba = np.asarray(a.boundary)
    la = np.asarray(a.extent)
    probabilities = np.prod(pb[np.arange(2)[None], ba[:, :2]], 1) * np.prod(
        pl[np.arange(3)[None], la], 1
    )
    losses = np.asarray(descendant(p, routed), np.float64)
    gb = losses[:, None, None] * (np.eye(2)[ba[:, :2]] - pb[None])
    gl = losses[:, None, None] * (np.eye(3)[la] - pl[None])
    gt = np.asarray(
        jax.jacfwd(lambda theta: descendant(p._replace(direct_theta=theta), routed))(
            p.direct_theta
        ),
        np.float64,
    )
    fields = (gb, gl, gt)
    exact_moments = []
    for field, record in zip(fields, saved["components"], strict=True):
        weighted = probabilities.reshape((108,) + (1,) * (field.ndim - 1))
        mean = (weighted * field).sum(0)
        variance = np.maximum((weighted * (field - mean) ** 2).sum(0), 0)
        known_se = np.sqrt(variance / (16 * 4096))
        error = np.abs(np.asarray(record["mean"]) - mean)
        exact_moments.append(
            {
                "independent_mean": mean,
                "true_per_draw_variance": variance,
                "known_variance_SE_diagnostic": known_se,
                "absolute_error_of_saved_MC_mean": error,
                "error_in_known_SE_diagnostic": np.divide(
                    error, known_se, out=np.zeros_like(error), where=known_se > 0
                ),
            }
        )
    counts = []
    allkeys = []
    empirical = []
    sample_count = 4096
    prev = CommittedStructure(
        jnp.broadcast_to(jnp.asarray([0, 0, 1]), (sample_count, 3)),
        jnp.ones((sample_count, 3), bool),
        jnp.ones((sample_count, 3), jnp.int32),
    )
    predicted = PredictedStructure(
        jnp.broadcast_to(jnp.pad(p.boundary_logits, ((0, 1), (0, 0))), (sample_count, 3, 2)),
        jnp.broadcast_to(p.extent_logits, (sample_count, 3, 3)),
    )
    for root in range(16):
        context = StructureKeyContext(
            named_key("query_oracle:peaked", 0, root),
            0,
            jnp.arange(sample_count, dtype=jnp.uint32),
            0,
            0,
            0,
        )
        commit = commit_predicted_structure_core(
            prev,
            predicted,
            full_commit_masks((sample_count, 3)),
            policy="sample",
            max_extent=2,
            key_context=context,
            temperature=1.0,
            estimator_owner=StructureEstimatorOwner.LOCAL_SCORE_BRIDGE,
            allow_reactivation=True,
        )
        assert commit.committed.length is not None
        b = np.asarray(commit.committed.c_b)
        ell = np.asarray(commit.committed.length)
        indices = (((b[:, 0] * 2 + b[:, 1]) * 3 + ell[:, 0]) * 3 + ell[:, 1]) * 3 + ell[:, 2]
        frequency = np.bincount(indices, minlength=108)
        empirical.append(
            [np.einsum("n,n...->...", frequency / sample_count, field) for field in fields]
        )
        counts.append(np.bincount(ell[:, 0], minlength=3))
        allkeys.extend(
            [
                np.asarray(jax.random.key_data(context.keys(kind, capacity))).reshape(-1, 2)
                for kind, capacity in (("boundary", 2), ("extent", 3))
            ]
        )
    keys = np.concatenate(allkeys)
    total_counts = np.asarray(counts).sum(0)
    rare = float(pl[0, 2])
    observed = int(total_counts[2])
    trials = 65536
    mass = math.exp(trials * math.log1p(-rare))
    cdf = mass
    for k in range(observed):
        mass *= ((trials - k) / (k + 1)) * rare / (1 - rare)
        cdf += mass
    reconstructed = [np.mean([x[k] for x in empirical], 0) for k in range(3)]
    result = {
        "identity": identity,
        "original_gate_passed": False,
        "gate_not_waived": True,
        "counted_G2_updates": 0,
        "new_roots_or_seed_rerolls": 0,
        "same_frozen_draws_reconstructed": True,
        "production_factor_all108_LP_error": lp_error,
        "production_bridge_all108_gradient_max_errors": exact_errors,
        "production_bridge_exact_within_2e_minus6": all(x <= 2e-6 for x in exact_errors),
        "all_stochastic_keys": len(keys),
        "unique_stochastic_keys": len(np.unique(keys, axis=0)),
        "extent_site0_counts_per_root": counts,
        "extent_site0_total_counts": total_counts,
        "extent_site0_class2_probability": rare,
        "class2_expected_count": trials * rare,
        "class2_observed_count": observed,
        "class2_binomial_lower_tail_probability": cdf,
        "moment_diagnostics": exact_moments,
        "saved_MC_reconstruction_max_errors": [
            float(np.max(np.abs(x - np.asarray(row["mean"]))))
            for x, row in zip(reconstructed, saved["components"], strict=True)
        ],
        "interpretation": "Exact production expectation and distinct keys are diagnosed independently. Rare-event undercount can depress the empirical root SE. This does not change or satisfy the frozen four-estimated-root-SE gate; dependent training remains blocked.",
    }
    g.dump(g.OUT / "ORACLE_NONPASS_DIAGNOSIS.json", result)
    print(
        json.dumps(
            {
                k: v
                for k, v in result.items()
                if k not in ("moment_diagnostics", "extent_site0_counts_per_root")
            },
            default=lambda x: np.asarray(x).tolist(),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
