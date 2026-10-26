"""Independent expectation versus production full-site capacity sampling."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from .generated_structure import (
    StructureEstimatorOwner,
    StructureKeyContext,
    commit_predicted_structure_core,
    full_commit_masks,
    score_corrected_structure_loss,
)
from .phase_g0 import dynamic_capacity_routing
from .phase_g_oracle import OracleActions, descendant_fixture, enumerate_actions, fixture_parameters
from .state import CommittedStructure, PredictedStructure


def independent_routes(edges: list[list[int]], k: int) -> jax.Array:
    rows = []
    for row in edges:
        run, out = 0, []
        for bit in row:
            run += 1
            bit = int(bit == 1 or run == k)
            out.append(bit)
            if bit:
                run = 0
        rows.append(out)
    return jnp.asarray(rows, jnp.int32)


def conformance(parameter_name: str, loss_name: str, *, seed: int = 975) -> dict:
    parameters = fixture_parameters(parameter_name)
    actions = enumerate_actions()
    routed = OracleActions(independent_routes(actions.boundary.tolist(), 2), actions.extent)
    descendant = descendant_fixture(loss_name)

    def probabilities(p):
        b = jax.nn.log_softmax(p.boundary_logits)
        ell = jax.nn.log_softmax(p.extent_logits)
        return jnp.exp(
            b[jnp.arange(2)[None], actions.boundary[:, :2]].sum(1)
            + ell[jnp.arange(3)[None], actions.extent].sum(1)
        )

    exact = jax.grad(lambda p: jnp.sum(probabilities(p) * descendant(p, routed)))(parameters)
    n = 4096
    previous = CommittedStructure(
        jnp.broadcast_to(jnp.asarray([0, 0, 1]), (n, 3)),
        jnp.ones((n, 3), bool),
        jnp.ones((n, 3), jnp.int32),
    )

    def sampled_loss(p, root):
        predicted = PredictedStructure(
            jnp.broadcast_to(jnp.pad(p.boundary_logits, ((0, 1), (0, 0))), (n, 3, 2)),
            jnp.broadcast_to(p.extent_logits, (n, 3, 3)),
        )
        commit = commit_predicted_structure_core(
            previous,
            predicted,
            full_commit_masks((n, 3)),
            policy="sample",
            max_extent=2,
            key_context=StructureKeyContext(root, 0, jnp.arange(n), 0, 0, 0),
            temperature=1.0,
            estimator_owner=StructureEstimatorOwner.LOCAL_SCORE_BRIDGE,
            allow_reactivation=True,
        )
        route = dynamic_capacity_routing(commit.committed, M_max=3, K=2, max_extent=2)
        losses = descendant(p, OracleActions(route.routing.structure.c_b, commit.sampled.length))
        return jnp.mean(score_corrected_structure_loss(losses, commit))

    estimate = jax.jit(jax.grad(sampled_loss))
    estimates = [
        estimate(parameters, jax.random.fold_in(jax.random.key(seed), r)) for r in range(16)
    ]
    leaves = jax.tree.map(lambda *x: jnp.stack(x), *estimates)
    flattened = []
    for values, target in zip(jax.tree.leaves(leaves), jax.tree.leaves(exact), strict=True):
        mean, sd = values.mean(0), values.std(0, ddof=1)
        se = sd / 4
        error = jnp.abs(mean - target)
        ok = jnp.asarray(jnp.where(se == 0, error <= 1e-6, error <= 4 * se))
        flattened.append(
            {
                "exact": target.tolist(),
                "mean": mean.tolist(),
                "root_sd": sd.tolist(),
                "root_se": se.tolist(),
                "absolute_bias": error.tolist(),
                "accepted": bool(jnp.all(ok)),
            }
        )
    return {
        "parameter_fixture": parameter_name,
        "loss_fixture": loss_name,
        "seed": seed,
        "roots": 16,
        "samples_per_root": n,
        "mass": float(probabilities(parameters).sum()),
        "action_count": 108,
        "accepted": all(x["accepted"] for x in flattened),
        "components": flattened,
    }
