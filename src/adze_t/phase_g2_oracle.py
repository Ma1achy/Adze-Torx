"""Independent exact expectation of the NEW globally aligned query objective."""

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
from .phase_g0_oracle import independent_routes
from .phase_g_oracle import enumerate_actions, fixture_parameters, OracleActions
from .phase_g2 import aligned_metrics, named_key
from .state import CommittedStructure, PredictedStructure


def action_lp(p, actions):
    return jax.nn.log_softmax(p.boundary_logits)[jnp.arange(2)[None], actions.boundary[:, :2]].sum(
        1
    ) + jax.nn.log_softmax(p.extent_logits)[jnp.arange(3)[None], actions.extent].sum(1)


def descendant(p, actions):
    n = len(actions.extent)
    target = jnp.broadcast_to(jnp.asarray([[5, 6], [7, 8], [9, 10]]), (n, 3, 2))
    mask = jnp.broadcast_to(jnp.asarray([[True, True], [True, False], [True, True]]), (n, 3, 2))
    effect = jnp.stack(
        (
            actions.boundary[:, 0],
            actions.boundary[:, 0] - actions.boundary[:, 1],
            actions.boundary[:, 1],
        ),
        1,
    )
    values = (
        p.direct_theta * jnp.asarray([[0.4, -0.2], [0.7, 0.3], [-0.5, 0.9]])[None]
        + 0.6 * effect[:, :, None]
    )
    logits = (
        jnp.zeros((n, 3, 2, 256))
        .at[
            jnp.arange(n)[:, None, None],
            jnp.arange(3)[None, :, None],
            jnp.arange(2)[None, None, :],
            target,
        ]
        .set(values)
    )
    metric = aligned_metrics(logits, actions.extent, target, mask, jnp.zeros(n, jnp.int32))
    return (
        metric["byte_loss"]
        + metric["query_loss"]
        + 0.5 * p.direct_theta**2 * (1 + actions.extent.mean(1))
    )


def conformance(name):
    p = fixture_parameters(name)
    actions = enumerate_actions()
    routed = OracleActions(independent_routes(actions.boundary.tolist(), 2), actions.extent)
    exact = jax.grad(lambda x: jnp.sum(jnp.exp(action_lp(x, actions)) * descendant(x, routed)))(p)
    probabilities = jax.lax.stop_gradient(jnp.exp(action_lp(p, actions)))

    def enumerated(x, multiplier):
        loss = descendant(x, routed)
        lp = action_lp(x, actions) * multiplier
        from .train.score_bridge import score_corrected_loss

        return jnp.sum(probabilities * score_corrected_loss(loss, lp))

    bridge = jax.grad(enumerated)(p, 1.0)
    duplicate = jax.grad(enumerated)(p, 2.0)
    exact_ok = all(
        bool(jnp.allclose(a, b, atol=2e-6, rtol=1e-6)) for a, b in zip(exact, bridge, strict=True)
    )
    mutation_rejected = any(
        not bool(jnp.allclose(a, b, atol=1e-4, rtol=1e-6))
        for a, b in zip(exact, duplicate, strict=True)
    )
    n = 4096
    previous = CommittedStructure(
        jnp.broadcast_to(jnp.asarray([0, 0, 1]), (n, 3)),
        jnp.ones((n, 3), bool),
        jnp.ones((n, 3), jnp.int32),
    )

    def estimate(x, key):
        predicted = PredictedStructure(
            jnp.broadcast_to(jnp.pad(x.boundary_logits, ((0, 1), (0, 0))), (n, 3, 2)),
            jnp.broadcast_to(x.extent_logits, (n, 3, 3)),
        )
        commit = commit_predicted_structure_core(
            previous,
            predicted,
            full_commit_masks((n, 3)),
            policy="sample",
            max_extent=2,
            key_context=StructureKeyContext(key, 0, jnp.arange(n, dtype=jnp.uint32), 0, 0, 0),
            temperature=1.0,
            estimator_owner=StructureEstimatorOwner.LOCAL_SCORE_BRIDGE,
            allow_reactivation=True,
        )
        route = dynamic_capacity_routing(commit.committed, M_max=3, K=2, max_extent=2)
        assert commit.committed.length is not None
        loss = descendant(x, OracleActions(route.routing.structure.c_b, commit.committed.length))
        return score_corrected_structure_loss(loss, commit).mean()

    fn = jax.jit(jax.grad(estimate))
    estimates = []
    for root in range(16):
        estimates.append(fn(p, named_key("query_oracle:" + name, 0, root)))
    stack = jax.tree.map(lambda *x: jnp.stack(x), *estimates)
    records = []
    ok = exact_ok and mutation_rejected
    for wanted, values in zip(exact, stack, strict=True):
        wanted = jnp.asarray(wanted)
        values = jnp.asarray(values)
        mean = values.mean(0)
        se = values.std(0, ddof=1) / 4
        error = jnp.abs(mean - wanted)
        accept = jnp.where(se == 0, error <= 1e-6, error <= 4 * se)
        ok = ok and bool(jnp.all(accept))
        records.append(
            {"exact": wanted, "mean": mean, "root_se": se, "error": error, "accepted": accept}
        )
    return {
        "fixture": name,
        "action_count": 108,
        "roots": 16,
        "samples_per_root": 4096,
        "exact_ok": exact_ok,
        "duplicate_score_rejected": mutation_rejected,
        "components": records,
        "passed": ok,
    }
