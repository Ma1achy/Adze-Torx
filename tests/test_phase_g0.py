"""Behavioral oracles for the separately versioned G.0 candidate."""

from itertools import product

import jax
import jax.numpy as jnp
import pytest

from adze_t.generated_structure import (
    StructureEstimatorOwner,
    StructureKeyContext,
    commit_predicted_structure_core,
    full_commit_masks,
    score_corrected_structure_loss,
)
from adze_t.phase_g0 import dynamic_capacity_routing
from adze_t.phase_g_oracle import (
    OracleActions,
    descendant_fixture,
    enumerate_actions,
    fixture_parameters,
)
from adze_t.state import CommittedStructure, PredictedStructure


def python_capacity(edges, k):
    run, result = 0, []
    for edge in edges:
        run += 1
        routed = bool(edge or run == k)
        result.append(int(routed))
        if routed:
            run = 0
    return result


@pytest.mark.parametrize("k", [1, 2, 3, 8])
def test_dynamic_capacity_matches_independent_partition_oracle_eager_jit(k):
    edges = jnp.asarray([list(x) + [1] for x in product((0, 1), repeat=7)], jnp.int32)
    semantic = CommittedStructure(edges, jnp.ones_like(edges, bool), jnp.ones_like(edges))

    def fn(s):
        return dynamic_capacity_routing(s, M_max=8, K=k, max_extent=4)

    eager, compiled = fn(semantic), jax.jit(fn)(semantic)
    expected = jnp.asarray([python_capacity(x, k) for x in edges.tolist()])
    assert jnp.array_equal(eager.routing.structure.c_b, expected)
    assert jax.tree_util.tree_all(jax.tree.map(jnp.array_equal, eager, compiled))
    assert jnp.array_equal(eager.semantic.c_b, edges)
    assert jnp.all(eager.routing.executed)
    assert jnp.array_equal(eager.capacity_cut, (expected == 1) & (edges == 0))


def test_capacity_does_not_repair_bad_semantics_and_count_overflow_fails_closed():
    s = CommittedStructure(
        jnp.zeros((1, 4), jnp.int32), jnp.ones((1, 4), bool), jnp.ones((1, 4), jnp.int32)
    )
    invalid = dynamic_capacity_routing(s, M_max=4, K=2, max_extent=4)
    assert not invalid.routing.executed
    valid = CommittedStructure(jnp.ones((1, 4), jnp.int32), s.activity, s.length)
    overflow = dynamic_capacity_routing(valid, M_max=1, K=2, max_extent=4)
    assert not overflow.routing.executed
    assert jnp.all(overflow.routing.metadata.packed_to_carrier == -1)


def test_dynamic_capacity_scan_vmap_and_inactive_identity():
    from adze_t.packing import pack_values
    from adze_t.unpool import unpool_values

    s = CommittedStructure(
        jnp.asarray([[0, 0, 1]], jnp.int32), jnp.zeros((1, 3), bool), jnp.zeros((1, 3), jnp.int32)
    )

    def fn(value):
        return dynamic_capacity_routing(value, M_max=3, K=2, max_extent=2)

    repeated = jax.tree.map(lambda x: jnp.stack((x, x)), s)
    mapped = jax.vmap(fn)(repeated)
    assert jnp.all(mapped.routing.executed)

    def body(state, _):
        route = fn(state)
        return route.semantic, route.capacity_cut

    final, cuts = jax.lax.scan(body, s, None, length=2)
    assert jnp.array_equal(final.c_b, s.c_b)
    assert cuts.shape == (2, 1, 3)
    route = fn(s).routing
    values = jnp.arange(6, dtype=jnp.float32).reshape(1, 3, 2)
    assert jnp.array_equal(
        unpool_values(pack_values(values, route.metadata), route.metadata, C=3), values
    )
    assert not jnp.any(route.metadata.kv_mask)
    assert jnp.sum(route.metadata.query_mask) == 3


@pytest.mark.parametrize("fixture", ["uniform", "asymmetric", "peaked"])
def test_production_capacity_pushforward_exact_gradient_and_logp_ownership(fixture):
    parameters = fixture_parameters(fixture)
    actions = enumerate_actions()
    # Exact objective uses an independent Python mapping, not production scan.
    independent = OracleActions(
        jnp.asarray([python_capacity(x, 2) for x in actions.boundary.tolist()]), actions.extent
    )
    descendant = descendant_fixture("direct_dependencies")

    def logp(p):
        b = jax.nn.log_softmax(p.boundary_logits)
        ell = jax.nn.log_softmax(p.extent_logits)
        return b[jnp.arange(2)[None], actions.boundary[:, :2]].sum(1) + ell[
            jnp.arange(3)[None], actions.extent
        ].sum(1)

    def expected(p):
        return jnp.sum(jnp.exp(logp(p)) * descendant(p, independent))

    exact = jax.grad(expected)(parameters)
    semantic = CommittedStructure(actions.boundary, actions.extent > 0, actions.extent)
    routed = dynamic_capacity_routing(semantic, M_max=3, K=2, max_extent=2)
    assert routed.routing.executed
    production = OracleActions(routed.routing.structure.c_b, actions.extent)
    from adze_t.train.score_bridge import score_corrected_loss

    probabilities = jax.lax.stop_gradient(jnp.exp(logp(parameters)))
    estimate = jax.grad(
        lambda p: jnp.sum(probabilities * score_corrected_loss(descendant(p, production), logp(p)))
    )(parameters)
    assert jax.tree_util.tree_all(
        jax.tree.map(lambda a, b: jnp.allclose(a, b, atol=2e-6), exact, estimate)
    )
    assert jnp.allclose(jnp.exp(logp(parameters)).sum(), 1.0)
    # Distinct semantic actions may have identical routed effects; their masses
    # are retained, never conditioned or renormalized to "valid" draws.
    assert len(set(map(tuple, production.boundary.tolist()))) < len(
        set(map(tuple, actions.boundary.tolist()))
    )


def test_full_site_sample_keeps_logp_for_capacity_hidden_semantic_actions():
    p = fixture_parameters("uniform")
    previous = CommittedStructure(
        jnp.asarray([[0, 0, 1]]), jnp.ones((1, 3), bool), jnp.ones((1, 3), jnp.int32)
    )
    predicted = PredictedStructure(
        jnp.pad(p.boundary_logits[None], ((0, 0), (0, 1), (0, 0))), p.extent_logits[None]
    )
    commit = commit_predicted_structure_core(
        previous,
        predicted,
        full_commit_masks((1, 3)),
        policy="sample",
        max_extent=2,
        key_context=StructureKeyContext(jax.random.key(970), 0, jnp.asarray([0]), 0, 0, 0),
        temperature=1.0,
        estimator_owner=StructureEstimatorOwner.LOCAL_SCORE_BRIDGE,
        allow_reactivation=True,
    )
    expected = -2 * jnp.log(2.0) - 3 * jnp.log(3.0)
    assert jnp.allclose(commit.sampled.descendant_log_prob, expected)
    assert commit.sampled.boundary_log_prob[0, -1] == 0
    assert jnp.isfinite(score_corrected_structure_loss(jnp.ones((1,)), commit)).all()


def test_candidate_update_and_overlay_preserve_noncodec_f1_state():
    from dataclasses import replace
    from adze_t.backends.mapping import deterministic_to_torx
    from adze_t.config import REFERENCE_SMALL_V0
    from adze_t.model import init_model_params
    from adze_t.objectives import adamw_init
    from adze_t.phase_g0 import codec_subtree, codec_update, with_frozen_g0_codec

    cfg = REFERENCE_SMALL_V0
    cfg = replace(
        cfg,
        carrier=replace(cfg.carrier, C=3, h_dim=8, L_max=2),
        model=replace(
            cfg.model,
            d_front=8,
            d_ctx=8,
            d_model=8,
            heads=1,
            head_dim=8,
            ffn_hidden=16,
            d_dec=8,
            frontend_layers=1,
            context_layers=1,
            target_layers=1,
            decoder_layers=1,
            proposal_layers=1,
            physical_blocks_L=1,
            cycles_Q=1,
            mamba_state_dim=2,
            mamba_expand=1,
        ),
    )
    full = init_model_params(jax.random.key(976), cfg)
    original, _ = deterministic_to_torx(full)
    candidate = codec_subtree(full)
    moments = (adamw_init(candidate), adamw_init(candidate))
    target = jnp.asarray([[1, 2, 3, 0, 0, 0]], jnp.int32)
    updated, updated_moments, result = jax.jit(
        codec_update, static_argnames=("decoder_only", "config")
    )(candidate, moments, jnp.asarray(1), target, target > 0, decoder_only=False, config=cfg)
    assert result["finite"]
    assert all(jnp.all(jnp.isfinite(x)) for x in jax.tree.leaves((updated, updated_moments)))
    installed = with_frozen_g0_codec(original, updated)
    for key in (
        "dit",
        "proposal",
        "h_head",
        "b_head",
        "l_head",
        "carrier_prior",
        "carrier_in",
        "carrier_out",
    ):
        assert jax.tree_util.tree_all(jax.tree.map(jnp.array_equal, original[key], installed[key]))
    for (path, before), (_, after) in zip(
        jax.tree_util.tree_leaves_with_path(original),
        jax.tree_util.tree_leaves_with_path(installed),
        strict=True,
    ):
        if "rho" in jax.tree_util.keystr(path):
            assert jnp.array_equal(before, after)
    assert any(
        not jnp.array_equal(a, b)
        for a, b in zip(jax.tree.leaves(candidate), jax.tree.leaves(updated), strict=True)
    )


@pytest.mark.integration
def test_real_model_two_pass_accepts_full_site_dynamic_capacity_policy():
    from dataclasses import replace
    from test_phase_g_integration import _fixture
    from adze_t.phase_g import generated_two_pass_objective

    cfg, params, batch = _fixture()
    cfg = replace(cfg, packing=replace(cfg.packing, K=2))
    loss, metrics = jax.jit(
        lambda p, b: generated_two_pass_objective(
            p,
            b,
            jax.random.key(978),
            jax.random.key(979),
            jax.random.key(980),
            0,
            arm="sample",
            temperature=1.0,
            dynamic_capacity=True,
            config=cfg,
        )
    )(params, batch)
    assert metrics["executed"]
    assert jnp.isfinite(loss)
    assert jnp.isfinite(metrics["byte_loss"])
