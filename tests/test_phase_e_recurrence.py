from dataclasses import replace

import jax
import jax.numpy as jnp

from adze_t.config import REFERENCE_SMALL_V0
from adze_t.dit import DiTConfig, apply_dit, init_dit_params
from adze_t.packing import build_pack_metadata_core


def _inputs(cycles=3):
    cfg = DiTConfig(
        d_model=8,
        heads=2,
        head_dim=4,
        ffn_hidden=16,
        physical_blocks=2,
        cycles=cycles,
        carrier_capacity=4,
        d_context=128,
        max_blocks=2,
        max_slots=2,
        max_extent=2,
    )
    metadata = build_pack_metadata_core(
        jnp.zeros((1, 4), dtype=jnp.int32), jnp.ones((1, 4), dtype=bool), M_max=2, K=2
    )
    return (
        cfg,
        init_dit_params(jax.random.PRNGKey(0), cfg),
        metadata,
        jnp.ones((1, 2, 2, 8)),
        jnp.ones((1, 128)),
    )


def test_phase_e_q_execution_and_cycle_capture_shapes():
    cfg, params, metadata, packed, context = _inputs()
    for q in (1, 2, 3):
        _, aux = apply_dit(packed, metadata, context, params, cfg, cycles=q)
        assert aux["trajectory"].shape == (q, 1, 4, 8)
        assert aux["effective_depths"].tolist() == list(range(q * 2))


def test_phase_e_q_zero_is_a_finite_diagnostic_bypass():
    cfg, params, metadata, packed, context = _inputs()
    output, aux = apply_dit(
        packed, metadata, context, params, cfg, cycles=0, capture_diagnostics=True
    )
    assert output.shape == packed.shape
    assert aux["trajectory"].shape == (0, 1, 4, 8)
    assert aux["block_trajectory"].shape == (0, 1, 4, 8)
    assert aux["effective_depths"].shape == (0,)
    assert jnp.all(jnp.isfinite(output))


def test_phase_e_depth_toggle_and_intervention_paths_are_finite():
    cfg, params, metadata, packed, context = _inputs()
    no_depth = replace(cfg, effective_depth_conditioning=False)
    baseline, _ = apply_dit(packed, metadata, context, params, cfg)
    suppressed, _ = apply_dit(packed, metadata, context, params, cfg, suppress_cycle=1)
    all_q0, _ = apply_dit(packed, metadata, context, params, no_depth)
    assert jnp.all(jnp.isfinite(baseline))
    assert jnp.all(jnp.isfinite(suppressed))
    assert jnp.all(jnp.isfinite(all_q0))


def test_phase_e_reference_and_q1_parameter_counts_match():
    from adze_t.model import init_model_params

    reference = init_model_params(jax.random.PRNGKey(2), REFERENCE_SMALL_V0)
    q1_config = replace(REFERENCE_SMALL_V0, model=replace(REFERENCE_SMALL_V0.model, cycles_Q=1))
    q1 = init_model_params(jax.random.PRNGKey(2), q1_config)
    assert sum(x.size for x in jax.tree.leaves(reference)) == sum(
        x.size for x in jax.tree.leaves(q1)
    )
