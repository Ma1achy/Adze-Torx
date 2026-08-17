from dataclasses import replace

import jax
import jax.numpy as jnp

from adze_t.backends.mapping import deterministic_to_torx
from adze_t.backends.torx import TorxOperatorConfig, TorxOps
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


def test_phase_e_q_zero_is_a_finite_zero_physical_block_baseline():
    cfg, params, metadata, packed, context = _inputs()
    output, aux = apply_dit(
        packed, metadata, context, params, cfg, cycles=0, capture_diagnostics=True
    )
    assert output.shape == packed.shape
    assert aux["trajectory"].shape == (0, 1, 4, 8)
    assert aux["block_trajectory"].shape == (0, 1, 4, 8)
    assert aux["effective_depths"].shape == (0,)
    assert jnp.all(jnp.isfinite(output))


def test_q_zero_ignores_physical_blocks_but_uses_dit_shell():
    cfg, params, metadata, packed, context = _inputs()
    baseline, _ = apply_dit(packed, metadata, context, params, cfg, cycles=0)
    changed_blocks = dict(params)
    changed_blocks["blocks"] = jax.tree.map(lambda value: value + 1.0, params["blocks"])
    block_output, _ = apply_dit(packed, metadata, context, changed_blocks, cfg, cycles=0)
    assert jnp.array_equal(baseline, block_output)

    changed_shell = dict(params)
    changed_shell["input_proj"] = jax.tree.map(lambda value: value + 1.0, params["input_proj"])
    shell_output, _ = apply_dit(packed, metadata, context, changed_shell, cfg, cycles=0)
    assert not jnp.allclose(baseline, shell_output)


def test_phase_e_depth_toggle_and_intervention_paths_are_finite():
    cfg, params, metadata, packed, context = _inputs()
    no_depth = replace(cfg, effective_depth_conditioning=False)
    baseline, _ = apply_dit(packed, metadata, context, params, cfg)
    suppressed, _ = apply_dit(packed, metadata, context, params, cfg, suppress_cycle=1)
    all_q0, _ = apply_dit(packed, metadata, context, params, no_depth)
    assert jnp.all(jnp.isfinite(baseline))
    assert jnp.all(jnp.isfinite(suppressed))
    assert jnp.all(jnp.isfinite(all_q0))


def test_depth_conditioning_overrides_preserve_actual_cycle_occurrence_keys():
    cfg, deterministic, metadata, packed, context = _inputs()
    params, _ = deterministic_to_torx(deterministic)

    def execute(override):
        records = []

        def observe(kind, name, path, key):
            del kind, path
            records.append((name, tuple(map(int, jax.random.key_data(key)))))

        ops = TorxOps.create(
            jax.random.key(71),
            config=TorxOperatorConfig(operator_stochasticity=True, lambda_op=1.0),
            occurrence_observer=observe,
        )
        output, auxiliary = apply_dit(
            packed,
            metadata,
            context,
            params,
            cfg,
            ops=ops,
            depth_code_override=override,
        )
        return output, auxiliary["effective_depths"], records

    normal, normal_depths, normal_keys = execute("correct")
    all_q0, all_q0_depths, all_q0_keys = execute("all_q0")
    reversed_output, reversed_depths, reversed_keys = execute("reversed")
    assert normal_keys == all_q0_keys == reversed_keys
    assert normal_depths.tolist() == [0, 1, 2, 3, 4, 5]
    assert all_q0_depths.tolist() == [0, 1, 0, 1, 0, 1]
    assert reversed_depths.tolist() == [4, 5, 2, 3, 0, 1]
    assert not jnp.allclose(normal, all_q0)
    assert not jnp.allclose(normal, reversed_output)

    q_keys = [key for name, key in normal_keys if name == "dit.block_0.q"]
    assert len(q_keys) == 3
    assert len(set(q_keys)) == 3


def test_stop_gradient_control_is_forward_identical_but_changes_gradient_path():
    cfg, params, metadata, packed, context = _inputs()

    def objective(values, stop_gradient_after_cycle=None):
        output, _ = apply_dit(
            values,
            metadata,
            context,
            params,
            cfg,
            stop_gradient_after_cycle=stop_gradient_after_cycle,
        )
        return jnp.sum(output)

    baseline = objective(packed)
    controlled = objective(packed, 0)
    assert jnp.array_equal(baseline, controlled)
    baseline_grad = jax.grad(objective)(packed)
    controlled_grad = jax.grad(lambda values: objective(values, 0))(packed)
    assert not jnp.allclose(baseline_grad, controlled_grad)

    suppressed, _ = apply_dit(packed, metadata, context, params, cfg, suppress_cycle=1)
    ordinary, _ = apply_dit(packed, metadata, context, params, cfg)
    assert not jnp.allclose(ordinary, suppressed)


def test_phase_e_reference_and_q1_parameter_counts_match():
    from adze_t.model import init_model_params

    reference = init_model_params(jax.random.PRNGKey(2), REFERENCE_SMALL_V0)
    q1_config = replace(REFERENCE_SMALL_V0, model=replace(REFERENCE_SMALL_V0.model, cycles_Q=1))
    q1 = init_model_params(jax.random.PRNGKey(2), q1_config)
    assert sum(x.size for x in jax.tree.leaves(reference)) == sum(
        x.size for x in jax.tree.leaves(q1)
    )
