"""Phase F.2 same-checkpoint trajectory contracts."""

from __future__ import annotations

from dataclasses import replace
import inspect

import jax
import jax.numpy as jnp

from adze_t.backends.mapping import deterministic_to_torx
from adze_t.backends.torx import TorxOperatorConfig, TorxOps
from adze_t.config import REFERENCE_SMALL_V0
from adze_t.corruption import DiffusionStage, diffusion_key, phase_f_schedule, recorrupt_h
from adze_t.denoise import (
    F2_NATIVE_S_CONDITIONING_UNTRAINED,
    F2_STEP0_CONDITIONING,
    apply_denoising_trajectory,
    make_sanitized_target_analysis,
)
from adze_t.model import apply_clean_target_teacher, apply_model, init_model_params


def _tiny_config():
    cfg = REFERENCE_SMALL_V0
    return replace(
        cfg,
        carrier=replace(cfg.carrier, C=8, h_dim=16, L_max=2),
        packing=replace(cfg.packing, M_max=4, K=2),
        model=replace(
            cfg.model,
            d_front=8,
            d_ctx=16,
            frontend_layers=1,
            context_layers=1,
            target_layers=1,
            proposal_layers=1,
            proposal_hidden_dim=8,
            d_model=16,
            heads=2,
            head_dim=8,
            ffn_hidden=32,
            physical_blocks_L=2,
            cycles_Q=2,
            d_dec=16,
            decoder_layers=1,
            mamba_expand=1,
            mamba_state_dim=4,
        ),
    )


def _fixture():
    cfg = _tiny_config()
    params = init_model_params(jax.random.key(1200), cfg)
    prompt = jnp.full((2, 1), 0xF1, dtype=jnp.int32)
    prompt_mask = jnp.ones_like(prompt, dtype=bool)
    target = jnp.arange(16, dtype=jnp.int32).reshape(2, 8)
    target_mask = jnp.ones_like(target, dtype=bool)
    clean = apply_clean_target_teacher(params, target, target_mask, config=cfg)
    teacher = clean["target"]["teacher"]
    sanitized = make_sanitized_target_analysis(
        teacher.boundaries,
        teacher.length,
        config=cfg,
        target_width=target.shape[1],
    )
    initial = jax.random.normal(jax.random.key(1201), clean["target"]["h0"].shape)
    nu = jnp.asarray([0.4, 0.8], dtype=jnp.float32)
    ids = jnp.asarray([17, 29], dtype=jnp.uint32)
    return cfg, params, prompt, prompt_mask, target, target_mask, clean, sanitized, initial, nu, ids


def _run(s_exec=4, eta=0, conditioning=F2_STEP0_CONDITIONING, sanitized=None):
    fixture = _fixture()
    cfg, params, prompt, prompt_mask, _, _, _, default_sanitized, initial, nu, ids = fixture
    trajectory = apply_denoising_trajectory(
        params,
        prompt,
        prompt_mask,
        initial,
        default_sanitized if sanitized is None else sanitized,
        nu,
        ids,
        s_exec=s_exec,
        eta_diff=eta,
        diffusion_root=jax.random.key(1202),
        conditioning_mode=conditioning,
        config=cfg,
    )
    return fixture, trajectory


def _tree_equal(left, right):
    return all(
        bool(jnp.array_equal(a, b))
        for a, b in zip(
            jax.tree.leaves(left),
            jax.tree.leaves(right),
            strict=True,
        )
    )


def test_trajectory_does_not_accept_clean_h0_and_s1_exactly_matches_f1():
    fixture, trajectory = _run(s_exec=1)
    cfg, params, prompt, prompt_mask, target, target_mask, clean, _, initial, nu, _ = fixture
    reference = apply_model(
        params,
        prompt,
        prompt_mask,
        target,
        target_mask,
        config=cfg,
        target_analysis=clean,
        carrier_h_input=initial,
        noise_level=nu,
        denoise_step=0,
    )
    assert "h0" not in inspect.signature(apply_denoising_trajectory).parameters
    assert jnp.array_equal(trajectory.h_hat[0], reference["prediction"][0])
    assert jnp.array_equal(trajectory.byte_logits[0], reference["byte_logits"])
    assert jnp.array_equal(trajectory.input_states[0], initial)


def test_shapes_strict_prefix_schedule_and_recorruption_keys_use_actual_next_s():
    fixture, full = _run(s_exec=4, eta=1)
    _, _, _, _, _, _, _, _, _, nu, ids = fixture
    for steps in (1, 2, 3):
        _, prefix = _run(s_exec=steps, eta=1)
        assert jnp.array_equal(prefix.h_hat, full.h_hat[:steps])
        assert jnp.array_equal(prefix.byte_logits, full.byte_logits[:steps])
        assert jnp.array_equal(prefix.input_states, full.input_states[:steps])
        assert jnp.array_equal(prefix.schedule, full.schedule[:steps])
        assert jnp.array_equal(prefix.recorruption_epsilon, full.recorruption_epsilon[: steps - 1])
    assert full.h_hat.shape[:2] == (4, 2)
    assert jnp.allclose(full.schedule, phase_f_schedule(nu, 4).T)
    for step in range(1, 4):
        keys = jax.vmap(
            lambda example_id: diffusion_key(
                jax.random.key(1202),
                global_example_id=example_id,
                stage=DiffusionStage.RECORRUPTION,
                denoise_step=step,
            )
        )(ids)
        expected = jax.vmap(
            lambda key: jax.random.normal(key, full.h_hat.shape[2:], dtype=full.h_hat.dtype)
        )(keys)
        assert jnp.array_equal(full.recorruption_epsilon[step - 1], expected)


def test_eta_formulas_have_no_hidden_s_or_noise_normalization():
    _, deterministic = _run(s_exec=4, eta=0)
    _, stochastic = _run(s_exec=4, eta=1)
    for step in range(1, 4):
        assert jnp.array_equal(
            deterministic.input_states[step],
            recorrupt_h(
                deterministic.h_hat[step - 1],
                deterministic.schedule[step],
                deterministic.recorruption_epsilon[step - 1],
                0,
            ),
        )
        assert jnp.array_equal(
            stochastic.input_states[step],
            recorrupt_h(
                stochastic.h_hat[step - 1],
                stochastic.schedule[step],
                stochastic.recorruption_epsilon[step - 1],
                1,
            ),
        )


def test_primary_freezes_conditioning_but_actual_s_and_native_diagnostic_do_not():
    _, primary = _run(s_exec=4, conditioning=F2_STEP0_CONDITIONING)
    _, native = _run(s_exec=4, conditioning=F2_NATIVE_S_CONDITIONING_UNTRAINED)
    expected_actual = jnp.arange(4, dtype=jnp.int32)[:, None]
    assert jnp.array_equal(primary.actual_s_indices, jnp.broadcast_to(expected_actual, (4, 2)))
    assert jnp.all(primary.denoise_condition_indices == 0)
    assert jnp.array_equal(native.actual_s_indices, primary.actual_s_indices)
    assert jnp.array_equal(native.denoise_condition_indices, primary.actual_s_indices)
    assert jnp.array_equal(native.h_hat[0], primary.h_hat[0])
    assert not jnp.array_equal(native.h_hat[1:], primary.h_hat[1:])


def test_actual_s_changes_torx_occurrence_keys_while_conditioning_stays_zero(monkeypatch):
    cfg, deterministic, prompt, prompt_mask, target, target_mask, clean, _, initial, nu, _ = (
        _fixture()
    )
    params, _ = deterministic_to_torx(deterministic)
    import adze_t.dit as dit_module

    conditioning_values = []
    original = dit_module.build_conditioning

    def observe_conditioning(prompt_global, noise, mode, denoise_step, *args):
        conditioning_values.append(int(denoise_step))
        return original(prompt_global, noise, mode, denoise_step, *args)

    monkeypatch.setattr(dit_module, "build_conditioning", observe_conditioning)
    key_sets = []
    for actual_s in (0, 1):
        records = []

        def observe(_kind, _name, path, key):
            records.append((path, key))

        ops = TorxOps.create(
            jax.random.key(1203),
            config=TorxOperatorConfig(operator_stochasticity=True, lambda_op=1.0),
            global_example_id=17,
            occurrence_observer=observe,
        )
        apply_model(
            params,
            prompt[:1],
            prompt_mask[:1],
            target[:1],
            target_mask[:1],
            config=cfg,
            ops=ops,
            target_analysis=jax.tree.map(lambda value: value[:1], clean),
            carrier_h_input=initial[:1],
            noise_level=nu[:1],
            actual_s_index=actual_s,
            denoise_condition_index=0,
        )
        key_sets.append({path: key for path, key in records})
        grouped = {}
        for path, key in records:
            grouped.setdefault(path, []).append(key)
        recurrent_paths = [keys for keys in grouped.values() if len(keys) > 1]
        assert recurrent_paths
        assert any(
            any(not jnp.array_equal(keys[0], key) for key in keys[1:]) for keys in recurrent_paths
        )
    shared_paths = set(key_sets[0]) & set(key_sets[1])
    assert shared_paths
    assert all(not jnp.array_equal(key_sets[0][path], key_sets[1][path]) for path in shared_paths)
    assert conditioning_values and set(conditioning_values) == {0}


def test_sanitized_placeholder_content_is_irrelevant_and_structure_never_recommits():
    fixture = _fixture()
    cfg, params, prompt, prompt_mask, _, _, _, sanitized, initial, nu, ids = fixture
    changed = make_sanitized_target_analysis(
        sanitized.teacher.boundaries,
        sanitized.teacher.length,
        config=cfg,
        target_width=sanitized.target_bytes_placeholder.shape[1],
        placeholder_value=173,
    )
    before = jax.tree.map(lambda value: value.copy(), params)

    def run(value):
        return apply_denoising_trajectory(
            params,
            prompt,
            prompt_mask,
            initial,
            value,
            nu,
            ids,
            s_exec=4,
            eta_diff=0,
            diffusion_root=jax.random.key(1204),
            conditioning_mode=F2_STEP0_CONDITIONING,
            config=cfg,
        )

    baseline, altered = run(sanitized), run(changed)
    assert jnp.array_equal(baseline.h_hat, altered.h_hat)
    assert jnp.array_equal(baseline.byte_logits, altered.byte_logits)
    assert jnp.array_equal(baseline.input_states, altered.input_states)
    for field in baseline.metadata.__dict__:
        values = getattr(baseline.metadata, field)
        assert jnp.all(values == values[0])
    assert _tree_equal(params, before)
    assert not bool(jnp.any(baseline.diagnostics["nonfinite"]))
