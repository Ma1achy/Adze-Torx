"""Phase-D clean-teacher, optimizer-mask, and trainability regressions."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import pickle

import jax
import jax.numpy as jnp
import pytest

from adze_t.backends.deterministic import DeterministicOps
from adze_t.backends.mapping import deterministic_to_torx
from adze_t.backends.mapping import torx_means_to_deterministic
from adze_t.backends.torx import TorxOperatorConfig, TorxOps
from adze_t.config import REFERENCE_SMALL_V0
from adze_t.model import apply_model, init_model_params
from adze_t.objectives import adamw_init
from adze_t.training import (
    accepted_b3_scratch_initialization,
    codec_update_mask,
    make_fixed_structure_batch,
    stochastic_model_update_mask,
    stochastic_train_step,
)


pytestmark = pytest.mark.integration


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


def _tree_equal(left, right):
    return all(
        bool(jnp.array_equal(a, b))
        for a, b in zip(
            jax.tree_util.tree_leaves(left), jax.tree_util.tree_leaves(right), strict=True
        )
    )


def _noisy(key, lambda_op=1.0):
    return TorxOps.create(
        jax.random.key(key),
        config=TorxOperatorConfig(operator_stochasticity=True, lambda_op=lambda_op),
    )


def test_clean_target_mean_teacher_uses_same_parameter_tree_without_noise_leakage():
    cfg = _tiny_config()
    deterministic = init_model_params(jax.random.key(30), cfg)
    params, _ = deterministic_to_torx(deterministic)
    values = jnp.arange(1, 9, dtype=jnp.int32)[None]
    mask = jnp.ones_like(values, dtype=bool)
    expected = apply_model(
        deterministic, values, mask, values, mask, config=cfg, ops=DeterministicOps()
    )

    def run(root):
        return apply_model(
            params,
            values,
            mask,
            values,
            mask,
            config=cfg,
            ops=_noisy(root),
            target_ops=_noisy(root, 0.0),
        )

    first = run(31)
    second = run(32)
    assert jnp.array_equal(first["target_frontend"], expected["target_frontend"])
    assert jnp.array_equal(first["target"]["h0"], expected["target"]["h0"])
    assert jnp.array_equal(first["target_frontend"], second["target_frontend"])
    assert jnp.array_equal(first["target"]["h0"], second["target"]["h0"])
    assert not jnp.array_equal(first["prompt_frontend"], second["prompt_frontend"])
    assert not jnp.array_equal(first["byte_logits"], second["byte_logits"])
    assert all(
        bool(jnp.all(jnp.isfinite(leaf)))
        for leaf in jax.tree_util.tree_leaves(first)
        if hasattr(leaf, "dtype") and jnp.issubdtype(leaf.dtype, jnp.inexact)
    )


def test_stochastic_mask_retains_phase_b_mask_and_disables_all_rho():
    cfg = _tiny_config()
    params, _ = deterministic_to_torx(init_model_params(jax.random.key(33), cfg))
    mask = stochastic_model_update_mask(params)
    for path, enabled in jax.tree_util.tree_leaves_with_path(mask):
        path_string = jax.tree_util.keystr(path)
        if "['rho']" in path_string:
            assert not bool(enabled)
    assert bool(mask["dit"]["blocks"][0]["q"]["mean"]["weight"])
    assert bool(mask["encoder"]["context"][0]["a_log"]["mean"])
    assert not bool(mask["encoder"]["frontend"][0]["a_log"]["mean"])


def test_finite_stochastic_training_step_freezes_rho_codec_and_moments():
    cfg = _tiny_config()
    params, _ = deterministic_to_torx(init_model_params(jax.random.key(34), cfg))
    zero = adamw_init(params)
    moments = (zero, zero)
    values = jnp.arange(1, 9, dtype=jnp.int32)[None]
    batch = make_fixed_structure_batch(values, values[:, ::-1], config=cfg)
    before = params
    before_moments = moments
    update = jax.jit(stochastic_train_step, static_argnames=("config",))
    params, moments, metrics = update(params, moments, 1, batch, jax.random.key(5100), config=cfg)
    assert all(bool(jnp.all(jnp.isfinite(value))) for value in jax.tree_util.tree_leaves(metrics))
    assert float(metrics["grad_raw_norm"]) > 0
    assert float(metrics["grad_rho_raw_norm"]) > 0
    assert float(metrics["grad_direct_ssm_norm"]) > 0
    assert float(metrics["grad_clipped_applied_norm"]) <= cfg.training.grad_clip_norm
    assert not _tree_equal(
        params["dit"]["blocks"][0]["q"]["mean"],
        before["dit"]["blocks"][0]["q"]["mean"],
    )
    assert _tree_equal(params["encoder"]["target"], before["encoder"]["target"])
    old_by_path = {
        jax.tree_util.keystr(path): value
        for path, value in jax.tree_util.tree_leaves_with_path(before)
    }
    new_by_path = {
        jax.tree_util.keystr(path): value
        for path, value in jax.tree_util.tree_leaves_with_path(params)
    }
    for path, old in old_by_path.items():
        if "['rho']" in path:
            assert jnp.array_equal(new_by_path[path], old)
    update_mask = stochastic_model_update_mask(before)
    for old_moment, new_moment in zip(before_moments, moments, strict=True):
        for old, new, enabled in zip(
            jax.tree_util.tree_leaves(old_moment),
            jax.tree_util.tree_leaves(new_moment),
            jax.tree_util.tree_leaves(update_mask),
            strict=True,
        ):
            if not bool(enabled):
                assert jnp.array_equal(old, new)


def test_d3_initialization_freshens_noncodec_and_restores_exact_codec():
    cfg = _tiny_config()
    accepted_codec = init_model_params(jax.random.key(35), cfg)
    initialized = accepted_b3_scratch_initialization(accepted_codec, cfg)
    fresh = init_model_params(jax.random.PRNGKey(700), cfg)
    mask = codec_update_mask(fresh)
    for actual, codec, generated, use_codec in zip(
        jax.tree_util.tree_leaves(initialized),
        jax.tree_util.tree_leaves(accepted_codec),
        jax.tree_util.tree_leaves(fresh),
        jax.tree_util.tree_leaves(mask),
        strict=True,
    ):
        assert jnp.array_equal(actual, codec if bool(use_codec) else generated)


def test_accepted_phase_b_checkpoints_map_to_torx_means_exactly():
    checkpoint_root = Path(__file__).resolve().parents[1] / "results/phase_b/checkpoints"
    names = ("target_codec_b1", "copy", "reverse")
    if not all((checkpoint_root / f"{name}.pkl").is_file() for name in names):
        pytest.skip("accepted Phase-B checkpoint fixtures are not present in this checkout")
    for name in names:
        with (checkpoint_root / f"{name}.pkl").open("rb") as stream:
            deterministic = jax.tree.map(
                jnp.asarray,
                pickle.load(stream),  # noqa: S301 - trusted committed test fixture
            )
        mapped, _ = deterministic_to_torx(deterministic)
        recovered = torx_means_to_deterministic(mapped)
        assert _tree_equal(deterministic, recovered)
