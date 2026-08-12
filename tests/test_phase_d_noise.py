"""Phase-D finite-noise primitive and occurrence contracts."""

from __future__ import annotations

from dataclasses import replace

import jax
import jax.numpy as jnp
import pytest

from adze_t.backends.mapping import deterministic_to_torx
from adze_t.backends.torx import (
    PHASE_D_INITIAL_SIGMA,
    PHASE_D_SIGMA_MAX,
    PHASE_D_SIGMA_MIN,
    TorxOperatorConfig,
    TorxOps,
    rho_from_sigma,
    sigma_from_rho,
)
from adze_t.config import REFERENCE_SMALL_V0
from adze_t.model import apply_model, init_model_params


def _config(lambda_op=1.0):
    return TorxOperatorConfig(operator_stochasticity=True, lambda_op=lambda_op)


def _mapped_primitives():
    deterministic = {
        "affine": {
            "weight": jnp.array([[0.25, -0.5], [0.75, 0.125]], dtype=jnp.float32),
            "bias": jnp.array([0.2, -0.1], dtype=jnp.float32),
        },
        "embedding": jnp.arange(12, dtype=jnp.float32).reshape(6, 2) / 9,
        "conv": {
            "kernel": jnp.arange(6, dtype=jnp.float32).reshape(3, 2) / 11,
            "bias": jnp.array([0.15, -0.2], dtype=jnp.float32),
        },
    }
    return deterministic, deterministic_to_torx(deterministic)[0]


def _factor_output(kind, params, key, lambda_op):
    ops = TorxOps.create(key, config=_config(lambda_op))
    x = jnp.array([[[0.3, -0.7], [0.5, 0.2]]], dtype=jnp.float32)
    indices = jnp.array([[1, 4]], dtype=jnp.int32)
    if kind == "affine":
        return ops.linear(x, params["affine"], name="oracle.affine")
    if kind == "logits":
        return ops.categorical_logits(x, params["affine"], name="oracle.logits")
    if kind == "embedding":
        return ops.embedding(indices, params["embedding"], name="oracle.embedding")
    return ops.depthwise_conv1d(x, params["conv"], name="oracle.conv")


def test_sigma_policy_initial_value_and_clamp_limits():
    rho = rho_from_sigma(PHASE_D_INITIAL_SIGMA)
    assert jnp.allclose(sigma_from_rho(rho), PHASE_D_INITIAL_SIGMA, rtol=1.0e-6)
    assert sigma_from_rho(jnp.asarray(-100.0)) == PHASE_D_SIGMA_MIN
    assert sigma_from_rho(jnp.asarray(100.0)) == PHASE_D_SIGMA_MAX


@pytest.mark.parametrize("kind", ["affine", "logits", "embedding", "conv"])
@pytest.mark.parametrize("lambda_op", [0.1, 0.25, 0.5, 1.0])
def test_fixed_epsilon_residual_scales_exactly_with_lambda(kind, lambda_op):
    _, params = _mapped_primitives()
    key = jax.random.key(11)
    clean = _factor_output(kind, params, key, 0.0)
    unit = _factor_output(kind, params, key, 1.0)
    scaled = _factor_output(kind, params, key, lambda_op)
    assert jnp.allclose(scaled - clean, lambda_op * (unit - clean), atol=1.0e-7)


def test_finite_factors_reproduce_same_key_and_vary_across_key_and_scope():
    _, params = _mapped_primitives()
    x = jnp.array([[[0.3, -0.7], [0.5, 0.2]]], dtype=jnp.float32)
    indices = jnp.array([[1, 4]], dtype=jnp.int32)

    def outputs(root, scope):
        ops = TorxOps.create(jax.random.key(root), config=_config()).with_scope(scope)
        return (
            ops.linear(x, params["affine"], name="oracle.affine"),
            ops.categorical_logits(x, params["affine"], name="oracle.logits"),
            ops.embedding(indices, params["embedding"], name="oracle.embedding"),
            ops.depthwise_conv1d(x, params["conv"], name="oracle.conv"),
        )

    first = outputs(12, "first")
    assert all(jnp.array_equal(a, b) for a, b in zip(first, outputs(12, "first"), strict=True))
    assert all(not jnp.array_equal(a, b) for a, b in zip(first, outputs(13, "first"), strict=True))
    assert all(not jnp.array_equal(a, b) for a, b in zip(first, outputs(12, "second"), strict=True))


def test_lambda_zero_is_exact_and_has_zero_rho_gradient():
    deterministic, params = _mapped_primitives()
    x = jnp.array([[0.3, -0.7]], dtype=jnp.float32)
    ops = TorxOps.create(jax.random.key(14), config=_config(0.0))
    expected = x @ deterministic["affine"]["weight"] + deterministic["affine"]["bias"]
    assert jnp.array_equal(ops.linear(x, params["affine"], name="oracle.affine"), expected)

    def objective(rho):
        changed = {**params["affine"], "rho": rho}
        return jnp.sum(ops.linear(x, changed, name="oracle.affine") ** 2)

    rho_grad = jax.grad(objective)(params["affine"]["rho"])
    assert jnp.array_equal(rho_grad, jnp.zeros_like(rho_grad))


def test_fixed_key_affine_gradients_match_explicit_reparameterization():
    _, params = _mapped_primitives()
    affine = params["affine"]
    x = jnp.array([[0.3, -0.7], [0.5, 0.2]], dtype=jnp.float32)
    key = jax.random.key(15)
    occurrence_key = TorxOps.create(key, config=_config()).context.key_for("oracle.affine")

    def factor_objective(input_x, parameter):
        output = TorxOps.create(key, config=_config()).linear(
            input_x, parameter, name="oracle.affine"
        )
        return 0.5 * jnp.sum(output**2)

    def reference_objective(input_x, parameter):
        mean = input_x @ parameter["mean"]["weight"] + parameter["mean"]["bias"]
        epsilon = jax.random.normal(occurrence_key, mean.shape, dtype=mean.dtype)
        output = mean + sigma_from_rho(parameter["rho"]) * epsilon
        return 0.5 * jnp.sum(output**2)

    actual = jax.grad(factor_objective, argnums=(0, 1))(x, affine)
    expected = jax.grad(reference_objective, argnums=(0, 1))(x, affine)
    assert all(
        jnp.array_equal(left, right)
        for left, right in zip(
            jax.tree_util.tree_leaves(actual), jax.tree_util.tree_leaves(expected), strict=True
        )
    )


@pytest.mark.slow
@pytest.mark.parametrize("kind", ["affine", "logits", "embedding", "conv"])
def test_4096_sample_factor_moments(kind):
    _, params = _mapped_primitives()
    x = jnp.array([[[0.3, -0.7], [0.5, 0.2]]], dtype=jnp.float32)
    indices = jnp.array([[1, 4]], dtype=jnp.int32)

    def sample(key):
        ops = TorxOps.create(key, config=_config())
        if kind == "affine":
            return ops.linear(x, params["affine"], name="oracle.affine")[0, 0, 0]
        if kind == "logits":
            return ops.categorical_logits(x, params["affine"], name="oracle.logits")[0, 0, 0]
        if kind == "embedding":
            return ops.embedding(indices, params["embedding"], name="oracle.embedding")[0, 0, 0]
        return ops.depthwise_conv1d(x, params["conv"], name="oracle.conv")[0, 0, 0]

    keys = jax.random.split(jax.random.key(16), 4096)
    values = jax.jit(jax.vmap(sample))(keys)
    clean = sample(jax.random.key(17))
    # Recover the analytic mean through the exact zero-noise branch.
    zero_ops = TorxOps.create(jax.random.key(17), config=_config(0.0))
    if kind == "affine":
        mean = zero_ops.linear(x, params["affine"], name="oracle.affine")[0, 0, 0]
    elif kind == "logits":
        mean = zero_ops.categorical_logits(x, params["affine"], name="oracle.logits")[0, 0, 0]
    elif kind == "embedding":
        mean = zero_ops.embedding(indices, params["embedding"], name="oracle.embedding")[0, 0, 0]
    else:
        mean = zero_ops.depthwise_conv1d(x, params["conv"], name="oracle.conv")[0, 0, 0]
    del clean
    variance = PHASE_D_INITIAL_SIGMA**2
    empirical_mean = jnp.mean(values)
    empirical_variance = jnp.var(values, ddof=1)
    mean_limit = 5 * PHASE_D_INITIAL_SIGMA / jnp.sqrt(values.size)
    variance_limit = 5 * variance * jnp.sqrt(2 / (values.size - 1))
    assert jnp.abs(empirical_mean - mean) <= mean_limit + 1.0e-7
    assert jnp.abs(empirical_variance - variance) <= variance_limit + 1.0e-9


@pytest.mark.slow
def test_4096_key_mean_pathwise_gradients_match_analytic_expectations():
    _, params = _mapped_primitives()
    affine = params["affine"]
    x = jnp.array([0.3, -0.7], dtype=jnp.float32)

    def one(key):
        def objective(input_x, parameter):
            output = TorxOps.create(key, config=_config()).linear(
                input_x, parameter, name="oracle.affine"
            )
            return 0.5 * jnp.sum(output**2)

        return jax.grad(objective, argnums=(0, 1))(x, affine)

    samples = jax.jit(jax.vmap(one))(jax.random.split(jax.random.key(18), 4096))
    weight = affine["mean"]["weight"]
    bias = affine["mean"]["bias"]
    mean = x @ weight + bias
    sigma = sigma_from_rho(affine["rho"])
    dsigma = jax.nn.sigmoid(affine["rho"])
    analytic = (
        mean @ weight.T,
        {
            "mean": {"weight": x[:, None] * mean[None, :], "bias": mean},
            "rho": sigma * dsigma,
        },
    )
    for observed, expected in zip(
        jax.tree_util.tree_leaves(samples), jax.tree_util.tree_leaves(analytic), strict=True
    ):
        empirical = jnp.mean(observed, axis=0)
        sem = jnp.std(observed, axis=0, ddof=1) / jnp.sqrt(observed.shape[0])
        assert jnp.all(jnp.abs(empirical - expected) <= 5 * sem + 2.0e-6)


@pytest.mark.slow
def test_equal_local_variance_across_q_occurrences_without_depth_normalization():
    _, params = _mapped_primitives()
    x = jnp.array([[0.3, -0.7]], dtype=jnp.float32)
    roots = jax.random.split(jax.random.key(19), 4096)

    def residual(root, cycle):
        base = TorxOps.create(root, config=_config()).with_scope("mode:draft")
        ops = base.with_occurrence(recurrence_cycle=cycle, physical_layer=0)
        noisy = ops.linear(x, params["affine"], name="dit.block_0.q")
        clean_ops = replace(ops, config=_config(0.0))
        clean = clean_ops.linear(x, params["affine"], name="dit.block_0.q")
        return noisy[0, 0] - clean[0, 0]

    residuals = jax.jit(jax.vmap(lambda root: jnp.stack([residual(root, q) for q in range(3)])))(
        roots
    )
    variances = jnp.var(residuals, axis=0, ddof=1)
    expected = PHASE_D_INITIAL_SIGMA**2
    limit = 5 * expected * jnp.sqrt(2 / (roots.shape[0] - 1)) + 1.0e-9
    assert jnp.all(jnp.abs(variances - expected) <= limit)


def test_full_model_shared_frontend_and_q_residuals_are_distinct_and_reproducible():
    cfg = replace(
        REFERENCE_SMALL_V0,
        carrier=replace(REFERENCE_SMALL_V0.carrier, C=8, h_dim=16, L_max=2),
        packing=replace(REFERENCE_SMALL_V0.packing, M_max=4, K=2),
        model=replace(
            REFERENCE_SMALL_V0.model,
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
            physical_blocks_L=1,
            cycles_Q=3,
            d_dec=16,
            decoder_layers=1,
            mamba_expand=1,
            mamba_state_dim=4,
        ),
    )
    deterministic = init_model_params(jax.random.key(20), cfg)
    params, _ = deterministic_to_torx(deterministic)
    values = jnp.arange(1, 9, dtype=jnp.int32)[None]
    mask = jnp.ones_like(values, dtype=bool)

    def run():
        records = []

        def observe(kind, name, path, key, mean, output):
            if name in ("frontend.byte_embed", "dit.block_0.q"):
                records.append((kind, name, path, jax.random.key_data(key), mean, output - mean))

        apply_model(
            params,
            values,
            mask,
            values,
            mask,
            config=cfg,
            ops=TorxOps.create(jax.random.key(21), config=_config(), sample_observer=observe),
        )
        return records

    first = run()
    second = run()
    assert len(first) == 5  # prompt + clean target frontend, then Q=3 block-0 q
    for left, right in zip(first, second, strict=True):
        assert left[:3] == right[:3]
        assert jnp.array_equal(left[3], right[3])
        assert jnp.array_equal(left[4], right[4])
        assert jnp.array_equal(left[5], right[5])
    frontend = [record for record in first if record[1] == "frontend.byte_embed"]
    q_records = [record for record in first if record[1] == "dit.block_0.q"]
    assert frontend[0][2] == "prompt/frontend.byte_embed"
    assert frontend[1][2] == "target/frontend.byte_embed"
    assert not jnp.array_equal(frontend[0][5], frontend[1][5])
    assert all(not jnp.array_equal(record[5], jnp.zeros_like(record[5])) for record in frontend)
    assert len({tuple(map(int, record[3])) for record in q_records}) == 3
    assert all(not jnp.array_equal(record[5], jnp.zeros_like(record[5])) for record in q_records)
    assert all(
        not jnp.array_equal(left[5], right[5])
        for left, right in zip(q_records, q_records[1:], strict=False)
    )
    assert params["encoder"]["byte_embed"]["mean"] is deterministic["encoder"]["byte_embed"]
    assert all(
        mapped is source
        for mapped, source in zip(
            jax.tree_util.tree_leaves(params["dit"]["blocks"][0]["q"]["mean"]),
            jax.tree_util.tree_leaves(deterministic["dit"]["blocks"][0]["q"]),
            strict=True,
        )
    )
