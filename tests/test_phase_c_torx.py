"""Phase C public-Torx zero-noise parity contracts."""

from __future__ import annotations

import subprocess
import sys
from dataclasses import replace

import jax
import jax.numpy as jnp
import pytest

from adze_t.backends.deterministic import DeterministicOps
from adze_t.backends.mapping import (
    deterministic_to_torx,
    parameter_counts,
    torx_means_to_deterministic,
)
from adze_t.backends.torx import TorxOperatorConfig, TorxOps, stable_occurrence_id
from adze_t.config import REFERENCE_SMALL_V0
from adze_t.dit import apply_dit
from adze_t.mamba import MambaConfig, apply_mamba_stack, init_mamba_stack
from adze_t.model import _dit_config, apply_model, init_model_params
from adze_t.objectives import loss_components, total_loss
from adze_t.packing import build_pack_metadata_core, pack_values, trim_padding_blocks
from adze_t.parity import compare_ordered_model_traces
from adze_t.teacher import canonical_teacher_structure


def _torx_ops(key: int = 0, *, observer=None, rho_enabled: bool = False) -> TorxOps:
    return TorxOps.create(
        jax.random.key(key),
        config=TorxOperatorConfig(lambda_op=0.0, operator_stochasticity=rho_enabled),
        observer=observer,
    )


def _assert_tree_exact(left, right):
    assert jax.tree_util.tree_structure(left) == jax.tree_util.tree_structure(right)
    for a, b in zip(jax.tree_util.tree_leaves(left), jax.tree_util.tree_leaves(right), strict=True):
        assert jnp.array_equal(a, b)


def _batch():
    values = jnp.arange(1, 9, dtype=jnp.int32)[None, :]
    mask = jnp.ones_like(values, dtype=bool)
    return values, mask


def test_stable_occurrence_id_is_process_independent():
    expected = stable_occurrence_id("dit.block_3.q")
    command = (
        "from adze_t.backends.torx import stable_occurrence_id; "
        "print(stable_occurrence_id('dit.block_3.q'))"
    )
    observed = int(subprocess.check_output([sys.executable, "-c", command], text=True).strip())
    assert observed == expected
    assert 0 <= expected <= 0xFFFFFFFF


def test_tied_mean_occurrences_receive_distinct_q_keys():
    ops = _torx_ops(3)
    q0 = ops.with_occurrence(recurrence_cycle=0, physical_layer=2)
    q1 = ops.with_occurrence(recurrence_cycle=1, physical_layer=2)
    assert not jnp.array_equal(
        q0.context.key_for("dit.block_2.q"), q1.context.key_for("dit.block_2.q")
    )


def test_parameter_mapping_is_semantic_complete_and_round_trips():
    deterministic = init_model_params(jax.random.key(10))
    torx_params, entries = deterministic_to_torx(deterministic)
    _assert_tree_exact(deterministic, torx_means_to_deterministic(torx_params))
    deterministic_paths = [entry.deterministic_path for entry in entries if entry.role == "mean"]
    assert len(deterministic_paths) == len(set(deterministic_paths)) == 237
    assert all(entry.deterministic_path is None for entry in entries if entry.role == "stochastic")
    counts = parameter_counts(entries)
    assert counts["deterministic"] == counts["torx_mean"] == 2_268_245
    assert counts["torx_stochastic"] > 0
    q1 = replace(
        REFERENCE_SMALL_V0,
        model=replace(REFERENCE_SMALL_V0.model, cycles_Q=1),
    )
    q3_params, q3_entries = deterministic_to_torx(deterministic)
    q1_params, q1_entries = deterministic_to_torx(init_model_params(jax.random.key(10), q1))
    assert jax.tree_util.tree_structure(q1_params) == jax.tree_util.tree_structure(q3_params)
    assert parameter_counts(q1_entries) == parameter_counts(q3_entries)
    assert not any("/q0/" in entry.torx_path or "/q1/" in entry.torx_path for entry in q3_entries)


@pytest.mark.parametrize("key_a,key_b", [(1, 2), (12, 999)])
def test_affine_factor_zero_noise_forward_key_rho_and_gradient_parity(key_a, key_b):
    dops = DeterministicOps()
    x = jnp.arange(12, dtype=jnp.float32).reshape(2, 2, 3) / 7
    params = {
        "weight": jnp.arange(15, dtype=jnp.float32).reshape(3, 5) / 11,
        "bias": jnp.arange(5, dtype=jnp.float32) / 13,
    }
    torx_params, _ = deterministic_to_torx({"projection": params})
    torx_params = torx_params["projection"]
    expected = dops.linear(x, params, name="primitive.linear")
    actual_a = _torx_ops(key_a).linear(x, torx_params, name="primitive.linear")
    changed = {**torx_params, "rho": jnp.full_like(torx_params["rho"], 80.0)}
    actual_b = _torx_ops(key_b, rho_enabled=True).linear(x, changed, name="primitive.linear")
    assert jnp.array_equal(actual_a, expected)
    assert jnp.array_equal(actual_b, expected)

    d_grad = jax.grad(lambda p: jnp.sum(dops.linear(x, p, name="primitive.linear") ** 2))(params)
    t_grad = jax.grad(
        lambda p: jnp.sum(
            _torx_ops(key_a, rho_enabled=True).linear(x, p, name="primitive.linear") ** 2
        )
    )(torx_params)
    _assert_tree_exact(d_grad, t_grad["mean"])
    assert jnp.array_equal(t_grad["rho"], jnp.zeros_like(t_grad["rho"]))

    compiled = jax.jit(
        lambda p, key: TorxOps.create(
            key, config=TorxOperatorConfig(lambda_op=jnp.asarray(0.0), operator_stochasticity=True)
        ).linear(x, p, name="primitive.linear")
    )(torx_params, jax.random.key(key_b))
    assert jnp.array_equal(
        compiled, jax.jit(lambda p: dops.linear(x, p, name="primitive.linear"))(params)
    )


def test_embedding_depthwise_and_mean_parameter_primitive_parity():
    dops = DeterministicOps()
    ops = _torx_ops(4)
    embedding = jnp.arange(28, dtype=jnp.float32).reshape(7, 4) / 17
    mapped_embedding, _ = deterministic_to_torx({"table": embedding})
    indices = jnp.array([[0, 4, 6], [1, 3, 5]])
    assert jnp.array_equal(
        dops.embedding(indices, embedding, name="primitive.embedding"),
        ops.embedding(indices, mapped_embedding["table"], name="primitive.embedding"),
    )

    x = jnp.arange(40, dtype=jnp.float32).reshape(2, 5, 4) / 19
    conv = {
        "kernel": jnp.arange(12, dtype=jnp.float32).reshape(3, 4) / 23,
        "bias": jnp.arange(4, dtype=jnp.float32) / 29,
    }
    mapped_conv, _ = deterministic_to_torx({"conv": conv})
    assert jnp.array_equal(
        dops.depthwise_conv1d(x, conv, name="primitive.conv"),
        ops.depthwise_conv1d(x, mapped_conv["conv"], name="primitive.conv"),
    )

    direct = jnp.arange(8, dtype=jnp.float32).reshape(2, 4)
    mapped_direct, _ = deterministic_to_torx({"a_log": direct})
    assert jnp.array_equal(direct, ops.parameter(mapped_direct["a_log"], name="primitive.a_log"))


@pytest.mark.parametrize(
    "mask",
    [
        [[1, 1, 1, 1, 1]],
        [[1, 1, 1, 0, 0]],
        [[0, 1, 1, 1, 1]],
        [[1, 0, 1, 1, 1]],
        [[1, 0, 0, 1, 1]],
    ],
)
def test_c2a_mamba_stack_parity_for_masks(mask):
    config = MambaConfig(width=8, layers=2, expand=2, state_dim=4, conv_kernel=3)
    deterministic = init_mamba_stack(jax.random.key(20), config, DeterministicOps(), name="probe")
    torx_tree, _ = deterministic_to_torx({"stack": deterministic})
    x = jax.random.normal(jax.random.key(21), (1, 5, 8))
    valid = jnp.asarray(mask, dtype=bool)
    expected = apply_mamba_stack(
        x, deterministic, config, DeterministicOps(), name="probe", mask=valid
    )
    actual = apply_mamba_stack(
        x, torx_tree["stack"], config, _torx_ops(22), name="probe", mask=valid
    )
    assert jnp.array_equal(actual, expected)
    compiled = jax.jit(
        lambda p, key: apply_mamba_stack(
            x, p, config, TorxOps.create(key), name="probe", mask=valid
        )
    )(torx_tree["stack"], jax.random.key(23))
    expected_compiled = jax.jit(
        lambda p: apply_mamba_stack(x, p, config, DeterministicOps(), name="probe", mask=valid)
    )(deterministic)
    assert jnp.array_equal(compiled, expected_compiled)


def test_c2b_reference_dit_recurrence_parity_and_key_invariance():
    cfg = REFERENCE_SMALL_V0
    deterministic = init_model_params(jax.random.key(30))
    torx_params, _ = deterministic_to_torx(deterministic)
    teacher_bytes, byte_mask = _batch()
    teacher = canonical_teacher_structure(teacher_bytes, byte_mask, cfg)
    metadata = build_pack_metadata_core(
        teacher.boundaries, teacher.activity, M_max=cfg.packing.M_max, K=cfg.packing.K
    )
    metadata = trim_padding_blocks(metadata, 4)
    carrier = jax.random.normal(jax.random.key(31), (1, cfg.carrier.C, cfg.model.d_model))
    packed = pack_values(carrier, metadata)
    context = jax.random.normal(jax.random.key(32), (1, cfg.model.d_ctx))
    expected, expected_aux = apply_dit(
        packed,
        metadata,
        context,
        deterministic["dit"],
        _dit_config(cfg),
        ops=DeterministicOps(),
        observed_b=teacher.boundaries,
        observed_l=teacher.length,
    )
    actual, actual_aux = apply_dit(
        packed,
        metadata,
        context,
        torx_params["dit"],
        _dit_config(cfg),
        ops=_torx_ops(33),
        observed_b=teacher.boundaries,
        observed_l=teacher.length,
    )
    assert jnp.array_equal(actual, expected)
    for name in ("trajectory", "block_trajectory", "mask", "effective_depths"):
        assert jnp.array_equal(actual_aux[name], expected_aux[name])
    assert actual_aux["block_trajectory"].shape[0] == 12
    compiled_t = jax.jit(
        lambda p, key: apply_dit(
            packed,
            metadata,
            context,
            p,
            _dit_config(cfg),
            ops=TorxOps.create(key),
            observed_b=teacher.boundaries,
            observed_l=teacher.length,
        )[0]
    )(torx_params["dit"], jax.random.key(34))
    compiled_d = jax.jit(
        lambda p: apply_dit(
            packed,
            metadata,
            context,
            p,
            _dit_config(cfg),
            ops=DeterministicOps(),
            observed_b=teacher.boundaries,
            observed_l=teacher.length,
        )[0]
    )(deterministic["dit"])
    assert jnp.array_equal(compiled_t, compiled_d)


def test_c3_full_model_zero_noise_factor_trace_and_eager_parity():
    deterministic = init_model_params(jax.random.key(40))
    torx_params, _ = deterministic_to_torx(deterministic)
    values, mask = _batch()
    calls: list[tuple[str, str]] = []
    expected = apply_model(deterministic, values, mask, values, mask, ops=DeterministicOps())
    actual = apply_model(
        torx_params,
        values,
        mask,
        values,
        mask,
        ops=_torx_ops(41, observer=lambda kind, name: calls.append((kind, name))),
    )
    ordered = (
        "prompt_frontend",
        "context_seq",
        "context_global",
        "proposal",
        "packed_carrier",
        "packed_output",
        "unpooled_carrier",
        "pre_head_carrier",
        "carrier",
        "byte_logits",
    )
    for name in ordered:
        _assert_tree_exact(expected[name], actual[name])
    _assert_tree_exact(expected["metadata"], actual["metadata"])
    _assert_tree_exact(
        expected["dit_aux"]["block_trajectory"], actual["dit_aux"]["block_trajectory"]
    )
    assert len(calls) > 200
    names = {name for _, name in calls}
    assert len({stable_occurrence_id(name) for name in names}) == len(names)
    assert sum(name == "frontend.byte_embed" for _, name in calls) == 2
    assert {
        "frontend.byte_embed",
        "frontend.layer_0.in_proj",
        "frontend.layer_0.conv",
        "frontend.layer_0.dbc_proj",
        "frontend.layer_0.a_log",
        "context.input_proj",
        "target.pool",
        "proposal.context",
        "proposal.layer_0.dbc_proj",
        "model.carrier_input",
        "dit.input_proj",
        "dit.conditioning_trunk",
        "dit.block_0.modulation",
        "dit.block_0.q",
        "dit.block_0.k",
        "dit.block_0.v",
        "dit.block_0.o",
        "dit.block_3.ffn_up",
        "dit.block_3.ffn_gate",
        "dit.block_3.ffn_down",
        "dit.output_proj",
        "model.carrier_output",
        "model.h_head",
        "model.b_head",
        "model.l_head",
        "decoder.stack.layer_0.dbc_proj",
        "decoder.out",
    } <= names
    metrics, first_divergence = compare_ordered_model_traces(expected, actual)
    assert first_divergence is None
    assert [metric.path for metric in metrics][11:15] == [
        "dit/q0/b0",
        "dit/q0/b1",
        "dit/q0/b2",
        "dit/q0/b3",
    ]


@pytest.mark.parametrize("mode", ["draft", "refine"])
@pytest.mark.parametrize("case", ["inactive_holes", "all_inactive", "padded_slots"])
def test_c2b_dit_edge_structure_parity(mode, case):
    cfg = REFERENCE_SMALL_V0
    deterministic = init_model_params(jax.random.key(70))
    torx_params, _ = deterministic_to_torx(deterministic)
    carrier_count = cfg.carrier.C
    if case == "padded_slots":
        boundaries = jnp.zeros((1, carrier_count), dtype=jnp.int32)
        boundaries = boundaries.at[:, 4::5].set(1).at[:, -1].set(1)
        n_blocks = 7
        activity = jnp.ones((1, carrier_count), dtype=bool)
    else:
        boundaries = jnp.zeros((1, carrier_count), dtype=jnp.int32)
        boundaries = boundaries.at[:, cfg.packing.K - 1 :: cfg.packing.K].set(1).at[:, -1].set(1)
        n_blocks = 4
        activity = jnp.ones((1, carrier_count), dtype=bool)
        if case == "inactive_holes":
            activity = activity.at[:, 2:4].set(False).at[:, 10].set(False)
        else:
            activity = jnp.zeros_like(activity)
    metadata = trim_padding_blocks(
        build_pack_metadata_core(boundaries, activity, M_max=cfg.packing.M_max, K=cfg.packing.K),
        n_blocks,
    )
    carrier = jax.random.normal(jax.random.key(71), (1, carrier_count, cfg.model.d_model))
    packed = pack_values(carrier, metadata)
    context = jax.random.normal(jax.random.key(72), (1, cfg.model.d_ctx))
    observed_l = jnp.zeros_like(boundaries)
    expected, expected_aux = apply_dit(
        packed,
        metadata,
        context,
        deterministic["dit"],
        _dit_config(cfg),
        ops=DeterministicOps(),
        mode=mode,
        observed_b=boundaries,
        observed_l=observed_l,
    )
    actual, actual_aux = apply_dit(
        packed,
        metadata,
        context,
        torx_params["dit"],
        _dit_config(cfg),
        ops=_torx_ops(73),
        mode=mode,
        observed_b=boundaries,
        observed_l=observed_l,
    )
    assert jnp.array_equal(actual, expected)
    assert jnp.array_equal(actual_aux["block_trajectory"], expected_aux["block_trajectory"])
    if case == "all_inactive":
        assert not jnp.any(actual_aux["mask"])


@pytest.mark.slow
def test_c3_full_model_jit_loss_and_raw_gradient_parity():
    cfg = REFERENCE_SMALL_V0
    deterministic = init_model_params(jax.random.key(50))
    torx_params, _ = deterministic_to_torx(deterministic)
    values, mask = _batch()

    def d_objective(p):
        return total_loss(
            loss_components(
                apply_model(p, values, mask, values, mask, config=cfg, ops=DeterministicOps())
            ),
            cfg,
        )

    def t_objective(p, key):
        return total_loss(
            loss_components(
                apply_model(p, values, mask, values, mask, config=cfg, ops=TorxOps.create(key))
            ),
            cfg,
        )

    d_loss, d_grad = jax.jit(jax.value_and_grad(d_objective))(deterministic)
    t_loss, t_grad = jax.jit(jax.value_and_grad(t_objective))(torx_params, jax.random.key(51))
    assert jnp.allclose(t_loss, d_loss, atol=1e-5, rtol=1e-5)
    _assert_tree_exact(d_grad, torx_means_to_deterministic(t_grad))
    rho_grads = [
        leaf
        for path, leaf in jax.tree_util.tree_leaves_with_path(t_grad)
        if "['rho']" in jax.tree_util.keystr(path)
    ]
    assert rho_grads
    assert all(jnp.array_equal(leaf, jnp.zeros_like(leaf)) for leaf in rho_grads)


@pytest.mark.parametrize(
    "prompt,target,prompt_mask,target_mask",
    [
        (
            [[0, 1, 2, 3, 4, 5, 6, 7]],
            [[7, 6, 5, 4, 3, 2, 1, 0]],
            [[1, 1, 1, 1, 1, 1, 1, 1]],
            [[1, 1, 1, 1, 1, 1, 1, 1]],
        ),
        (
            [[9, 8, 7, 6, 5, 0, 0, 0]],
            [[5, 6, 7, 8, 9, 0, 0, 0]],
            [[1, 1, 1, 1, 1, 0, 0, 0]],
            [[1, 1, 1, 1, 1, 0, 0, 0]],
        ),
    ],
)
def test_full_model_zero_byte_reverse_and_padding_parity(prompt, target, prompt_mask, target_mask):
    deterministic = init_model_params(jax.random.key(80))
    torx_params, _ = deterministic_to_torx(deterministic)
    prompt = jnp.asarray(prompt, dtype=jnp.int32)
    target = jnp.asarray(target, dtype=jnp.int32)
    prompt_mask = jnp.asarray(prompt_mask, dtype=bool)
    target_mask = jnp.asarray(target_mask, dtype=bool)
    expected = apply_model(
        deterministic, prompt, prompt_mask, target, target_mask, ops=DeterministicOps()
    )
    actual = apply_model(torx_params, prompt, prompt_mask, target, target_mask, ops=_torx_ops(81))
    metrics, first_divergence = compare_ordered_model_traces(expected, actual)
    assert first_divergence is None
    assert all(metric.passed for metric in metrics)
    _assert_tree_exact(expected["metadata"], actual["metadata"])


def test_full_phase_b_loss_components_parity():
    deterministic = init_model_params(jax.random.key(90))
    torx_params, _ = deterministic_to_torx(deterministic)
    values, mask = _batch()
    d_components = loss_components(
        apply_model(deterministic, values, mask, values, mask, ops=DeterministicOps())
    )
    t_components = loss_components(
        apply_model(torx_params, values, mask, values, mask, ops=_torx_ops(91))
    )
    _assert_tree_exact(d_components, t_components)


def test_full_model_rho_and_key_invariance_under_jit():
    deterministic = init_model_params(jax.random.key(60))
    torx_params, _ = deterministic_to_torx(deterministic)
    changed_rho = jax.tree_util.tree_map_with_path(
        lambda path, value: (
            jnp.full_like(value, 50.0) if "['rho']" in jax.tree_util.keystr(path) else value
        ),
        torx_params,
    )
    values, mask = _batch()

    def run(p, key):
        return apply_model(
            p,
            values,
            mask,
            values,
            mask,
            ops=TorxOps.create(
                key, config=TorxOperatorConfig(lambda_op=0.0, operator_stochasticity=True)
            ),
        )["byte_logits"]

    compiled = jax.jit(run)
    first = compiled(torx_params, jax.random.key(61))
    second = compiled(changed_rho, jax.random.key(999))
    assert jnp.array_equal(first, second)
