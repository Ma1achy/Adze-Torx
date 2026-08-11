"""Phase C.1 stochastic occurrence and zero-noise hardening contracts."""

from __future__ import annotations

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
from adze_t.backends.torx import TorxOperatorConfig, TorxOps
from adze_t.config import REFERENCE_SMALL_V0
from adze_t.model import apply_model, init_model_params
from adze_t.objectives import loss_components, total_loss
from adze_t.parity import compare_ordered_model_traces


def _batch() -> tuple[jax.Array, jax.Array]:
    values = jnp.arange(1, 9, dtype=jnp.int32)[None, :]
    return values, jnp.ones_like(values, dtype=bool)


def _enabled_zero_noise(key: jax.Array, **kwargs) -> TorxOps:
    return TorxOps.create(
        key,
        config=TorxOperatorConfig(operator_stochasticity=True, lambda_op=0.0),
        **kwargs,
    )


def _key_tuple(key: jax.Array) -> tuple[int, int]:
    data = jax.random.key_data(key)
    return int(data[0]), int(data[1])


def _record_occurrences(params, key):
    records: list[tuple[str, str, str, tuple[int, int]]] = []

    def observe(kind, name, occurrence_path, occurrence_key):
        records.append((kind, name, occurrence_path, _key_tuple(occurrence_key)))

    values, mask = _batch()
    apply_model(
        params,
        values,
        mask,
        values,
        mask,
        ops=_enabled_zero_noise(key, occurrence_observer=observe),
    )
    return records


def test_shared_frontend_same_mean_distinct_reproducible_branch_occurrences():
    deterministic = init_model_params(jax.random.key(100))
    torx_params, entries = deterministic_to_torx(deterministic)
    counts = parameter_counts(entries)
    assert counts == {
        "deterministic": 2_268_245,
        "torx_mean": 2_268_245,
        "torx_stochastic": 18_901,
        "torx_total": 2_287_146,
    }

    byte_mean = torx_params["encoder"]["byte_embed"]["mean"]
    assert byte_mean is deterministic["encoder"]["byte_embed"]
    byte_entries = [
        entry
        for entry in entries
        if entry.deterministic_path == "encoder/byte_embed" and entry.role == "mean"
    ]
    assert len(byte_entries) == 1

    first = _record_occurrences(torx_params, jax.random.key(101))

    def frontend(records):
        return [record for record in records if record[1] == "frontend.byte_embed"]

    first_frontend = frontend(first)
    assert [(record[2]) for record in first_frontend] == [
        "prompt/frontend.byte_embed",
        "target/frontend.byte_embed",
    ]
    assert first_frontend[0][3] != first_frontend[1][3]
    same_root = _enabled_zero_noise(jax.random.key(101))
    repeated_keys = [
        _key_tuple(same_root.with_scope(scope).context.key_for("frontend.byte_embed"))
        for scope in ("prompt", "target")
    ]
    assert repeated_keys == [record[3] for record in first_frontend]
    changed_root = _enabled_zero_noise(jax.random.key(102))
    changed_keys = [
        _key_tuple(changed_root.with_scope(scope).context.key_for("frontend.byte_embed"))
        for scope in ("prompt", "target")
    ]
    assert changed_keys[0] != first_frontend[0][3]
    assert changed_keys[1] != first_frontend[1][3]


def test_q3_reuses_block_means_but_has_three_distinct_reproducible_keys():
    deterministic = init_model_params(jax.random.key(110))
    torx_params, entries = deterministic_to_torx(deterministic)
    records = _record_occurrences(torx_params, jax.random.key(111))

    for parameter_name in ("dit.block_0.q", "dit.block_0.modulation"):
        selected = [record for record in records if record[1] == parameter_name]
        assert len(selected) == 3
        assert len({record[3] for record in selected}) == 3
        base = _enabled_zero_noise(jax.random.key(111)).with_scope("mode:draft")
        repeated_keys = [
            _key_tuple(
                base.with_occurrence(recurrence_cycle=cycle, physical_layer=0).context.key_for(
                    parameter_name
                )
            )
            for cycle in range(3)
        ]
        assert repeated_keys == [record[3] for record in selected]

    q_mean = torx_params["dit"]["blocks"][0]["q"]["mean"]
    modulation_mean = torx_params["dit"]["blocks"][0]["modulation"]["mean"]
    assert q_mean["weight"] is deterministic["dit"]["blocks"][0]["q"]["weight"]
    assert modulation_mean["weight"] is deterministic["dit"]["blocks"][0]["modulation"]["weight"]

    q1_config = replace(
        REFERENCE_SMALL_V0,
        model=replace(REFERENCE_SMALL_V0.model, cycles_Q=1),
    )
    _, q1_entries = deterministic_to_torx(init_model_params(jax.random.key(110), q1_config))
    assert parameter_counts(q1_entries) == parameter_counts(entries)


def test_zero_noise_scopes_preserve_factor_forward_and_key_rho_invariance():
    deterministic = init_model_params(jax.random.key(120))
    torx_params, _ = deterministic_to_torx(deterministic)
    changed_rho = jax.tree_util.tree_map_with_path(
        lambda path, value: (
            jnp.full_like(value, 75.0) if "['rho']" in jax.tree_util.keystr(path) else value
        ),
        torx_params,
    )
    values, mask = _batch()
    factor_calls: list[tuple[str, str]] = []
    expected = apply_model(deterministic, values, mask, values, mask, ops=DeterministicOps())
    actual = apply_model(
        torx_params,
        values,
        mask,
        values,
        mask,
        ops=_enabled_zero_noise(
            jax.random.key(121), observer=lambda kind, name: factor_calls.append((kind, name))
        ),
    )
    changed = apply_model(
        changed_rho,
        values,
        mask,
        values,
        mask,
        ops=_enabled_zero_noise(jax.random.key(999)),
    )
    assert compare_ordered_model_traces(expected, actual)[1] is None
    assert compare_ordered_model_traces(actual, changed)[1] is None
    assert len(factor_calls) == 237

    def compiled(p, key):
        return apply_model(
            p,
            values,
            mask,
            values,
            mask,
            ops=_enabled_zero_noise(key),
        )["byte_logits"]

    run = jax.jit(compiled)
    key_a = run(torx_params, jax.random.key(122))
    key_b = run(changed_rho, jax.random.key(123))
    assert jnp.array_equal(key_a, key_b)


@pytest.mark.slow
def test_full_raw_gradient_parity_on_stochastic_capable_zero_noise_path():
    deterministic = init_model_params(jax.random.key(130))
    torx_params, _ = deterministic_to_torx(deterministic)
    values, mask = _batch()

    def deterministic_objective(params):
        components = loss_components(
            apply_model(params, values, mask, values, mask, ops=DeterministicOps())
        )
        return total_loss(components, REFERENCE_SMALL_V0), components

    def torx_objective(params):
        components = loss_components(
            apply_model(
                params,
                values,
                mask,
                values,
                mask,
                ops=_enabled_zero_noise(jax.random.key(131)),
            )
        )
        return total_loss(components, REFERENCE_SMALL_V0), components

    (d_loss, d_components), d_grad = jax.jit(
        jax.value_and_grad(deterministic_objective, has_aux=True)
    )(deterministic)
    (t_loss, t_components), t_grad = jax.jit(jax.value_and_grad(torx_objective, has_aux=True))(
        torx_params
    )
    assert jnp.array_equal(d_loss, t_loss)
    assert all(
        bool(value)
        for value in jax.tree_util.tree_leaves(
            jax.tree.map(jnp.array_equal, d_components, t_components)
        )
    )
    assert all(
        bool(value)
        for value in jax.tree_util.tree_leaves(
            jax.tree.map(jnp.array_equal, d_grad, torx_means_to_deterministic(t_grad))
        )
    )
    rho_gradients = [
        leaf
        for path, leaf in jax.tree_util.tree_leaves_with_path(t_grad)
        if "['rho']" in jax.tree_util.keystr(path)
    ]
    assert rho_gradients
    assert all(jnp.array_equal(leaf, jnp.zeros_like(leaf)) for leaf in rho_gradients)
