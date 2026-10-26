"""Behavioral regressions for the authorized teacher-only bounded run."""

import json
from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from adze_t.objectives import adamw_init
from adze_t.phase_g11_records import Journal, nonquery_guard, paired_bootstrap
from adze_t.phase_g11_training import teacher_objective, train_step
from test_phase_g_integration import _fixture


def fixture():
    cfg, p, old = _fixture()
    batch = {
        "source_h": jax.random.normal(jax.random.key(2), (1, 4, 8)),
        "target_h": jnp.zeros((1, 4, 8)),
        "boundary": old["structure_boundary"],
        "length": old["structure_length"],
        "q": jnp.array([0]),
        "prompt": old["prompt"],
        "global_example_id": old["global_example_id"],
        "slots": old["target"].reshape(1, 4, 2),
        "slot_mask": jnp.ones((1, 4, 2), bool),
    }
    return cfg, p, batch


def same(a, b):
    return all(
        np.array_equal(x, y) for x, y in zip(jax.tree.leaves(a), jax.tree.leaves(b), strict=True)
    )


def test_guard_is_conjunction_and_inclusive():
    base = {"nonquery_accuracy": 0.5, "nonquery_nll": 2.0}
    assert nonquery_guard({"nonquery_accuracy": 0.48, "nonquery_nll": 2.1}, base)["passed"]
    assert not nonquery_guard({"nonquery_accuracy": 0.47, "nonquery_nll": 1.0}, base)["passed"]
    assert not nonquery_guard({"nonquery_accuracy": 0.6, "nonquery_nll": 2.11}, base)["passed"]


def test_stratified_bootstrap_preserves_pairs_and_rejects_unbalanced():
    lengths = np.repeat(np.arange(8, 65), 16)
    b = np.linspace(1, 2, 912)
    a = b + 0.25
    result = paired_bootstrap(a, b, lengths, seed=1161)
    np.testing.assert_allclose(result["delta_ci95"], [0.25, 0.25])
    assert result == paired_bootstrap(a, b, lengths, seed=1161)
    with pytest.raises(ValueError):
        paired_bootstrap(a[:-1], b[:-1], lengths[:-1], seed=1161)
    zero = paired_bootstrap(np.ones(912), np.zeros(912), lengths, seed=1161)
    assert zero["relative"] is None and zero["relative_ci95"] is None
    assert zero["delta_ci95"] == [1.0, 1.0]


def test_journal_roundtrip_orphan_stale_pointer_and_protocol(tmp_path):
    journal = Journal(tmp_path, "protocol")

    def state(n):
        return {
            "step": n,
            "protocol": "protocol",
            "params": {"x": np.array([n], np.float32)},
            "moments": ({"x": np.array([n], np.float32)}, {"x": np.array([n], np.float32)}),
        }

    journal.append(state(0), {"finite": True})
    (tmp_path / "payloads/001_000.pkl").write_bytes(b"interrupted uncounted attempt")
    recovered = journal.recover()
    assert recovered is not None and recovered["step"] == 0
    journal.append(state(1), {"finite": True})
    (tmp_path / "LATEST.json").write_text('{"step":0}')
    recovered = journal.recover()
    assert recovered is not None and recovered["step"] == 1
    assert json.loads((tmp_path / "LATEST.json").read_text())["step"] == 1
    with pytest.raises(ValueError):
        Journal(tmp_path, "wrong").recover()
    with pytest.raises(ValueError):
        journal.append(state(1), {})
    bad = state(2)
    bad["params"]["x"][0] = np.nan
    with pytest.raises(ValueError):
        journal.append(bad, {})
    assert len(list((tmp_path / "receipts").glob("*.json"))) == 2
    (tmp_path / "payloads/001_001.pkl").write_bytes(b"corrupted")
    with pytest.raises(ValueError):
        journal.recover()


def test_teacher_loss_weights_target_loss_only_and_jit():
    cfg, p, b = fixture()
    loss, m = teacher_objective(p, b, 0, config=cfg)
    np.testing.assert_allclose(
        loss,
        jnp.mean(
            m["b_ce"]
            + m["ell_ce"]
            + 0.5 * m["replay_h_mse"]
            + m["global_byte_loss"]
            + m["query_nll"]
        ),
        rtol=1e-6,
    )
    _, other = teacher_objective(p, {**b, "target_h": b["target_h"] + 10}, 0, config=cfg)
    for key in ("query_nll", "query_accuracy", "predicted_b", "predicted_l"):
        assert same(m[key], other[key])
    compiled = jax.jit(lambda p, b: teacher_objective(p, b, 0, config=cfg))(p, b)
    np.testing.assert_allclose(loss, compiled[0], rtol=1e-5)


@pytest.mark.parametrize("compiled", [False, True])
def test_invalid_and_overflow_leave_full_optimizer_bitwise_unchanged(compiled):
    cfg, p, b = fixture()
    moments = (adamw_init(p), adamw_init(p))
    for config, batch in (
        (cfg, {**b, "boundary": b["boundary"].at[:, -1].set(0)}),
        (replace(cfg, packing=replace(cfg.packing, M_max=1, K=1)), b),
    ):

        def call(p, m, b):
            return train_step(p, m, jnp.array(0), b, config=config)

        fn = jax.jit(call) if compiled else call
        updated, m, s, metrics = fn(p, moments, batch)
        assert int(s) == 0 and not bool(metrics["executed"])
        assert same((p, moments), (updated, m))


def test_valid_prospective_update_changes_decoder_not_frozen_leaves():
    from adze_t.training import stochastic_model_update_mask

    cfg, p, b = fixture()
    moments = (adamw_init(p), adamw_init(p))
    updated, m, s, metrics = jax.jit(lambda p, m, b: train_step(p, m, 0, b, config=cfg))(
        p, moments, b
    )
    assert int(s) == 1 and bool(metrics["executed"])
    assert not same(p["decoder"], updated["decoder"])
    mask = stochastic_model_update_mask(p)
    for before, after, enabled in zip(
        jax.tree.leaves(p), jax.tree.leaves(updated), jax.tree.leaves(mask), strict=True
    ):
        if not bool(enabled):
            assert np.array_equal(before, after)
    assert all(np.isfinite(x).all() for x in jax.tree.leaves((updated, m)))
