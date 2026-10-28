from __future__ import annotations

from dataclasses import replace
import importlib
from pathlib import Path

import jax
import jax.numpy as jnp
import pytest

from test_phase_g_integration import _fixture
from adze_t.objectives import adamw_init
from adze_t.phase_g import (
    generated_train_step,
    generated_two_pass_objective,
    fail_closed_adamw_step,
)


@pytest.mark.parametrize("compiled", [False, True])
def test_amended_update_forwards_full_site_dynamic_capacity_and_fails_closed(compiled):
    cfg, params, batch = _fixture()
    cfg = replace(cfg, packing=replace(cfg.packing, K=2))
    batch = {**batch, "structure_boundary": jnp.asarray([[0, 0, 0, 1]], jnp.int32)}
    moments = (adamw_init(params), adamw_init(params))

    def update(p, m, b):
        return generated_train_step(
            p,
            m,
            0,
            b,
            jax.random.key(1200),
            jax.random.key(1201),
            jax.random.key(1202),
            arm="teacher",
            dynamic_capacity=True,
            config=cfg,
        )

    apply = jax.jit(update) if compiled else update
    new, new_moments, step, metrics = apply(params, moments, batch)
    assert metrics["executed"] and step == 1
    assert metrics["complete_structure_accuracy"] == 1
    assert jnp.array_equal(metrics["example_committed_b"], batch["structure_boundary"])
    invalid = {**batch, "structure_boundary": batch["structure_boundary"].at[:, -1].set(0)}
    stopped, stopped_moments, stopped_step, metric = apply(params, moments, invalid)
    assert not metric["executed"] and stopped_step == 0
    assert jax.tree_util.tree_all(
        jax.tree.map(jnp.array_equal, (params, moments), (stopped, stopped_moments))
    )
    assert all(jnp.isfinite(x).all() for x in jax.tree.leaves((new, new_moments)))


def test_amended_loss_matches_unchanged_frozen_common_descendant_weights():
    cfg, p, b = _fixture()
    loss, m = generated_two_pass_objective(
        p,
        b,
        jax.random.key(1203),
        jax.random.key(1204),
        jax.random.key(1205),
        0,
        arm="teacher",
        dynamic_capacity=True,
        config=cfg,
    )
    assert jnp.allclose(loss, m["common_loss"] + m["descendant_loss"])
    assert jnp.allclose(m["descendant_loss"], 0.5 * m["replay_h_mse"] + m["byte_loss"])


def test_nonfinite_prospective_state_is_atomically_rejected():
    params = {"x": jnp.asarray([1.0])}
    moments = (adamw_init(params), adamw_init(params))
    for apply in (fail_closed_adamw_step, jax.jit(fail_closed_adamw_step)):
        p, m, s, _ = apply(
            params,
            {"x": jnp.asarray([jnp.inf])},
            moments,
            0,
            jnp.asarray(True),
            {"x": jnp.asarray(True)},
        )
        assert s == 0
        assert jax.tree_util.tree_all(jax.tree.map(jnp.array_equal, (params, moments), (p, m)))


def test_controller_does_not_persist_invalid_completed_update(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    runner = importlib.import_module("run_phase_g_amended")
    cfg, p, b = _fixture()
    bad = {**b, "structure_length": b["structure_length"].at[0, 0].set(-1)}
    monkeypatch.setattr(runner, "OUT", tmp_path / "evidence")
    monkeypatch.setattr(runner, "RUN", tmp_path / "run")
    monkeypatch.setattr(runner, "manifest", lambda: {"test": "durability"})
    monkeypatch.setattr(runner, "initialized", lambda: p)
    monkeypatch.setattr(runner, "config", lambda: cfg)
    monkeypatch.setattr(runner, "require", lambda *args: None)
    monkeypatch.setattr(runner, "training_batch", lambda occurrence: bad)
    completed = []
    monkeypatch.setattr(
        runner, "checkpoint", lambda state, metrics: completed.append(state["updates"])
    )
    with pytest.raises(RuntimeError, match="durable state unchanged"):
        runner.train_arm("teacher")
    state = runner.load_pickle(tmp_path / "run/teacher/state.pkl")
    assert state["updates"] == 0 and state["step"] == 0
    assert completed == [0]
    assert not (tmp_path / "run/teacher/update_001.pkl").exists()
    failure = __import__("json").loads(
        (tmp_path / "evidence/teacher/failed_attempt_001.json").read_text()
    )
    assert not failure["completed_update"]


def test_immutable_receipts_and_matching_gate_required(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    runner = importlib.import_module("run_phase_g_amended")
    monkeypatch.setattr(runner, "OUT", tmp_path)
    runner.record(tmp_path / "x.json", {"protocol": {"id": 1}, "passed": True})
    runner.require("x", {"id": 1})
    with pytest.raises(RuntimeError):
        runner.require("x", {"id": 2})
    with pytest.raises(RuntimeError):
        runner.record(tmp_path / "x.json", {"changed": True})


def test_latest_state_rejects_stale_receipt_before_resuming(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    runner = importlib.import_module("run_phase_g_amended")
    path = tmp_path / "state.pkl"
    state = {
        "protocol": {"v": 1},
        "arm": "teacher",
        "step": 0,
        "updates": 0,
        "params": {"x": jnp.asarray([1.0])},
    }
    runner.save_current(path, state)
    assert runner.load_current(path, {"v": 1}, "teacher")["step"] == 0
    runner.save_pickle(path, {**state, "step": 1, "updates": 1})
    with pytest.raises(RuntimeError, match="checksum"):
        runner.load_current(path, {"v": 1}, "teacher")


def test_nonemitting_output_reports_empty_overlap_support():
    cfg, p, b = _fixture()
    p = {
        **p,
        "l_head": {
            **p["l_head"],
            "mean": {
                **p["l_head"]["mean"],
                "weight": jnp.zeros_like(p["l_head"]["mean"]["weight"]),
                "bias": jnp.asarray([100.0, -100.0, -100.0]),
            },
        },
    }
    loss, m = generated_two_pass_objective(
        p,
        b,
        jax.random.key(1206),
        jax.random.key(1207),
        jax.random.key(1208),
        0,
        arm="argmax",
        dynamic_capacity=True,
        config=cfg,
    )
    assert m["executed"] and jnp.isfinite(loss)
    assert m["overlap_example_support"] == 0 and m["overlap_byte_count"] == 0
    assert m["missing_bytes"] == 8 and m["extra_bytes"] == 0
    assert m["byte_accuracy"] == 0
