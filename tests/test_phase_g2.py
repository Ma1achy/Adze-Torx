"""Generated G2 inference, length alignment, estimator ownership and recovery."""

import json

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from adze_t.phase_g2 import aligned_metrics, inference, objective, named_key, train_step
from adze_t.phase_g2_storage import RollingJournal
from adze_t.objectives import adamw_init
from adze_t.state import CommittedStructure
from test_phase_g11_teacher import fixture as teacher_fixture, same


def fixture():
    cfg, p, b = teacher_fixture()
    return cfg, p, {**b, "peer": jnp.asarray([1])}


def test_global_query_alignment_missing_extra_zero_and_teacher_parity():
    labels = jnp.asarray([[[1, 2], [3, 4], [5, 6]]])
    mask = jnp.ones((1, 3, 2), bool)
    logits = (
        jnp.full((1, 3, 2, 256), -10.0)
        .at[
            jnp.arange(1)[:, None, None],
            jnp.arange(3)[None, :, None],
            jnp.arange(2)[None, None, :],
            labels,
        ]
        .set(10.0)
    )
    correct = aligned_metrics(logits, jnp.asarray([[2, 2, 2]]), labels, mask, jnp.asarray([1]))
    assert float(correct["query_accuracy"][0]) == 1
    zero = aligned_metrics(logits, jnp.zeros((1, 3), jnp.int32), labels, mask, jnp.asarray([1]))
    np.testing.assert_allclose(zero["query_loss"], np.log(256), rtol=1e-6)
    assert int(zero["query_nll_support"][0]) == 0
    assert float(zero["query_accuracy"][0]) == 0
    # Deleting a predecessor shifts emitted bytes globally: never compare
    # carrier1/slot1 to itself when its global position has moved.
    shifted = aligned_metrics(logits, jnp.asarray([[1, 2, 2]]), labels, mask, jnp.asarray([1]))
    assert float(shifted["query_accuracy"][0]) == 0
    assert all(np.isfinite(x).all() for x in jax.tree.leaves(shifted))


@pytest.mark.parametrize("policy", ["B", "C"])
def test_generated_api_rejects_teacher_and_is_metadata_independent(policy):
    cfg, p, b = fixture()
    key = named_key("test_structure")

    def fn(p, h, prompt, ids):
        return inference(
            p, h, prompt, ids, key, named_key("operator"), 0, policy=policy, config=cfg
        )

    first = fn(p, b["source_h"], b["prompt"], b["global_example_id"])
    compiled = jax.jit(fn)(p, b["source_h"], b["prompt"], b["global_example_id"])
    assert bool(first.executed) and bool(compiled.executed)
    np.testing.assert_allclose(first.logits, compiled.logits, atol=1e-5)
    with pytest.raises(ValueError):
        inference(
            p,
            b["source_h"],
            b["prompt"],
            b["global_example_id"],
            key,
            key,
            0,
            policy=policy,
            config=cfg,
            teacher=CommittedStructure(b["boundary"], b["length"] > 0, b["length"]),
        )
    _, m = objective(p, b, 0, key, key, policy=policy, config=cfg)
    changed = {
        **b,
        "target_h": b["target_h"] + 20,
        "slots": b["slots"] + 1,
        "boundary": 1 - b["boundary"],
        "length": jnp.zeros_like(b["length"]),
        "slot_mask": ~b["slot_mask"],
    }
    _, other = objective(p, changed, 0, key, key, policy=policy, config=cfg)
    assert same((m["b"], m["ell"]), (other["b"], other["ell"]))
    assert first.commit.sampled.boundary_log_prob[0, -1] == 0


def test_score_connection_changes_real_model_gradient_and_update_is_finite():
    cfg, p, b = fixture()
    key = named_key("test_score")

    def loss(p, detach):
        return objective(p, b, 0, key, key, policy="C", config=cfg, detach_score=detach)[0]

    connected = jax.jit(jax.grad(lambda p: loss(p, False)))(p)
    detached = jax.jit(jax.grad(lambda p: loss(p, True)))(p)
    difference = sum(
        float(jnp.sum((x - y) ** 2))
        for x, y in zip(jax.tree.leaves(connected), jax.tree.leaves(detached), strict=True)
    )
    assert difference > 1e-8
    moments = (adamw_init(p), adamw_init(p))
    updated, m, step, metric = jax.jit(
        lambda p, m, b: train_step(p, m, 0, b, key, key, policy="C", config=cfg)
    )(p, moments, b)
    assert int(step) == 1 and bool(metric["executed"])
    assert all(np.isfinite(x).all() for x in jax.tree.leaves((updated, m)))


def test_rolling_journal_recovery_snapshot_immutability_and_mismatch(tmp_path):
    journal = RollingJournal(tmp_path, "version")

    def state(n):
        return {
            "identity": "version",
            "step": n,
            "params": {"x": np.asarray([n], np.float32)},
            "moments": ({"x": np.asarray([n], np.float32)}, {"x": np.asarray([n], np.float32)}),
        }

    for n in range(4):
        journal.append(state(n), {"executed": True}, snapshot=n in (0, 2))
    assert len(list(tmp_path.glob("slot_*.pkl"))) == 2
    with (tmp_path / "snapshots/update_00000.pkl").open("rb") as stream:
        import pickle

        assert pickle.load(stream)["step"] == 0
    (tmp_path / "LATEST.json").write_text("{}")
    recovered = RollingJournal(tmp_path, "version").recover()
    assert recovered is not None and recovered["step"] == 3
    with pytest.raises(ValueError):
        RollingJournal(tmp_path, "different").recover()
    with pytest.raises(ValueError):
        journal.append(state(3), {})
    with pytest.raises(ValueError):
        journal.append(state(4), {"executed": False})
    assert json.loads((tmp_path / "LATEST.json").read_text())["step"] == 3


def test_receipt_corruption_and_uncommitted_payload_are_not_successes(tmp_path):
    import pickle

    journal = RollingJournal(tmp_path, "v")

    def state(n):
        return {
            "identity": "v",
            "step": n,
            "params": {"x": np.asarray([n], np.float32)},
            "moments": (np.asarray([0.0]), np.asarray([0.0])),
        }

    journal.append(state(0), {"executed": True}, snapshot=True)
    with (tmp_path / "slot_1.pkl").open("wb") as stream:
        pickle.dump(state(1), stream)
    recovered = RollingJournal(tmp_path, "v").recover()
    assert recovered is not None and recovered["step"] == 0
    assert len(list(tmp_path.glob("orphan_*.pkl"))) == 1
    receipt = tmp_path / "receipts/00000.json"
    value = json.loads(receipt.read_text())
    value["metrics"]["executed"] = False
    receipt.write_text(json.dumps(value))
    with pytest.raises(ValueError):
        RollingJournal(tmp_path, "v").recover()
