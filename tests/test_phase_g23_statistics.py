"""Decision and checkpoint-selection regressions, with no neural execution."""

import numpy as np
from typing import Any

from science_phase_g23 import contrast, seed_interval
from run_phase_g23 import mean_roots


def records(accuracy, loss):
    return [
        {
            "query_accuracy": np.full(912, accuracy),
            "query_loss": np.full(912, loss),
            "query_nll": np.full(912, loss),
        }
        for _ in range(3)
    ]


def test_comparative_benefit_neutral_negative_and_no_zero_floor_invention():
    import jax.numpy as jnp

    data = {"q": np.zeros(912), "length": jnp.asarray(np.repeat(np.arange(8, 65), 16)[:, None])}
    baseline = records(0.5, 2.0)
    assert contrast(records(0.52, 1.9), baseline, data, "test-benefit")["outcome"] == "BENEFIT"
    assert contrast(records(0.48, 2.1), baseline, data, "test-negative")["outcome"] == "NEGATIVE"
    assert contrast(records(0.5, 2.0), baseline, data, "test-neutral")["outcome"] == "NEUTRAL"
    assert (
        contrast(records(1.0, 0.0), records(1.0, 0.0), data, "test-zero")["outcome"] == "UNRESOLVED"
    )
    # Seed variability cannot be replaced by an arbitrarily narrow example CI.
    mixed = records(0.52, 1.9)
    mixed[2] = records(0.4, 3.0)[0]
    assert contrast(mixed, baseline, data, "test-conflict")["outcome"] == "UNRESOLVED"


def test_root_means_do_not_treat_missing_overlap_as_zero_nll():
    a = {
        "query_nll": np.array([0.0]),
        "query_nll_support": np.array([0]),
        "query_loss": np.array([5.545]),
    }
    b = {
        "query_nll": np.array([2.0]),
        "query_nll_support": np.array([3]),
        "query_loss": np.array([2.0]),
    }
    mean = mean_roots([a, b])
    assert mean["query_nll"][0] == 2.0
    assert mean["query_loss"][0] == (5.545 + 2.0) / 2


def test_training_seed_interval_uses_three_not_example_count():
    r = seed_interval([0.0, 1.0, 2.0])
    assert r["n_training_seeds"] == 3 and r["df"] == 2
    np.testing.assert_allclose(
        r["ci95"], [1 - 4.302652729911275 / np.sqrt(3), 1 + 4.302652729911275 / np.sqrt(3)]
    )


def test_shards_bind_exact_checkpoint_examples_and_optimizer(tmp_path, monkeypatch):
    import json
    import pickle
    import jax.numpy as jnp
    import pytest
    import run_phase_g23 as g

    monkeypatch.setattr(g, "RUN", tmp_path / "run")
    monkeypatch.setattr(g, "IDENTITY", tmp_path / "identity.json")
    monkeypatch.setattr(g, "audit", lambda: "id")
    g.IDENTITY.write_text(
        json.dumps({"binding": {"datasets": {"ordered_sha256": {"validation": "dataset"}}}})
    )
    params = {"x": jnp.asarray([1.0])}
    checkpoint = g.RUN / "seed0/B/snapshots/update_00000.pkl"
    checkpoint.parent.mkdir(parents=True)
    state = {
        "step": 0,
        "seed": 0,
        "arm": "B",
        "campaign_identity": "id",
        "params": params,
        "moments": (params, params),
    }
    with checkpoint.open("wb") as f:
        pickle.dump(state, f)
    # This is an I/O/provenance unit stub, NOT full-model integration evidence.
    monkeypatch.setitem(
        g.OBJECTIVES,
        "B",
        lambda *args: (
            jnp.asarray(0.0),
            {
                "executed": jnp.asarray(True),
                "valid": jnp.asarray([True]),
                "query_accuracy": jnp.asarray([0.5]),
            },
        ),
    )
    data = {"q": jnp.asarray([0]), "global_example_id": jnp.asarray([10000])}
    args: dict[str, Any] = dict(seed=0, arm="B", step=0, split="validation", mode="argmax")
    g.evaluate(params, data, **args)
    with pytest.raises(RuntimeError):
        g.evaluate({"x": jnp.asarray([2.0])}, data, **args)
    with pytest.raises(RuntimeError):
        g.evaluate(params, {**data, "global_example_id": jnp.asarray([10001])}, **args)
    state["moments"] = ({"x": jnp.asarray([3.0])}, params)
    with checkpoint.open("wb") as f:
        pickle.dump(state, f)
    with pytest.raises(RuntimeError):
        g.evaluate(params, data, **args)
