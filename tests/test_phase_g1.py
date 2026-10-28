"""Behavioral controls for diagnosis, not trainability claims."""

import importlib
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from adze_t.masking import draft_attention_mask
from adze_t.backends.mapping import torx_means_to_deterministic
from test_phase_g_integration import _fixture


def runner(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    return importlib.import_module("run_phase_g1")


def test_draft_cuts_remove_future_not_earlier_peer_information():
    q = jnp.ones((1, 4), bool)
    correct = draft_attention_mask(jnp.asarray([[0, 0, 1, 1]]), q, q)
    split = draft_attention_mask(jnp.asarray([[0, 1, 2, 3]]), q, q)
    assert correct[0, 0, 1] and not split[0, 0, 1]
    assert correct[0, 1, 0] and split[0, 1, 0]
    assert jnp.all(split[0][jnp.tril_indices(4)])


def test_empty_stratum_is_not_a_zero_quality_score(monkeypatch):
    r = runner(monkeypatch)
    x = {"count": np.ones((1, 2)), "correct": np.ones((1, 2)), "nll": np.ones((1, 2))}
    m = r.group(x, np.zeros((1, 2), bool))
    assert m == {"sites": 0, "bytes": 0, "accuracy": None, "nll": None}


def test_counterfactual_pairs_hold_background_positions_lengths_fixed(monkeypatch):
    r = runner(monkeypatch)
    data, q, b, wrong = r.fixtures()
    assert jnp.array_equal(data[0]["target_mask"], data[1]["target_mask"])
    for i, query in enumerate(q.tolist()):
        assert b[i, query] == 0 and wrong[i, query] == 1
        assert b[i, query - 1] == b[i, query + 1] == 1
        prefix = query * 4
        assert jnp.array_equal(data[0]["target"][i, :prefix], data[1]["target"][i, :prefix])
        for d in data:
            assert jnp.array_equal(
                d["target"][i, prefix : prefix + 4], d["target"][i, prefix + 4 : prefix + 8]
            )
    assert jnp.all((data[0]["target"] >= 0) & (data[0]["target"] <= 32))


def test_decoder_reset_preserves_first_carrier_but_blocks_history(monkeypatch):
    r = runner(monkeypatch)
    cfg, params, _ = _fixture()
    monkeypatch.setattr(r, "CFG", cfg)
    p = torx_means_to_deterministic(params)["decoder"]
    h = jax.random.normal(jax.random.key(1110), (2, cfg.carrier.C, cfg.carrier.h_dim))
    ell = jnp.full((2, cfg.carrier.C), cfg.carrier.L_max)
    full, local = r.decode(h, ell, p), r.decode_local(h, ell, p)
    assert jnp.allclose(full[:, 0], local[:, 0], atol=1e-6, rtol=1e-6)
    changed = h.at[:, 0].add(0.2)
    full_delta = r.decode(changed, ell, p) - full
    local_delta = r.decode_local(changed, ell, p) - local
    assert jnp.max(jnp.abs(full_delta[:, 1:])) > 0
    assert jnp.array_equal(local_delta[:, 1:], jnp.zeros_like(local_delta[:, 1:]))


def test_geometry_ablation_preserves_carrier_position_embedding(monkeypatch):
    runner(monkeypatch)
    module = importlib.import_module("analyze_phase_g1_geometry")
    ops = module.GeometryOps("zero_geometry_noattention")
    p = jnp.arange(12, dtype=jnp.float32).reshape(3, 4)
    indices = jnp.asarray([[0, 1]])
    assert jnp.array_equal(ops.embedding(indices, p, name="dit.carrier_embed"), p[indices])
    assert not jnp.any(ops.embedding(indices, p, name="dit.block_embed"))
    assert not jnp.any(ops.embedding(indices, p, name="dit.within_embed"))
    assert ops.variant == "no_attention"
