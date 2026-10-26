"""Independent grammar, paired information-set and actual-model routing controls."""

from dataclasses import replace
import importlib
from pathlib import Path

import jax
import jax.numpy as jnp
import pytest

from adze_t.phase_g11_task import (
    make_examples,
    symbolic_structure,
    symbolic_target,
    relabel,
    observation_control,
    cut_after_query,
)
from adze_t.phase_g0 import dynamic_capacity_routing
from adze_t.state import CommittedStructure
from adze_t.masking import draft_attention_mask
from adze_t.packing import pack_values
from adze_t.unpool import unpool_values
from adze_t.backends.mapping import torx_means_to_deterministic
from test_phase_g_integration import _fixture


@pytest.fixture(scope="module")
def data():
    return make_examples(1150, 1)


def test_grammar_is_identifiable_without_teacher_fields(data):
    b, ell = symbolic_structure(data["source"])
    assert jnp.array_equal(b, data["boundary"])
    assert jnp.array_equal(ell, data["length"])
    assert jnp.array_equal(symbolic_target(data["source"], data["q"]), data["target"])
    assert jnp.array_equal(jnp.sort(ell.sum(1)), jnp.arange(8, 65))
    slots = data["source"].reshape(57, 16, 4)
    row = jnp.arange(57)
    assert jnp.all(slots[row, data["q"], 0] == 3)
    assert jnp.all(slots[row, data["q"], 1:] == 4)
    assert jnp.all(slots[row, data["peer"], 0] == 1)
    assert jnp.all((data["peer"] > data["q"]) & (data["peer"] - data["q"] <= 5))
    assert jnp.all(data["prompt"][:, 1] == data["q"] + 1)


def test_counterlabels_do_not_change_prefix_or_metadata(data):
    label = jax.random.randint(jax.random.key(1151), (57, 3), 5, 33)
    other = relabel(data, label)
    for key in ("boundary", "length", "prompt", "source_mask", "target_mask", "q", "peer"):
        assert jnp.array_equal(data[key], other[key])
    positions = jnp.arange(64)[None] < 4 * data["peer"][:, None] + 1
    assert jnp.all(~positions | (data["source"] == other["source"]))
    nuisance = jax.random.randint(jax.random.key(1152), (57, 16, 3), 5, 33)
    for case in ("C", "D_substitute"):
        assert jnp.array_equal(
            observation_control(data, case, nuisance), observation_control(other, case, nuisance)
        )
    d = observation_control(data, "D_outside", nuisance).reshape(57, 16, 4)
    a = data["source"].reshape(57, 16, 4)
    assert jnp.array_equal(d[:, :, 0], a[:, :, 0])
    assert jnp.array_equal(d[jnp.arange(57), data["peer"]], a[jnp.arange(57), data["peer"]])


def test_true_and_cut_routes_preserve_identity_and_remove_only_future_access(data):
    def route(b):
        return dynamic_capacity_routing(
            CommittedStructure(b, data["length"] > 0, data["length"]), M_max=32, K=8, max_extent=4
        )

    for b, allowed in ((data["boundary"], True), (cut_after_query(data), False)):
        eager, compiled = route(b), jax.jit(route)(b)
        assert jax.tree_util.tree_all(jax.tree.map(jnp.array_equal, eager, compiled))
        assert eager.routing.executed
        m = eager.routing.metadata
        pos = m.carrier_to_m * 8 + m.carrier_to_k
        mask = draft_attention_mask(m.block_id, m.query_mask, m.kv_mask)
        assert jnp.all(
            mask[jnp.arange(57), pos[jnp.arange(57), data["q"]], pos[jnp.arange(57), data["peer"]]]
            == allowed
        )
        x = jnp.broadcast_to(jnp.arange(32)[None, :, None], (57, 32, 1))
        assert jnp.array_equal(unpool_values(pack_values(x, m), m, C=32), x)
        assert jnp.array_equal(eager.semantic.c_b, b)


def test_real_model_transmits_future_signal_only_when_permitted(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    mod = importlib.import_module("run_phase_g11_validation")
    cfg, p, _ = _fixture()
    p = torx_means_to_deterministic(p)
    h = jax.random.normal(jax.random.key(1153), (1, cfg.carrier.C, cfg.carrier.h_dim))
    changed = h.at[:, 3].add(0.2)
    ell = jnp.full((1, cfg.carrier.C), cfg.carrier.L_max, jnp.int32)
    prompt = jnp.asarray([[0xF1, 2]], jnp.int32)
    run = jax.jit(mod.forward, static_argnames=("no_attention", "torx", "cfg"))
    correct = jnp.asarray([[1, 0, 0, 1]], jnp.int32)
    cut = correct.at[:, 1].set(1)
    for b, no_attention, transmits in (
        (correct, False, True),
        (cut, False, False),
        (correct, True, False),
    ):
        a = run(p, h, prompt, b, ell, cfg=cfg, no_attention=no_attention)
        c = run(p, changed, prompt, b, ell, cfg=cfg, no_attention=no_attention)
        assert a["executed"]
        difference = jnp.max(jnp.abs(a["logits"][:, 1] - c["logits"][:, 1]))
        assert bool(difference > 1e-6) == transmits


def test_invalid_capacity_skips_actual_model_eager_and_jit(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    mod = importlib.import_module("run_phase_g11_validation")
    cfg, p, _ = _fixture()
    p = torx_means_to_deterministic(p)
    cfg = replace(cfg, packing=replace(cfg.packing, K=1, M_max=1))
    h = jnp.ones((1, cfg.carrier.C, cfg.carrier.h_dim))
    b = jnp.ones((1, cfg.carrier.C), jnp.int32)
    ell = jnp.full((1, cfg.carrier.C), cfg.carrier.L_max, jnp.int32)
    for apply in (
        mod.forward,
        jax.jit(mod.forward, static_argnames=("no_attention", "torx", "cfg")),
    ):
        out = apply(p, h, jnp.asarray([[0xF1, 1]]), b, ell, cfg=cfg)
        assert not out["executed"] and not jnp.any(out["logits"])
