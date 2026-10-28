"""Conformance: V1 changes statistics/provenance, never scientific semantics."""

import ast
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def definitions(name):
    tree = ast.parse((ROOT / "scripts" / name).read_text())
    return {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}


class Normalize(ast.NodeTransformer):
    def visit_ImportFrom(self, node):
        if node.module in ("preflight_phase_g23_v1", "science_phase_g23_v1"):
            node.module = node.module.removesuffix("_v1")
        return node

    def visit_Constant(self, node):
        if node.value == "results/phase_g/G_FINAL_RESULTS_V1.md":
            node.value = "results/phase_g/G_FINAL_RESULTS.md"
        return node

    def visit_Dict(self, node):
        pairs = [
            (k, v)
            for k, v in zip(node.keys, node.values, strict=True)
            if not isinstance(k, ast.Constant) or k.value != "adjudication"
        ]
        node.keys, node.values = [p[0] for p in pairs], [p[1] for p in pairs]
        return self.generic_visit(node)


def normalized(node):
    return ast.dump(Normalize().visit(node), include_attributes=False)


def test_every_training_evaluation_selection_and_decision_function_unchanged():
    old = definitions("run_phase_g23.py")
    new = definitions("run_phase_g23_v1.py")
    unchanged = set(old) - {"freeze", "audit"}
    assert len(unchanged) >= 20
    for name in unchanged:
        assert normalized(old[name]) == normalized(new[name]), name
    old = definitions("science_phase_g23.py")
    new = definitions("science_phase_g23_v1.py")
    for name in old:
        assert normalized(old[name]) == normalized(new[name]), name


def test_full_model_preflight_unchanged_except_adjudicated_oracle_gate():
    old = definitions("preflight_phase_g23.py")
    new = definitions("preflight_phase_g23_v1.py")
    for name in old.keys() - {"run"}:
        assert normalized(old[name]) == normalized(new[name]), name
    # Remove only the old query-oracle loop and its accumulator, or the new
    # verified-adjudication assignment. All other full-model gates must match.
    for f in (old["run"], new["run"]):
        f.body = [
            n
            for n in f.body
            if not (
                isinstance(n, ast.Assign)
                and any(
                    isinstance(t, ast.Name) and t.id in ("oracle", "adjudication")
                    for t in n.targets
                )
            )
            and not (
                isinstance(n, ast.For)
                and isinstance(n.target, ast.Name)
                and n.target.id == "name"
                and isinstance(n.iter, ast.Tuple)
                and [x.value for x in n.iter.elts if isinstance(x, ast.Constant)]
                == ["uniform", "asymmetric", "peaked"]
            )
        ]
    assert normalized(old["run"]) == normalized(new["run"])


def test_v1_requires_actual_immutable_adjudication_and_rejects_changed_manifest(monkeypatch):
    import run_phase_g23_v1 as g

    proof = g.verified_adjudication()
    assert proof["identity"] == "9f0ae45d25a0d09c8ae9927f5d209c4bf9a805634f947086347bbffb9e714375"
    original = g.sha
    monkeypatch.setattr(
        g, "sha", lambda p: "0" * 64 if str(p).endswith("g2_1/MANIFEST.json") else original(p)
    )
    with pytest.raises(RuntimeError, match="manifest mismatch"):
        g.verified_adjudication()
