"""Static protocol-conformance checks; not numerical/full-model proof."""

import ast
from pathlib import Path


SOURCE = Path(__file__).resolve().parents[1] / "scripts/complete_phase_g23_validation.py"


def test_confirmation_is_fixed_sixteen_root_validation_not_test_or_training():
    tree = ast.parse(SOURCE.read_text())
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)]
    evaluations = [
        n for n in calls if isinstance(n.func, ast.Attribute) and n.func.attr == "evaluate"
    ]
    assert len(evaluations) == 1
    keywords = {x.arg: x.value for x in evaluations[0].keywords}
    assert ast.literal_eval(keywords["split"]) == "validation"
    assert ast.literal_eval(keywords["mode"]) == "sample"
    comprehensions = [n for n in ast.walk(tree) if isinstance(n, ast.ListComp)]
    root_loop = next(n for n in comprehensions if n.elt is evaluations[0]).generators[0].iter
    assert isinstance(root_loop, ast.Call) and ast.unparse(root_loop.func) == "range"
    assert [ast.literal_eval(x) for x in root_loop.args] == [16]
    controls = next(
        n.iter
        for n in ast.walk(tree)
        if isinstance(n, ast.For) and isinstance(n.target, ast.Name) and n.target.id == "control"
    )
    assert ast.literal_eval(controls) == ("policy", "cut", "neutral")
    assert not any(
        isinstance(n.func, ast.Attribute)
        and n.func.attr in ("train_step", "train_arm", "adamw_update", "test_data", "select")
        for n in calls
    )


def test_confirmation_records_selection_and_before_after_optimizer_identity():
    text = SOURCE.read_text()
    tree = ast.parse(text)
    assert "LOCK_EX | fcntl.LOCK_NB" in text
    assert 'selection["cap"] != 1000' in text
    assert 'state["completed_updates"] != 1000' in text
    assert 'chosen["optimizer_tree"]' in text
    checks = [n for n in ast.walk(tree) if isinstance(n, ast.If)]
    assert any(
        "g.tree_hash((state['params'], state['moments'])) != before" in ast.unparse(n.test)
        for n in checks
    )
    assert '"selection_sha256": g.sha(path)' in text
    assert '"optimizer_updates": 0' in text
