"""Phase-D Monte Carlo root and interval helpers."""

import jax
import jax.numpy as jnp

from adze_t.evaluation import phase_d_root, phase_d_stage_names, student_t_summary


def test_mc_roots_are_nested_reproducible_and_distinct():
    first_16 = [phase_d_root(4100, index) for index in range(16)]
    first_32 = [phase_d_root(4100, index) for index in range(32)]
    assert all(
        jnp.array_equal(left, right) for left, right in zip(first_16, first_32[:16], strict=True)
    )
    assert len({tuple(map(int, jax.random.key_data(key))) for key in first_32}) == 32


def test_student_t_summary_and_stage_manifest_are_frozen():
    singleton = student_t_summary([0.75])
    assert singleton == {"count": 1, "mean": 0.75, "sample_sd": 0.0, "ci95": [0.75, 0.75]}
    summary = student_t_summary([float(value) for value in range(16)])
    assert summary["count"] == 16
    assert summary["sample_sd"] > 0
    assert summary["ci95"][0] < summary["mean"] < summary["ci95"][1]
    names = phase_d_stage_names()
    assert names[:3] == ("frontend", "proposal", "pack")
    assert names[-4:] == ("unpool", "h_hat", "carrier", "decoder_logits")
    assert len([name for name in names if name.startswith("dit.q") and ".block" in name]) == 12
