"""M4.5 metrics, paired comparisons, and trajectory diagnostics."""

from __future__ import annotations

from math import sqrt

import jax.numpy as jnp

T_CRITICAL_95 = {3: 4.303, 5: 2.776}


def paired_error_interval(q1_accuracy: list[float], q_accuracy: list[float]) -> dict[str, float]:
    q1_error = 1.0 - jnp.asarray(q1_accuracy)
    q_error = 1.0 - jnp.asarray(q_accuracy)
    differences = q1_error - q_error
    n = len(differences)
    mean = float(jnp.mean(differences))
    se = float(jnp.std(differences, ddof=1) / sqrt(n)) if n > 1 else float("inf")
    critical = T_CRITICAL_95.get(n, 2.0)
    baseline = float(jnp.mean(q1_error))
    relative = mean / baseline if baseline > 1e-6 else float("nan")
    return {
        "mean_error_difference": mean,
        "ci_low": mean - critical * se,
        "ci_high": mean + critical * se,
        "relative_error_reduction": relative,
    }


def best_loop(metric_rows: list[dict[str, float]], name: str) -> dict[str, float]:
    best = min(metric_rows, key=lambda row: row[name])
    return {
        "best_loop": best["loop"],
        "best_metric": best[name],
        "final_metric": metric_rows[-1][name],
    }
