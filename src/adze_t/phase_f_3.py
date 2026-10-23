"""Frozen decision and statistics policy for Phase F.3 S=2 rollout training."""

from __future__ import annotations

import math
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np


PHASE_F_3_S2_TRAINING_BENEFIT = "PHASE_F_3_S2_TRAINING_BENEFIT"
PHASE_F_3_S2_TRAINING_NEUTRAL = "PHASE_F_3_S2_TRAINING_NEUTRAL"
PHASE_F_3_S2_TRAINING_NEGATIVE = "PHASE_F_3_S2_TRAINING_NEGATIVE"
PHASE_F_3_S2_TRAINING_UNRESOLVED = "PHASE_F_3_S2_TRAINING_UNRESOLVED"
PHASE_F_3_BLOCKED = "PHASE_F_3_BLOCKED"
F3_LABELS = (
    PHASE_F_3_S2_TRAINING_BENEFIT,
    PHASE_F_3_S2_TRAINING_NEUTRAL,
    PHASE_F_3_S2_TRAINING_NEGATIVE,
    PHASE_F_3_S2_TRAINING_UNRESOLVED,
    PHASE_F_3_BLOCKED,
)

INFORMATIVE_LEVELS = (0.25, 0.50, 0.75)
BOOTSTRAP_RESAMPLES = 2_000
ROOT_T_95_DF15 = 2.131449545559323


def summary(values: np.ndarray) -> dict[str, float | int]:
    values = np.asarray(values, dtype=np.float64)
    count = len(values)
    mean = float(np.mean(values))
    sd = float(np.std(values, ddof=1)) if count > 1 else 0.0
    se = sd / math.sqrt(count)
    return {
        "count": count,
        "mean": mean,
        "sd": sd,
        "se": se,
        "normal_95_ci_low": mean - 1.96 * se,
        "normal_95_ci_high": mean + 1.96 * se,
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


def paired_summary(
    before: np.ndarray,
    after: np.ndarray,
    *,
    bootstrap_seed: int,
) -> dict[str, Any]:
    before = np.asarray(before, dtype=np.float64)
    after = np.asarray(after, dtype=np.float64)
    change = after - before
    stats = summary(change)
    indices = jax.random.randint(
        jax.random.PRNGKey(bootstrap_seed),
        (BOOTSTRAP_RESAMPLES, len(change)),
        0,
        len(change),
    )
    bootstrap = jnp.mean(jnp.asarray(change)[indices], axis=1)
    low, high = map(float, np.asarray(jnp.percentile(bootstrap, jnp.asarray([2.5, 97.5]))))
    before_mean = float(np.mean(before))
    return {
        "definition": "after minus before",
        "before_mean": before_mean,
        "after_mean": float(np.mean(after)),
        "absolute_change": stats["mean"],
        "relative_change": stats["mean"] / max(abs(before_mean), 1.0e-12),
        "paired_change": stats,
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        "bootstrap_percentile_95_ci_low": low,
        "bootstrap_percentile_95_ci_high": high,
    }


def bucket_effect(row: dict[str, Any], *, require_uncertainty: bool) -> str:
    accuracy_change = float(row["s2_accuracy"] - row["s1_accuracy"])
    nll_ratio = float(row["s2_nll"] / max(row["s1_nll"], 1.0e-12))
    benefit = accuracy_change >= 0.01 and nll_ratio <= 0.98
    degradation = accuracy_change <= -0.01 and nll_ratio >= 1.02
    if require_uncertainty:
        paired = row["paired"]
        benefit = (
            benefit
            and paired["byte_accuracy"]["bootstrap_percentile_95_ci_low"] > 0
            and paired["nll"]["bootstrap_percentile_95_ci_high"] < 0
        )
        degradation = (
            degradation
            and paired["byte_accuracy"]["bootstrap_percentile_95_ci_high"] < 0
            and paired["nll"]["bootstrap_percentile_95_ci_low"] > 0
        )
    if benefit:
        return "benefit"
    if degradation:
        return "degradation"
    return "neutral"


def step0_forgetting(row: dict[str, Any]) -> bool:
    accuracy_drop = float(row["trained_s1_accuracy"] - row["f1_s1_accuracy"])
    nll_ratio = float(row["trained_s1_nll"] / max(row["f1_s1_nll"], 1.0e-12))
    return accuracy_drop < -0.02 or nll_ratio > 1.05


def should_extend_to_10k(rows_2k: list[dict[str, Any]], rows_5k: list[dict[str, Any]]) -> bool:
    two = [row for row in rows_2k if row["nu"] in INFORMATIVE_LEVELS]
    five = [row for row in rows_5k if row["nu"] in INFORMATIVE_LEVELS]
    if len(two) != 3 or len(five) != 3:
        raise ValueError("extension policy requires all informative nu buckets")
    accuracy_2k = float(np.mean([row["s2_accuracy"] for row in two]))
    accuracy_5k = float(np.mean([row["s2_accuracy"] for row in five]))
    nll_2k = float(np.mean([row["s2_nll"] for row in two]))
    nll_5k = float(np.mean([row["s2_nll"] for row in five]))
    material = accuracy_5k - accuracy_2k >= 0.01 or nll_5k <= 0.98 * nll_2k
    no_forgetting = not any(row.get("step0_forgetting", False) for row in five)
    healthy = not any(row.get("nonfinite", False) for row in five)
    return material and no_forgetting and healthy


def is_pareto_best_5k(
    earlier_checkpoints: list[list[dict[str, Any]]], rows_5k: list[dict[str, Any]]
) -> bool:
    """Require 5k to lie on the informative S2 accuracy/NLL validation frontier."""

    def aggregate(rows):
        informative = [row for row in rows if row["nu"] in INFORMATIVE_LEVELS]
        if len(informative) != 3:
            raise ValueError("Pareto policy requires all informative nu buckets")
        return (
            float(np.mean([row["s2_accuracy"] for row in informative])),
            float(np.mean([row["s2_nll"] for row in informative])),
        )

    accuracy_5k, nll_5k = aggregate(rows_5k)
    earlier = [aggregate(rows) for rows in earlier_checkpoints]
    dominated = any(
        accuracy >= accuracy_5k and nll <= nll_5k and (accuracy > accuracy_5k or nll < nll_5k)
        for accuracy, nll in earlier
    )
    reaches_frontier = accuracy_5k >= max(accuracy for accuracy, _ in earlier) or nll_5k <= min(
        nll for _, nll in earlier
    )
    return not dominated and reaches_frontier


def classify(
    informative_rows: list[dict[str, Any]],
    *,
    healthy: bool,
    converged: bool,
    stochastic_agrees: bool,
    blocked: bool = False,
) -> str:
    if blocked:
        return PHASE_F_3_BLOCKED
    if not healthy:
        return PHASE_F_3_S2_TRAINING_UNRESOLVED
    if len(informative_rows) != 3:
        raise ValueError("F3 classification requires exactly three informative rows")
    effects = [bucket_effect(row, require_uncertainty=True) for row in informative_rows]
    forgetting = sum(bool(row["step0_forgetting"]) for row in informative_rows)
    benefit_count = effects.count("benefit")
    degradation_count = effects.count("degradation")
    remaining_nonnegative = all(
        row["s2_accuracy"] >= row["s1_accuracy"] and row["s2_nll"] <= row["s1_nll"]
        for row, effect in zip(informative_rows, effects, strict=True)
        if effect != "benefit"
    )
    if (
        benefit_count >= 2
        and degradation_count == 0
        and remaining_nonnegative
        and forgetting < 2
        and stochastic_agrees
    ):
        return PHASE_F_3_S2_TRAINING_BENEFIT
    if not converged or forgetting >= 2 or (benefit_count and degradation_count):
        return PHASE_F_3_S2_TRAINING_UNRESOLVED
    if degradation_count >= 2 and benefit_count == 0:
        return PHASE_F_3_S2_TRAINING_NEGATIVE
    if benefit_count < 2 and degradation_count < 2 and forgetting == 0:
        return PHASE_F_3_S2_TRAINING_NEUTRAL
    return PHASE_F_3_S2_TRAINING_UNRESOLVED
