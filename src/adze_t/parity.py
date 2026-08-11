"""Ordered diagnostics for deterministic/Torx parity gates."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import jax
import jax.numpy as jnp


@dataclass(frozen=True)
class ParityMetric:
    path: str
    deterministic_shape: tuple[int, ...]
    torx_shape: tuple[int, ...]
    max_absolute_difference: float
    max_relative_difference: float
    rms_difference: float
    atol: float
    rtol: float
    passed: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def array_metric(
    path: str, deterministic: jax.Array, torx_value: jax.Array, *, atol: float, rtol: float
) -> ParityMetric:
    left = jnp.asarray(deterministic)
    right = jnp.asarray(torx_value)
    difference = jnp.abs(left.astype(jnp.float32) - right.astype(jnp.float32))
    denominator = jnp.maximum(jnp.abs(left.astype(jnp.float32)), jnp.asarray(1.0e-12))
    return ParityMetric(
        path=path,
        deterministic_shape=left.shape,
        torx_shape=right.shape,
        max_absolute_difference=float(jnp.max(difference, initial=0.0)),
        max_relative_difference=float(jnp.max(difference / denominator, initial=0.0)),
        rms_difference=float(jnp.sqrt(jnp.mean(difference**2))) if difference.size else 0.0,
        atol=atol,
        rtol=rtol,
        passed=bool(jnp.allclose(left, right, atol=atol, rtol=rtol)),
    )


def ordered_model_trace(outputs: dict[str, Any]) -> list[tuple[str, jax.Array]]:
    proposal_h, proposal_b, proposal_l = outputs["proposal"]
    h_hat, b_logits, l_logits = outputs["prediction"]
    trace: list[tuple[str, jax.Array]] = [
        ("frontend/prompt", outputs["prompt_frontend"]),
        ("frontend/target", outputs["target_frontend"]),
        ("context/sequence", outputs["context_seq"]),
        ("context/global", outputs["context_global"]),
        ("target/h0", outputs["target"]["h0"]),
        ("target/b_logits", outputs["target"]["b_logits"]),
        ("target/l_logits", outputs["target"]["l_logits"]),
        ("proposal/h", proposal_h),
        ("proposal/b_logits", proposal_b),
        ("proposal/l_logits", proposal_l),
        ("packed/input", outputs["packed_carrier"]),
    ]
    block_states = outputs["dit_aux"]["block_trajectory"]
    for index in range(block_states.shape[0]):
        trace.append((f"dit/q{index // 4}/b{index % 4}", block_states[index]))
    trace.extend(
        [
            ("packed/output", outputs["packed_output"]),
            ("carrier/unpool", outputs["unpooled_carrier"]),
            ("carrier/pre_head", outputs["pre_head_carrier"]),
            ("heads/h_hat", h_hat),
            ("heads/b_logits", b_logits),
            ("heads/l_logits", l_logits),
            ("decoder/logits", outputs["byte_logits"]),
        ]
    )
    return trace


def compare_ordered_model_traces(
    deterministic: dict[str, Any],
    torx_outputs: dict[str, Any],
    *,
    atol: float = 1.0e-5,
    rtol: float = 1.0e-5,
) -> tuple[list[ParityMetric], ParityMetric | None]:
    left = ordered_model_trace(deterministic)
    right = ordered_model_trace(torx_outputs)
    if [path for path, _ in left] != [path for path, _ in right]:
        raise ValueError("deterministic and Torx trace paths differ")
    metrics = [
        array_metric(path, a, b, atol=atol, rtol=rtol)
        for (path, a), (_, b) in zip(left, right, strict=True)
    ]
    return metrics, next((metric for metric in metrics if not metric.passed), None)
