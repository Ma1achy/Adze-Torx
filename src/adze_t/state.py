"""Persistent carrier and structural-state types.

Phase A owns this module.

Architecture reference:
- docs/architecture/adze-architecture-v3.md sections 8, 9, 43, 53.
"""

from dataclasses import dataclass
from jax import Array
import jax.numpy as jnp


@dataclass(frozen=True)
class ObservedStructure:
    """Corrupted/observed structure used as model input."""

    s_b: Array
    s_l: Array


@dataclass(frozen=True)
class PredictedStructure:
    """Current model predictions; must not immediately rewrite routing."""

    b_logits: Array
    l_logits: Array


@dataclass(frozen=True)
class CommittedStructure:
    """Structure that controls block construction and participation."""

    c_b: Array
    activity: Array


@dataclass(frozen=True)
class CarrierState:
    """Persistent fixed-capacity carrier."""

    h: Array
    observed: ObservedStructure
    committed: CommittedStructure
    predicted: PredictedStructure | None = None


def validate_carrier_state(state: CarrierState) -> None:
    """Validate shape, value, and terminal-sentinel invariants."""
    if state.h.ndim != 3:
        raise ValueError("h must have shape [batch, carrier, h_dim]")
    batch, capacity = state.h.shape[:2]

    observed = state.observed
    committed = state.committed
    for name, value in (
        ("observed.s_b", observed.s_b),
        ("observed.s_l", observed.s_l),
        ("committed.c_b", committed.c_b),
        ("committed.activity", committed.activity),
    ):
        if value.shape != (batch, capacity):
            raise ValueError(f"{name} must have shape {(batch, capacity)}, got {value.shape}")

    if state.predicted is not None:
        for name, value in (
            ("predicted.b_logits", state.predicted.b_logits),
            ("predicted.l_logits", state.predicted.l_logits),
        ):
            if value.shape[:2] != (batch, capacity):
                raise ValueError(
                    f"{name} must start with shape {(batch, capacity)}, got {value.shape}"
                )

    valid_binary, terminal = carrier_invariant_flags(committed.c_b, committed.activity)
    if not bool(valid_binary):
        raise ValueError("committed boundary/activity must contain only binary values")
    if not bool(terminal):
        raise ValueError("committed.c_b must have a terminal cut after the final carrier")


def carrier_invariant_flags(c_b: Array, activity: Array) -> tuple[Array, Array]:
    """Pure-JAX committed-state checks for compiled callers.

    Returns scalar ``(binary_values_valid, terminal_boundary_valid)`` flags;
    the eager ``validate_carrier_state`` wrapper turns failures into errors.
    """
    c_b = jnp.asarray(c_b)
    activity = jnp.asarray(activity)
    binary = jnp.all((c_b == 0) | (c_b == 1)) & jnp.all((activity == 0) | (activity == 1))
    terminal = jnp.all(c_b[:, -1] == 1)
    return binary, terminal
