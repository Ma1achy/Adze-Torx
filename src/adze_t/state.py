"""Persistent carrier and structural-state types.

Phase A owns this module.

Architecture reference:
- docs/architecture/adze-architecture-v3.md sections 8, 9, 43, 53.
"""

from dataclasses import dataclass
import jax
from jax import Array
import jax.numpy as jnp


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class ObservedStructure:
    """Corrupted/observed structure used as model input."""

    s_b: Array
    s_l: Array

    def tree_flatten(self):
        return (self.s_b, self.s_l), None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        del aux_data
        return cls(*children)


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class PredictedStructure:
    """Current model predictions; must not immediately rewrite routing."""

    b_logits: Array
    l_logits: Array

    def tree_flatten(self):
        return (self.b_logits, self.l_logits), None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        del aux_data
        return cls(*children)


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class CommittedStructure:
    """Structure that controls block construction and participation."""

    c_b: Array
    activity: Array
    length: Array | None = None

    def effective_length(self) -> Array:
        """Return explicit extent, or the Phase A activity-compatible fallback."""
        if self.length is None:
            return self.activity.astype(jnp.int32)
        return self.length

    def tree_flatten(self):
        return (self.c_b, self.activity, self.length), None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        del aux_data
        return cls(*children)


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class CarrierState:
    """Persistent fixed-capacity carrier."""

    h: Array
    observed: ObservedStructure
    committed: CommittedStructure
    predicted: PredictedStructure | None = None

    def tree_flatten(self):
        return (self.h, self.observed, self.committed, self.predicted), None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        del aux_data
        return cls(*children)


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
    if committed.length is not None and committed.length.shape != (batch, capacity):
        raise ValueError("committed.length must match [batch, carrier]")

    valid_binary, terminal = carrier_invariant_flags(committed.c_b, committed.activity)
    if not bool(valid_binary):
        raise ValueError("committed boundary/activity must contain only binary values")
    if not bool(terminal):
        raise ValueError("committed.c_b must have a terminal cut after the final carrier")
    if committed.length is not None:
        if not bool(jnp.all(committed.length >= 0)):
            raise ValueError("committed.length must be non-negative")
        if not bool(jnp.all((committed.length > 0) == (committed.activity > 0))):
            raise ValueError("activity must agree with committed.length > 0")


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
