"""JAX-friendly state containers.

The fields are deliberately generic at scaffold time. Milestones M2-M5 will
freeze exact dtypes, shapes, and Torx state encodings.
"""

from typing import Any, NamedTuple


class CarrierState(NamedTuple):
    """Fixed-capacity persistent carrier."""

    h: Any
    b: Any
    length: Any


class RoutingState(NamedTuple):
    """Observed, predicted, and committed structural state."""

    observed_b: Any
    observed_length: Any
    predicted_b: Any
    predicted_length: Any
    committed_b: Any
    activity: Any


class ModelState(NamedTuple):
    carrier: CarrierState
    routing: RoutingState
