"""Backend protocol.

Do not fork the architecture into deterministic and Torx model implementations.
Both backends must satisfy one shared operator interface.
"""

from typing import Protocol, Any
from jax import Array


class LearnedOps(Protocol):
    def linear(self, x: Array, params: Any) -> Array: ...

    def categorical_logits(self, x: Array, params: Any) -> Array: ...

    def ssm_transition(self, state: Array, x: Array, params: Any) -> Array: ...
