"""Backend protocol.

Do not fork the architecture into deterministic and Torx model implementations.
Both backends must satisfy one shared operator interface.
"""

from typing import Any, Protocol

from jax import Array


class LearnedOps(Protocol):
    def init_linear(self, key: Array, in_dim: int, out_dim: int, *, scale: float = 1.0) -> Any: ...

    def init_embedding(self, key: Array, size: int, width: int) -> Any: ...

    def init_depthwise_conv(self, key: Array, kernel_size: int, channels: int) -> Any: ...

    def linear(self, x: Array, params: Any, *, name: str) -> Array: ...

    def embedding(self, indices: Array, params: Any, *, name: str) -> Array: ...

    def depthwise_conv1d(self, x: Array, params: Any, *, name: str) -> Array: ...

    def categorical_logits(self, x: Array, params: Any, *, name: str) -> Array: ...

    def parameter(self, value: Array, *, name: str) -> Array: ...
