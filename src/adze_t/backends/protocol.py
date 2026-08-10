"""Backend protocol.

Do not fork the architecture into deterministic and Torx model implementations.
Both backends must satisfy one shared operator interface.
"""

from typing import Protocol, Any


class LearnedOps(Protocol):
    def linear(self, *args: Any, **kwargs: Any) -> Any: ...

    def categorical(self, *args: Any, **kwargs: Any) -> Any: ...

    def ssm_transition(self, *args: Any, **kwargs: Any) -> Any: ...
