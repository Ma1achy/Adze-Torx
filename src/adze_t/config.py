"""Configuration objects shared by milestone implementations.

These configs encode shape/control invariants only. They intentionally do not
choose unresolved Torx factor semantics.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class CarrierConfig:
    capacity: int = 64
    latent_dim: int = 128
    max_bytes_per_site: int = 8
    length_categories: int = 9  # 0..max_bytes_per_site

    def __post_init__(self) -> None:
        if self.capacity <= 0:
            raise ValueError("capacity must be positive")
        if self.latent_dim <= 0:
            raise ValueError("latent_dim must be positive")
        if self.max_bytes_per_site <= 0:
            raise ValueError("max_bytes_per_site must be positive")
        if self.length_categories != self.max_bytes_per_site + 1:
            raise ValueError("length_categories must include zero/non-emitting")


@dataclass(frozen=True)
class LoopConfig:
    physical_blocks: int = 4
    core_cycles: int = 3  # Q
    denoise_steps: int = 4  # S
    refinement_steps: int = 1  # R

    def __post_init__(self) -> None:
        if min(
            self.physical_blocks,
            self.core_cycles,
            self.denoise_steps,
            self.refinement_steps,
        ) < 1:
            raise ValueError("loop counts must be >= 1")

    @property
    def effective_core_applications(self) -> int:
        return self.physical_blocks * self.core_cycles
