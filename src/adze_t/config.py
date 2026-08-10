"""Configuration types for the faithful Adze implementation."""

from dataclasses import dataclass


@dataclass(frozen=True)
class CarrierConfig:
    C: int
    h_dim: int
    L_max: int


@dataclass(frozen=True)
class PackingConfig:
    M_max: int
    K: int


@dataclass(frozen=True)
class ReferenceConfig:
    carrier: CarrierConfig = CarrierConfig(C=32, h_dim=64, L_max=4)
    packing: PackingConfig = PackingConfig(M_max=32, K=8)


REFERENCE_SMALL_V0 = ReferenceConfig()
