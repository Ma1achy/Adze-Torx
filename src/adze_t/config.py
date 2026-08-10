"""Configuration types for the faithful Adze implementation."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


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
class ModelConfig:
    byte_vocab: int = 256
    prompt_max_bytes: int = 128
    target_max_bytes: int = 128
    d_front: int = 64
    d_ctx: int = 128
    frontend_layers: int = 2
    target_layers: int = 2
    proposal_layers: int = 2
    proposal_hidden_dim: int = 64
    d_model: int = 128
    heads: int = 4
    head_dim: int = 32
    ffn_hidden: int = 256
    physical_blocks_L: int = 4
    cycles_Q: int = 3
    d_dec: int = 128
    decoder_layers: int = 2


@dataclass(frozen=True)
class TrainingConfig:
    learning_rate: float = 3.0e-4
    weight_decay: float = 0.01
    grad_clip_norm: float = 1.0
    batch_size: int = 32
    h_weight: float = 1.0
    boundary_weight: float = 1.0
    extent_weight: float = 1.0
    byte_weight: float = 1.0
    proposal_weight: float = 0.25


@dataclass(frozen=True)
class ReferenceConfig:
    carrier: CarrierConfig = CarrierConfig(C=32, h_dim=64, L_max=4)
    packing: PackingConfig = PackingConfig(M_max=32, K=8)
    model: ModelConfig = ModelConfig()
    training: TrainingConfig = TrainingConfig()


REFERENCE_SMALL_V0 = ReferenceConfig()


def load_reference_config(path: str | Path) -> ReferenceConfig:
    """Load the frozen YAML reference configuration into typed defaults."""
    raw: dict[str, Any] = yaml.safe_load(Path(path).read_text())
    io = raw["io"]
    carrier = raw["carrier"]
    packing = raw["packing"]
    frontend = raw["frontend"]
    context = raw["context_encoder"]
    target = raw["target_encoder"]
    proposal = raw["proposal"]
    dit = raw["dit"]
    decoder = raw["decoder"]
    training = raw["training"]
    return ReferenceConfig(
        carrier=CarrierConfig(**carrier),
        packing=PackingConfig(M_max=packing["M_max"], K=packing["K"]),
        model=ModelConfig(
            byte_vocab=io["byte_vocab"],
            prompt_max_bytes=io["prompt_max_bytes"],
            target_max_bytes=io["target_max_bytes"],
            d_front=frontend["d_front"],
            d_ctx=context["d_ctx"],
            frontend_layers=frontend["layers"],
            target_layers=target["layers"],
            proposal_layers=proposal["layers"],
            proposal_hidden_dim=proposal["hidden_dim"],
            d_model=dit["d_model"],
            heads=dit["heads"],
            head_dim=dit["head_dim"],
            ffn_hidden=dit["ffn_hidden"],
            physical_blocks_L=dit["physical_blocks_L"],
            cycles_Q=dit["cycles_Q"],
            d_dec=decoder["d_dec"],
            decoder_layers=decoder["layers"],
        ),
        training=TrainingConfig(
            **training.get("objective", {})
            | {
                "learning_rate": training["learning_rate"],
                "weight_decay": training["weight_decay"],
                "grad_clip_norm": training["grad_clip_norm"],
                "batch_size": training["batch_size"],
            }
        ),
    )
