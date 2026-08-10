"""Runtime checks for the public Torx surface required by Adze-T."""

from __future__ import annotations

import importlib
from dataclasses import dataclass


@dataclass(frozen=True)
class TorxPublicSurface:
    required_symbols: tuple[str, ...] = (
        "DFG",
        "Site",
        "ChainFactor",
        "TiledFactor",
    )


def inspect_public_surface() -> dict[str, bool]:
    """Return presence of required public symbols without using private APIs."""
    torx = importlib.import_module("torx")
    required = TorxPublicSurface().required_symbols
    return {name: hasattr(torx, name) for name in required}
