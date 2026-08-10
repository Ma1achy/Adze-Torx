"""Runtime checks for the pinned, documented Torx public surface."""

from __future__ import annotations

import importlib
import importlib.metadata

PUBLIC_SYMBOLS = (
    "torx.AbstractFactor",
    "torx.AbstractHasLogProbability",
    "torx.AbstractFiniteStateSpaceFactor",
    "torx.AbstractHasExplicitOutputDistribution",
    "torx.DFG",
    "torx.Site",
    "torx.ChainFactor",
    "torx.TiledFactor",
    "torx.psc.DiscretePCircuit",
    "torx.psc.HybridPCircuit",
    "torx.psc.PNOT",
    "torx.psc.AffineGaussianGate",
    "torx.psc.MixtureGaussianGate",
    "torx.psc.BranchingSimulator",
    "torx.psc.AffineGaussianSimulator",
)


def _resolve(path: str):
    module_name, symbol = path.rsplit(".", 1)
    return getattr(importlib.import_module(module_name), symbol, None)


def inspect_public_surface() -> dict[str, bool]:
    """Return presence of every public symbol used by M1."""
    return {path: _resolve(path) is not None for path in PUBLIC_SYMBOLS}


def environment() -> dict[str, str]:
    """Return reproducibility metadata without depending on private Torx state."""
    import jax

    try:
        torx_version = importlib.metadata.version("extro-torx")
    except importlib.metadata.PackageNotFoundError:
        torx_version = "not installed"
    return {
        "python": __import__("platform").python_version(),
        "jax": jax.__version__,
        "torx_distribution": torx_version,
        "torx_commit": "f1fc858ed950ecd41935d15c06d0ec7c5e0674ae",
        "platform": __import__("platform").platform(),
        "devices": repr(jax.devices()),
        "jax_enable_x64": repr(getattr(jax.config, "jax_enable_x64", False)),
    }
