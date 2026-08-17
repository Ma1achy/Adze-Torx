"""Collision-free Phase E.1 working-state and evidence identities."""

from pathlib import Path
from typing import Any, Callable


def seed_identity(init_seed: int, stochastic_training_seed: int) -> str:
    """Return the canonical seed suffix used by all Phase E.1 artifacts."""
    return f"init{init_seed}_stoch{stochastic_training_seed}"


def checkpoint_path(
    root: Path,
    *,
    benchmark: str,
    stage: str,
    arm: str,
    init_seed: int,
    stochastic_training_seed: int,
) -> Path:
    """Resolve a checkpoint whose identity includes benchmark, stage, arm, and seeds."""
    return (
        root / benchmark / stage / arm / f"{seed_identity(init_seed, stochastic_training_seed)}.pkl"
    )


def evidence_path(
    root: Path,
    *,
    benchmark: str,
    stage: str,
    stem: str,
    init_seed: int,
    stochastic_training_seed: int,
    suffix: str,
) -> Path:
    """Resolve seed-specific evidence without permitting cross-seed overwrite."""
    return (
        root
        / benchmark
        / stage
        / f"{stem}_{seed_identity(init_seed, stochastic_training_seed)}{suffix}"
    )


def resolve_run_state(
    checkpoint: Path,
    *,
    load_state: Callable[[Path], tuple[Any, Any, int]],
    initialize_state: Callable[[], tuple[Any, Any, int]],
) -> tuple[Any, Any, int]:
    """Resume only the exact resolved checkpoint or return a scratch state."""
    return load_state(checkpoint) if checkpoint.exists() else initialize_state()
