"""Deterministic random layered pointer-chasing benchmark for Phase E.1."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp


@dataclass(frozen=True)
class PointerSpec:
    """Versioned pointer benchmark layout."""

    name: str = "POINTER_V0"
    n_states: int = 10
    max_depth: int = 11
    queries: int = 8
    task_tag: int = 0xE1

    @property
    def prompt_bytes(self) -> int:
        return 2 + self.n_states * self.max_depth + self.queries

    @property
    def target_bytes(self) -> int:
        return self.queries


POINTER_V0 = PointerSpec()


def _permutation(key: jax.Array, n_states: int) -> jax.Array:
    return jax.random.permutation(key, n_states).astype(jnp.int32)


def generate_pointer_dataset(
    count: int, seed: int, *, spec: PointerSpec = POINTER_V0
) -> tuple[jax.Array, jax.Array, jax.Array, dict[str, Any]]:
    """Generate prompts, targets, active depths, and an audit payload."""
    if count < 1:
        raise ValueError("count must be positive")
    root = jax.random.PRNGKey(seed)
    keys = jax.random.split(root, count * (spec.max_depth + 2)).reshape(
        count, spec.max_depth + 2, 2
    )
    permutations = jax.vmap(
        lambda row: jax.vmap(lambda key: _permutation(key, spec.n_states))(row[: spec.max_depth])
    )(keys)
    depths = jax.vmap(lambda row: jax.random.randint(row[-2], (), 1, spec.max_depth + 1))(keys)
    starts = jax.vmap(lambda row: jax.random.randint(row[-1], (spec.queries,), 0, spec.n_states))(
        keys
    ).astype(jnp.int32)
    active = jnp.arange(spec.max_depth)[None, :] < depths[:, None]
    states = starts
    for layer in range(spec.max_depth):
        mapped = jnp.take_along_axis(permutations[:, layer, :], states, axis=1)
        states = jnp.where(active[:, layer, None], mapped, states)
    prompt = jnp.concatenate(
        (
            jnp.full((count, 1), spec.task_tag, dtype=jnp.int32),
            depths[:, None],
            permutations.reshape(count, -1),
            starts,
        ),
        axis=1,
    )
    audit = {
        "version": spec.name,
        "seed": seed,
        "count": count,
        "n_states": spec.n_states,
        "max_depth": spec.max_depth,
        "queries": spec.queries,
        "prompt_bytes": spec.prompt_bytes,
        "target_bytes": spec.target_bytes,
        "chance_byte_accuracy": 1.0 / spec.n_states,
        "chance_exact_sequence_accuracy": (1.0 / spec.n_states) ** spec.queries,
    }
    return prompt, states, depths.astype(jnp.int32), audit


def pointer_oracle(prompt: jax.Array, *, spec: PointerSpec = POINTER_V0) -> jax.Array:
    """Recompute targets directly from an encoded prompt."""
    if prompt.ndim != 2 or prompt.shape[1] != spec.prompt_bytes:
        raise ValueError("prompt has the wrong fixed width")
    depth = prompt[:, 1]
    tables = prompt[:, 2 : 2 + spec.n_states * spec.max_depth].reshape(
        prompt.shape[0], spec.max_depth, spec.n_states
    )
    states = prompt[:, 2 + spec.n_states * spec.max_depth :].astype(jnp.int32)
    for layer in range(spec.max_depth):
        mapped = jnp.take_along_axis(tables[:, layer, :], states, axis=1)
        states = jnp.where(layer < depth[:, None], mapped, states)
    return states


def pointer_intermediate_states(prompt: jax.Array, *, spec: PointerSpec = POINTER_V0) -> jax.Array:
    """Return x_1..x_D for each encoded example, retaining the final active state."""
    if prompt.ndim != 2 or prompt.shape[1] != spec.prompt_bytes:
        raise ValueError("prompt has the wrong fixed width")
    depth = prompt[:, 1]
    tables = prompt[:, 2 : 2 + spec.n_states * spec.max_depth].reshape(
        prompt.shape[0], spec.max_depth, spec.n_states
    )
    states = prompt[:, 2 + spec.n_states * spec.max_depth :].astype(jnp.int32)
    trajectory = []
    for layer in range(spec.max_depth):
        mapped = jnp.take_along_axis(tables[:, layer, :], states, axis=1)
        states = jnp.where(layer < depth[:, None], mapped, states)
        trajectory.append(states)
    return jnp.stack(trajectory, axis=1)


def audit_pointer_dataset(
    prompt: jax.Array,
    target: jax.Array,
    depths: jax.Array,
    *,
    spec: PointerSpec = POINTER_V0,
) -> dict[str, Any]:
    """Return deterministic structural checks used by the experiment runner."""
    tables = prompt[:, 2 : 2 + spec.n_states * spec.max_depth].reshape(
        prompt.shape[0], spec.max_depth, spec.n_states
    )
    sorted_tables = jnp.sort(tables, axis=-1)
    expected = jnp.broadcast_to(jnp.arange(spec.n_states), sorted_tables.shape)
    return {
        "version": spec.name,
        "prompt_width_constant": bool(prompt.shape[1] == spec.prompt_bytes),
        "target_width_constant": bool(target.shape[1] == spec.target_bytes),
        "all_depths_present": bool(jnp.all(jnp.isin(jnp.arange(1, spec.max_depth + 1), depths))),
        "permutations_valid": bool(jnp.all(sorted_tables == expected)),
        "oracle_matches_target": bool(jnp.all(pointer_oracle(prompt, spec=spec) == target)),
        "target_bytes_in_state_space": bool(jnp.all((target >= 0) & (target < spec.n_states))),
    }
