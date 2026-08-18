"""Leakage-safe DENOISE_V0 data and initial-corruption helpers."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Any

import jax
from jax import Array
import jax.numpy as jnp

from .corruption import (
    DiffusionStage,
    corrupt_h,
    diffusion_key,
    training_diffusion_root,
)


PHASE_F_1_DENOISE_V0_PROPOSAL_AUX_DISABLED = "PHASE_F_1_DENOISE_V0_PROPOSAL_AUX_DISABLED"


@dataclass(frozen=True)
class DenoiseV0Spec:
    name: str = "DENOISE_V0"
    task_tag: int = 0xF1
    prompt_bytes: int = 1
    target_bytes: int = 8


DENOISE_V0 = DenoiseV0Spec()


def generate_denoise_v0(
    count: int, seed: int, *, spec: DenoiseV0Spec = DENOISE_V0
) -> tuple[Array, Array, Array]:
    """Generate constant-context examples with uniformly random byte targets."""
    if count < 1:
        raise ValueError("count must be positive")
    target = jax.random.randint(
        jax.random.PRNGKey(seed),
        (count, spec.target_bytes),
        0,
        256,
        dtype=jnp.int32,
    )
    prompt = jnp.full((count, spec.prompt_bytes), spec.task_tag, dtype=jnp.int32)
    global_example_ids = jnp.arange(count, dtype=jnp.uint32)
    return prompt, target, global_example_ids


def denoise_example_hashes(prompt: Array, target: Array) -> set[str]:
    """Hash every complete prompt/target example for split audits."""
    if prompt.shape[0] != target.shape[0]:
        raise ValueError("prompt and target must contain the same number of examples")
    prompt_host, target_host = jax.device_get((prompt, target))
    hashes = set()
    for sample_prompt, sample_target in zip(prompt_host, target_host, strict=True):
        digest = hashlib.sha256()
        digest.update(sample_prompt.tobytes())
        digest.update(sample_target.tobytes())
        hashes.add(digest.hexdigest())
    return hashes


def initial_diffusion_epsilon(
    h0: Array,
    root_key: Array,
    global_example_ids: Array,
    *,
    optimizer_step: int | Array | None = None,
) -> Array:
    """Sample per-example epsilon, optionally freshened by training occurrence."""
    if h0.shape[0] != global_example_ids.shape[0]:
        raise ValueError("h0 and global_example_ids batch dimensions must match")
    root = root_key if optimizer_step is None else training_diffusion_root(root_key, optimizer_step)
    keys = jax.vmap(
        lambda example_id: diffusion_key(
            root,
            global_example_id=example_id,
            stage=DiffusionStage.INITIAL_CORRUPTION,
            denoise_step=0,
        )
    )(global_example_ids)
    return jax.vmap(lambda key: jax.random.normal(key, h0.shape[1:], dtype=h0.dtype))(keys)


def make_initial_corruption(
    h0: Array,
    nu: Array,
    root_key: Array,
    global_example_ids: Array,
    *,
    optimizer_step: int | Array | None = None,
) -> tuple[Array, Array]:
    """Construct paired validation or occurrence-fresh training corruption."""
    epsilon = initial_diffusion_epsilon(
        h0,
        root_key,
        global_example_ids,
        optimizer_step=optimizer_step,
    )
    return corrupt_h(h0, nu, epsilon), epsilon


def dataset_audit(
    prompt: Array, target: Array, *, spec: DenoiseV0Spec = DENOISE_V0
) -> dict[str, Any]:
    """Return model-independent DENOISE_V0 format checks."""
    return {
        "task_version": spec.name,
        "count": int(prompt.shape[0]),
        "constant_prompt": bool(jnp.all(prompt == spec.task_tag)),
        "prompt_shape_valid": prompt.shape == (target.shape[0], spec.prompt_bytes),
        "target_shape_valid": target.shape == (prompt.shape[0], spec.target_bytes),
        "target_bytes_in_range": bool(jnp.all((target >= 0) & (target < 256))),
        "all_target_bytes_meaningful": True,
    }
