"""Fixed nonlinear state-transition benchmark for Phase E.1B."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from typing import Any

import jax
import jax.numpy as jnp


@dataclass(frozen=True)
class FixedTransitionSpec:
    """Versioned 64-bit periodic Rule-30 benchmark."""

    name: str = "FIXED_STATE_TRANSITION_V0"
    rule_name: str = "ELEMENTARY_CA_RULE_30_PERIODIC_64"
    task_tag: int = 0xE2
    state_bytes: int = 8
    depths: tuple[int, ...] = (1, 2, 3, 4, 6, 8, 12, 16)

    @property
    def state_bits(self) -> int:
        return 8 * self.state_bytes

    @property
    def prompt_bytes(self) -> int:
        return 2 + self.state_bytes

    @property
    def target_bytes(self) -> int:
        return self.state_bytes

    @property
    def max_depth(self) -> int:
        return max(self.depths)


FIXED_TRANSITION_V0 = FixedTransitionSpec()


def bytes_to_little_endian_bits(states: jax.Array) -> jax.Array:
    """Expand `[batch, 8]` bytes to bit positions 0..63, LSB first per byte."""
    if states.ndim != 2 or states.shape[1] != FIXED_TRANSITION_V0.state_bytes:
        raise ValueError("states must have shape [batch, 8]")
    shifts = jnp.arange(8, dtype=jnp.int32)
    return ((states.astype(jnp.int32)[..., None] >> shifts) & 1).reshape(
        states.shape[0], FIXED_TRANSITION_V0.state_bits
    )


def little_endian_bits_to_bytes(bits: jax.Array) -> jax.Array:
    """Pack bit positions 0..63 into `[batch, 8]` bytes, LSB first."""
    if bits.ndim != 2 or bits.shape[1] != FIXED_TRANSITION_V0.state_bits:
        raise ValueError("bits must have shape [batch, 64]")
    weights = 1 << jnp.arange(8, dtype=jnp.int32)
    return jnp.sum(bits.reshape(bits.shape[0], 8, 8).astype(jnp.int32) * weights, axis=-1)


def rule30_step(states: jax.Array) -> jax.Array:
    """Apply one periodic Rule-30 update: left XOR (center OR right)."""
    bits = bytes_to_little_endian_bits(states)
    left = jnp.roll(bits, 1, axis=1)
    right = jnp.roll(bits, -1, axis=1)
    updated = left ^ (bits | right)
    return little_endian_bits_to_bytes(updated)


def iterate_rule30(
    states: jax.Array,
    depths: jax.Array | int,
    *,
    spec: FixedTransitionSpec = FIXED_TRANSITION_V0,
) -> jax.Array:
    """Apply the fixed rule exactly `depths` times to each initial state."""
    requested = jnp.asarray(depths, dtype=jnp.int32)
    if requested.ndim == 0:
        requested = jnp.full((states.shape[0],), requested, dtype=jnp.int32)
    if requested.shape != (states.shape[0],):
        raise ValueError("depths must be scalar or have one value per state")
    if bool(jnp.any(~jnp.isin(requested, jnp.asarray(spec.depths)))):
        raise ValueError("requested depth is outside the predeclared depth set")
    current = states.astype(jnp.int32)
    for step in range(1, spec.max_depth + 1):
        updated = rule30_step(current)
        current = jnp.where((requested >= step)[:, None], updated, current)
    return current


def generate_fixed_transition_dataset(
    count: int,
    seed: int,
    *,
    spec: FixedTransitionSpec = FIXED_TRANSITION_V0,
) -> tuple[jax.Array, jax.Array, jax.Array, dict[str, Any]]:
    """Generate a balanced fixed-depth corpus with random compact initial states."""
    if count < 1 or count % len(spec.depths) != 0:
        raise ValueError("count must be positive and divisible by the number of depths")
    state_key, order_key = jax.random.split(jax.random.PRNGKey(seed))
    initial = jax.random.randint(state_key, (count, spec.state_bytes), 0, 256, dtype=jnp.int32)
    depths = jnp.tile(jnp.asarray(spec.depths, dtype=jnp.int32), count // len(spec.depths))
    order = jax.random.permutation(order_key, count)
    initial = initial[order]
    depths = depths[order]
    target = iterate_rule30(initial, depths, spec=spec)
    prompt = jnp.concatenate(
        (
            jnp.full((count, 1), spec.task_tag, dtype=jnp.int32),
            depths[:, None],
            initial,
        ),
        axis=1,
    )
    audit = {
        "version": spec.name,
        "rule": spec.rule_name,
        "seed": seed,
        "count": count,
        "depths": spec.depths,
        "prompt_bytes": spec.prompt_bytes,
        "target_bytes": spec.target_bytes,
        "chance_bit_accuracy": 0.5,
        "chance_byte_accuracy": 1.0 / 256.0,
        "chance_exact_state_accuracy": 2.0**-spec.state_bits,
    }
    return prompt, target, depths, audit


def fixed_transition_oracle(
    prompt: jax.Array, *, spec: FixedTransitionSpec = FIXED_TRANSITION_V0
) -> jax.Array:
    """Decode a fixed-width prompt and independently request exact iteration."""
    if prompt.ndim != 2 or prompt.shape[1] != spec.prompt_bytes:
        raise ValueError("prompt has the wrong fixed width")
    if bool(jnp.any(prompt[:, 0] != spec.task_tag)):
        raise ValueError("prompt task tag does not match FIXED_STATE_TRANSITION_V0")
    return iterate_rule30(prompt[:, 2:], prompt[:, 1], spec=spec)


def fixed_transition_example_hashes(
    prompt: jax.Array, target: jax.Array, depths: jax.Array
) -> set[str]:
    """Hash complete examples for full split-independence audits."""
    if prompt.shape[0] != target.shape[0] or prompt.shape[0] != depths.shape[0]:
        raise ValueError("prompt, target, and depths must contain the same number of examples")
    prompt_host, target_host, depths_host = jax.device_get((prompt, target, depths))
    hashes = set()
    for sample_prompt, sample_target, sample_depth in zip(
        prompt_host, target_host, depths_host, strict=True
    ):
        digest = hashlib.sha256()
        digest.update(sample_prompt.tobytes())
        digest.update(sample_target.tobytes())
        digest.update(sample_depth.tobytes())
        hashes.add(digest.hexdigest())
    return hashes


def audit_fixed_transition_dataset(
    prompt: jax.Array,
    target: jax.Array,
    depths: jax.Array,
    *,
    spec: FixedTransitionSpec = FIXED_TRANSITION_V0,
) -> dict[str, Any]:
    """Return hard structural and oracle checks for a generated corpus."""
    return {
        "version": spec.name,
        "rule": spec.rule_name,
        "prompt_width_constant": bool(prompt.ndim == 2 and prompt.shape[1] == spec.prompt_bytes),
        "target_width_constant": bool(target.ndim == 2 and target.shape[1] == spec.target_bytes),
        "task_tag_valid": bool(jnp.all(prompt[:, 0] == spec.task_tag)),
        "depths_valid": bool(jnp.all(jnp.isin(depths, jnp.asarray(spec.depths)))),
        "depth_encoding_matches": bool(jnp.all(prompt[:, 1] == depths)),
        "depths_balanced": bool(
            jnp.all(
                jnp.asarray([jnp.sum(depths == depth) for depth in spec.depths])
                == prompt.shape[0] // len(spec.depths)
            )
        ),
        "oracle_matches_target": bool(
            jnp.all(fixed_transition_oracle(prompt, spec=spec) == target)
        ),
        "every_target_byte_derived": bool(
            jnp.all(iterate_rule30(prompt[:, 2:], depths, spec=spec) == target)
        ),
        "bytes_in_range": bool(jnp.all((prompt >= 0) & (prompt < 256)) and jnp.all(target < 256)),
    }


def _binary_entropy(probability: jax.Array) -> jax.Array:
    safe = jnp.clip(probability, 1.0e-12, 1.0 - 1.0e-12)
    return -(safe * jnp.log2(safe) + (1.0 - safe) * jnp.log2(1.0 - safe))


def _byte_entropy(states: jax.Array) -> float:
    entropies = []
    for byte_index in range(states.shape[1]):
        probabilities = jnp.bincount(states[:, byte_index], length=256) / states.shape[0]
        positive = probabilities > 0
        entropies.append(
            -jnp.sum(jnp.where(positive, probabilities * jnp.log2(probabilities), 0.0))
        )
    return float(jnp.mean(jnp.stack(entropies)))


def transition_quality_audit(
    initial: jax.Array,
    *,
    spec: FixedTransitionSpec = FIXED_TRANSITION_V0,
) -> dict[str, Any]:
    """Audit Rule 30 without consulting model performance."""
    if initial.ndim != 2 or initial.shape[1] != spec.state_bytes:
        raise ValueError("initial states must have shape [batch, 8]")
    trajectory = [initial.astype(jnp.int32)]
    for _ in range(spec.max_depth):
        trajectory.append(rule30_step(trajectory[-1]))

    encountered_short_cycle = jnp.zeros((initial.shape[0],), dtype=bool)
    rows = []
    for depth in range(1, spec.max_depth + 1):
        output = trajectory[depth]
        previous = trajectory[depth - 1]
        for period in range(1, min(4, depth) + 1):
            encountered_short_cycle |= jnp.all(output == trajectory[depth - period], axis=1)
        bits = bytes_to_little_endian_bits(output)
        bit_probabilities = jnp.mean(bits, axis=0)
        host = jax.device_get(output)
        unique_outputs = len({row.tobytes() for row in host})
        exact_counts: dict[bytes, int] = {}
        for row in host:
            key = row.tobytes()
            exact_counts[key] = exact_counts.get(key, 0) + 1
        rows.append(
            {
                "depth": depth,
                "mean_hamming_from_initial": jnp.mean(
                    jnp.sum(bits != bytes_to_little_endian_bits(initial), axis=1)
                ),
                "mean_hamming_last_step": jnp.mean(
                    jnp.sum(bits != bytes_to_little_endian_bits(previous), axis=1)
                ),
                "collision_rate": 1.0 - unique_outputs / initial.shape[0],
                "identity_rate": jnp.mean(jnp.all(output == initial, axis=1)),
                "fixed_transition_rate": jnp.mean(jnp.all(output == rule30_step(output), axis=1)),
                "entered_period_le_4_rate": jnp.mean(encountered_short_cycle),
                "mean_bit_entropy": jnp.mean(_binary_entropy(bit_probabilities)),
                "mean_byte_entropy": _byte_entropy(output),
                "all_zero_rate": jnp.mean(jnp.all(output == 0, axis=1)),
                "all_one_rate": jnp.mean(jnp.all(output == 255, axis=1)),
                "modal_exact_state_rate": max(exact_counts.values()) / initial.shape[0],
            }
        )
    selected = [row for row in rows if row["depth"] in spec.depths]
    return {
        "rule": spec.rule_name,
        "sample_count": initial.shape[0],
        "depths": selected,
        "quality_gate": {
            "non_identity": all(float(row["mean_hamming_last_step"]) > 8.0 for row in selected),
            "no_constant_collapse": all(
                float(row["all_zero_rate"] + row["all_one_rate"]) < 0.01 for row in selected
            ),
            "no_short_cycle_collapse": all(
                float(row["entered_period_le_4_rate"]) < 0.01 for row in selected
            ),
            "high_bit_entropy": all(float(row["mean_bit_entropy"]) > 0.95 for row in selected),
            "high_byte_entropy": all(
                float(row["mean_byte_entropy"]) > math.log2(256) * 0.9 for row in selected
            ),
        },
    }
