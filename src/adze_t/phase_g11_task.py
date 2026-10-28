"""Separately versioned observation-latent future-peer benchmark; no training."""

from __future__ import annotations

import jax
import jax.numpy as jnp

TASK_VERSION = "PHASE_G11_FUTURE_PEER_OBSERVATION_LATENT_V0"
ERASURE_VERSION = "PHASE_G11_PUBLIC_QUERY_PAYLOAD_ERASURE_V0"
END, CONTINUE, QUERY, BLANK = 1, 2, 3, 4
PAYLOAD_MIN, PAYLOAD_MAX = 5, 33


def symbolic_structure(source: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Read only declared nonzero observation bytes, not teacher metadata."""
    padded = jnp.pad(source, ((0, 0), (0, 64))).reshape(-1, 32, 4)
    ell = jnp.sum(padded != 0, -1).astype(jnp.int32)
    b = ((padded[:, :, 0] == END) & (ell > 0)).astype(jnp.int32).at[:, -1].set(1)
    return b, ell


def symbolic_target(source: jax.Array, q: jax.Array) -> jax.Array:
    """Information-availability oracle, NOT a causal model/decoder replacement."""
    slots = source.reshape(-1, 16, 4)
    candidate = (slots[:, :, 0] == END) & (jnp.arange(16)[None] > q[:, None])
    peer = jnp.argmax(candidate, -1)
    return (
        slots.at[jnp.arange(len(q)), q, 1:].set(slots[jnp.arange(len(q)), peer, 1:]).reshape(-1, 64)
    )


def make_examples(seed: int, per_length: int, *, id_offset: int = 0) -> dict[str, jax.Array]:
    if per_length < 1:
        raise ValueError("positive support for every length is required")
    keys = jax.random.split(jax.random.key(seed), 6)
    n = per_length * 57
    counts = jax.random.permutation(keys[0], jnp.repeat(jnp.arange(8, 65), per_length))
    full = counts // 4
    q = jax.random.randint(keys[1], (n,), 0, full - 1)
    span = 2 + jax.random.randint(keys[2], (n,), 0, jnp.minimum(5, full - q - 1))
    peer = q + span - 1
    sites = jnp.arange(32)[None]
    ell = jnp.clip(counts[:, None] - 4 * sites, 0, 4).astype(jnp.int32)
    active = ell > 0
    b = (jax.random.bernoulli(keys[3], 0.5, (n, 32)) & active).astype(jnp.int32)
    b = jnp.where((sites >= q[:, None]) & (sites < peer[:, None]), 0, b)
    rows = jnp.arange(n)
    b = b.at[rows, peer].set(1)
    prior = jnp.maximum(q - 1, 0)
    b = b.at[rows, prior].set(jnp.where(q > 0, 1, b[rows, prior]))
    b = b.at[rows, (counts + 3) // 4 - 1].set(1).at[:, -1].set(1)
    payload = jax.random.randint(keys[4], (n, 32, 3), PAYLOAD_MIN, PAYLOAD_MAX)
    role = jnp.where(b == 1, END, CONTINUE).at[rows, q].set(QUERY)
    source = jnp.concatenate((role[:, :, None], payload), -1)
    source = source.at[rows, q, 1:].set(BLANK)
    slot_mask = jnp.arange(4)[None, None] < ell[:, :, None]
    source = jnp.where(slot_mask, source, 0).reshape(n, 128)[:, :64]
    target = symbolic_target(source, q)
    mask = jnp.arange(64)[None] < counts[:, None]
    # Marker truth is defined from the public observation grammar, not latent labels.
    from_source_b, from_source_l = symbolic_structure(source)
    if not bool(jnp.array_equal(b, from_source_b) & jnp.array_equal(ell, from_source_l)):
        raise RuntimeError("grammar construction failed")
    return {
        "source": source,
        "source_mask": mask,
        "target": target,
        "target_mask": mask,
        "boundary": b,
        "length": ell,
        "q": q,
        "peer": peer,
        "prompt": jnp.stack((jnp.full_like(q, 0xF1), q + 1), -1),
        "prompt_mask": jnp.ones((n, 2), bool),
        "global_example_id": jnp.arange(id_offset, id_offset + n, dtype=jnp.uint32),
    }


def relabel(data: dict[str, jax.Array], payload: jax.Array) -> dict[str, jax.Array]:
    """Change only future-peer label; observation query stays answer-free."""
    n = len(data["q"])
    rows = jnp.arange(n)
    source = data["source"].reshape(n, 16, 4).at[rows, data["peer"], 1:].set(payload)
    source = source.reshape(n, 64)
    return {**data, "source": source, "target": symbolic_target(source, data["q"])}


def cut_after_query(data: dict[str, jax.Array]) -> jax.Array:
    return data["boundary"].at[jnp.arange(len(data["q"])), data["q"]].set(1)


def observation_control(data: dict[str, jax.Array], name: str, nuisance: jax.Array) -> jax.Array:
    """Return observation bytes only. Never consult target payloads/masks."""
    n = len(data["q"])
    slots = data["source"].reshape(n, 16, 4)
    rows = jnp.arange(n)
    if name == "C":
        slots = slots.at[rows, data["peer"], 1:].set(BLANK)
    elif name == "D_substitute":
        slots = slots.at[rows, data["peer"], 1:].set(nuisance[rows, data["peer"]])
    elif name == "D_outside":
        keep = (jnp.arange(16)[None] == data["q"][:, None]) | (
            jnp.arange(16)[None] == data["peer"][:, None]
        )
        payload = jnp.where(keep[:, :, None], slots[:, :, 1:], nuisance)
        valid = data["source_mask"].reshape(n, 16, 4)[:, :, 1:]
        slots = slots.at[:, :, 1:].set(jnp.where(valid, payload, 0))
    else:
        raise ValueError("unknown observation control")
    return slots.reshape(n, 64)
