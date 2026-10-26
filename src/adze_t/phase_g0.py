"""Separately versioned G.0 codec and semantic/capacity routing policy.

Nothing in this module changes the accepted Phase-B/F teachers or the historical
Phase-G gate. Capacity edges are mechanical routing edges, NOT semantic labels.
"""

from __future__ import annotations

from typing import Any, NamedTuple

import jax
import jax.numpy as jnp

from .config import REFERENCE_SMALL_V0, ReferenceConfig
from .generated_structure import GeneratedRoutingResult, generated_routing_core
from .model import apply_target_codec
from .objectives import adamw_step, cross_entropy
from .phase_g import generated_structure_truth
from .state import CommittedStructure
from .training import _CODEC_ENCODER_NAMES


CODEC_VERSION = "PHASE_G0_VARIABLE_LENGTH_CODEC_V1"
CAPACITY_POLICY = "PHASE_G0_SEMANTIC_PUSHFORWARD_DYNAMIC_CAPACITY_V1"


def with_frozen_g0_codec(f1_torx_params: Any, candidate: Any) -> Any:
    """Build NEW matched-arm initialization; never mutate accepted F1 state.

    Overlay only candidate codec means. Preserve every existing rho and all
    noncodec means exactly. All three future G arms must hash this same tree.
    """
    from .backends.mapping import deterministic_to_torx

    mapped, _ = deterministic_to_torx(candidate)

    def overlay(old, new):
        if isinstance(old, dict):
            return {
                key: value if key == "rho" else overlay(value, new[key])
                for key, value in old.items()
            }
        if isinstance(old, (tuple, list)):
            return type(old)(overlay(a, b) for a, b in zip(old, new, strict=True))
        if old.shape != new.shape or old.dtype != new.dtype:
            raise ValueError("codec shape/dtype incompatible with F1")
        return jnp.array(new)

    encoder = {**f1_torx_params["encoder"]}
    for name in _CODEC_ENCODER_NAMES:
        encoder[name] = overlay(encoder[name], mapped["encoder"][name])
    return {
        **f1_torx_params,
        "encoder": encoder,
        "decoder": overlay(f1_torx_params["decoder"], mapped["decoder"]),
    }


class CapacityRouting(NamedTuple):
    semantic: CommittedStructure
    capacity_cut: jax.Array
    routing: GeneratedRoutingResult


def dynamic_capacity_routing(
    semantic: CommittedStructure,
    *,
    M_max: int,
    K: int,
    max_extent: int,
    upstream_valid: jax.Array | None = None,
    upstream_error: jax.Array | None = None,
) -> CapacityRouting:
    """Route full-site semantic actions without changing their probability law.

    Every nonterminal semantic action is sampled once, even at sites where the
    subsequent deterministic routing scan adds a capacity edge. The terminal is
    never sampled. Added edges have conditional probability one and no score
    term. Their mask remains separate from committed semantic boundaries.
    """
    if K <= 0 or semantic.c_b.ndim != 2 or semantic.c_b.shape[1] == 0:
        raise ValueError("positive K and nonempty [batch, carrier] boundaries required")
    shape = semantic.c_b.shape
    if (
        semantic.length is None
        or semantic.activity.shape != shape
        or semantic.length.shape != shape
    ):
        raise ValueError("explicit aligned semantic boundary/activity/length fields required")

    def step(run, edge):
        run = run + 1
        forced = (run == K) & (edge == 0)
        routing_edge = (edge == 1) | forced
        return jnp.where(routing_edge, 0, run), forced

    _, cuts = jax.lax.scan(step, jnp.zeros((shape[0],), jnp.int32), semantic.c_b.T)
    cuts = cuts.T
    routed = CommittedStructure(
        jnp.where(cuts, 1, semantic.c_b), semantic.activity, semantic.length
    )
    # Validate the original terminal/domain too; capacity cuts must never repair
    # malformed semantic input (e.g. missing terminal at a capacity boundary).
    original = generated_routing_core(
        semantic,
        M_max=shape[1],
        K=shape[1],
        max_extent=max_extent,
        upstream_valid=upstream_valid,
        upstream_error=upstream_error,
    )
    route = generated_routing_core(
        routed,
        M_max=M_max,
        K=K,
        max_extent=max_extent,
        upstream_valid=original.valid,
        upstream_error=original.error_code,
    )
    return CapacityRouting(semantic, cuts, route)


def codec_subtree(params: Any) -> dict[str, Any]:
    """No context, DiT, proposal, rho or accepted optimizer leaves enter training."""
    return {
        "encoder": {name: params["encoder"][name] for name in _CODEC_ENCODER_NAMES},
        "decoder": params["decoder"],
    }


def codec_objective(
    params: Any,
    target: jax.Array,
    mask: jax.Array,
    *,
    config: ReferenceConfig = REFERENCE_SMALL_V0,
) -> tuple[jax.Array, dict[str, jax.Array]]:
    output = apply_target_codec(params, target, mask, config=config)
    analysis = output["target"]
    teacher = analysis["teacher"]
    truth = generated_structure_truth(teacher.slot_bytes, teacher.length, K=config.packing.K)
    byte = cross_entropy(output["codec_logits"], teacher.slot_bytes, teacher.slot_mask)
    boundary = cross_entropy(
        analysis["b_logits"][:, :-1],
        truth.c_b[:, :-1],
        jnp.ones_like(truth.c_b[:, :-1], bool),
    )
    extent = cross_entropy(
        analysis["l_logits"], teacher.length, jnp.ones_like(teacher.length, bool)
    )
    return byte + boundary + extent, {"byte": byte, "boundary": boundary, "extent": extent}


def codec_update(
    params: Any,
    moments: tuple[Any, Any],
    step: jax.Array,
    target: jax.Array,
    mask: jax.Array,
    *,
    decoder_only: bool,
    config: ReferenceConfig = REFERENCE_SMALL_V0,
) -> tuple[Any, tuple[Any, Any], dict[str, jax.Array]]:
    (loss, metrics), gradients = jax.value_and_grad(codec_objective, has_aux=True)(
        params, target, mask, config=config
    )
    enabled = jax.tree.map(lambda _: jnp.asarray(True), params)
    if decoder_only:
        enabled["encoder"] = jax.tree.map(lambda _: jnp.asarray(False), params["encoder"])
    finite = jnp.isfinite(loss) & jnp.all(
        jnp.stack([jnp.all(jnp.isfinite(x)) for x in jax.tree.leaves(gradients)])
    )

    def update():
        return adamw_step(
            params,
            gradients,
            moments,
            step,
            learning_rate=3e-4,
            weight_decay=0.01,
            clip_norm=1.0,
            update_mask=enabled,
        )

    updated, new_moments, norm = jax.lax.cond(
        finite, update, lambda: (params, moments, jnp.asarray(0.0))
    )
    return updated, new_moments, {"loss": loss, "finite": finite, "grad_norm": norm, **metrics}
