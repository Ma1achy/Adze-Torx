"""Frozen G.1.1 teacher objective and fail-closed optimizer transition."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp

from .backends.torx import TorxOps, TorxOperatorConfig
from .denoise import make_sanitized_target_analysis, as_sanitized_model_target_analysis
from .model import apply_model
from .phase_g import neutral_structure, generated_byte_metrics, fail_closed_adamw_step
from .phase_g0 import dynamic_capacity_routing
from .state import CommittedStructure
from .training import stochastic_model_update_mask

LOSS_VERSION = "PHASE_G11_TEACHER_QUERY_RECONSTRUCTION_LOSS_V0"


def ce(logits, labels):
    return -jnp.take_along_axis(jax.nn.log_softmax(logits), labels[..., None], -1)[..., 0]


def apply_observation(params, h, prompt, structure, ops, *, config):
    route = dynamic_capacity_routing(
        structure, M_max=config.packing.M_max, K=config.packing.K, max_extent=config.carrier.L_max
    ).routing
    neutral = neutral_structure(len(h), config)
    assert neutral.length is not None
    safe = make_sanitized_target_analysis(
        neutral.c_b,
        neutral.length,
        config=config,
        target_width=min(64, config.carrier.C * config.carrier.L_max),
    )

    def execute(_):
        out = apply_model(
            params,
            prompt,
            jnp.ones_like(prompt, bool),
            safe.target_bytes_placeholder,
            safe.target_mask_placeholder,
            config=config,
            ops=ops,
            target_analysis=as_sanitized_model_target_analysis(safe),
            carrier_h_input=h,
            noise_level=0.0,
            actual_s_index=0,
            denoise_condition_index=0,
            committed_c_b=route.structure.c_b,
            committed_length=route.structure.length,
            committed_activity=route.structure.activity,
            observed_b=neutral.c_b,
            observed_l=neutral.length,
            routing_metadata=route.metadata,
        )
        return out["prediction"][0], out["prediction"][1], out["prediction"][2], out["byte_logits"]

    def skip(_):
        return (
            jnp.zeros_like(h),
            jnp.zeros((*h.shape[:2], 2), h.dtype),
            jnp.zeros((*h.shape[:2], config.carrier.L_max + 1), h.dtype),
            jnp.zeros((*h.shape[:2], config.carrier.L_max, 256), h.dtype),
        )

    return route.executed, jax.lax.cond(route.executed, execute, skip, operand=None)


def teacher_objective(params, batch, step, *, cut=False, config):
    """Target tensors enter loss only; both model applications use source_h."""
    h = jax.lax.stop_gradient(batch["source_h"])
    target_h = jax.lax.stop_gradient(batch["target_h"])
    b = batch["boundary"]
    if cut:
        b = b.at[jnp.arange(len(h)), batch["q"]].set(1)
    ell = batch["length"]
    structure = CommittedStructure(b, ell > 0, ell)
    valid = dynamic_capacity_routing(
        structure, M_max=config.packing.M_max, K=config.packing.K, max_extent=config.carrier.L_max
    ).routing.executed
    ops = TorxOps.create(
        jax.random.key(1146),
        config=TorxOperatorConfig(operator_stochasticity=True, lambda_op=0.0),
        optimizer_step=step,
        # Frozen lambda=0 validation convention: one application context. IDs
        # remain on every evidence row; no stochastic sample is shared/reused.
        global_example_id=0,
    )

    def execute(_):
        _, prediction = apply_observation(
            params, h, batch["prompt"], neutral_structure(len(h), config), ops, config=config
        )
        _, replay = apply_observation(params, h, batch["prompt"], structure, ops, config=config)
        _, pb, pl, _ = prediction
        hh, _, _, logits = replay
        bce = ce(pb[:, :-1], batch["boundary"][:, :-1]).mean(1)
        lce = ce(pl, ell).mean(1)
        mse = jnp.mean((hh - target_h) ** 2, (1, 2))
        slots, mask = batch["slots"], batch["slot_mask"]
        byte = generated_byte_metrics(logits, ell, slots, mask)
        row = jnp.arange(len(h))
        q = batch["q"]
        qlogits = logits[row, q, 1:]
        qlabel = slots[row, q, 1:]
        qnll = ce(qlogits, qlabel).mean(1)
        qcorrect = jnp.argmax(qlogits, -1) == qlabel
        is_q = jnp.arange(config.carrier.C)[None, :, None] == q[:, None, None]
        payload = jnp.arange(config.carrier.L_max)[None, None, :] > 0
        nmask = mask & ~is_q
        complement = mask & ~(is_q & payload)
        all_ce = ce(logits, slots)
        correct = jnp.argmax(logits, -1) == slots
        ncount = nmask.sum((1, 2))
        cn = complement.sum((1, 2))
        bp = jnp.argmax(pb, -1).at[:, -1].set(1)
        lp = jnp.argmax(pl, -1)
        boundary = batch["boundary"]
        tp = ((bp[:, :-1] == 1) & (boundary[:, :-1] == 1)).sum(1)
        fp = ((bp[:, :-1] == 1) & (boundary[:, :-1] == 0)).sum(1)
        fn = ((bp[:, :-1] == 0) & (boundary[:, :-1] == 1)).sum(1)
        last = jnp.maximum(jnp.sum(ell > 0, 1) - 1, 0)
        partial = (ell.sum(1) % config.carrier.L_max) > 0
        full_mask = nmask & (ell[:, :, None] == config.carrier.L_max)
        partial_mask = nmask & (ell[:, :, None] < config.carrier.L_max)
        metrics = {
            "b_ce": bce,
            "ell_ce": lce,
            "replay_h_mse": mse,
            "global_byte_loss": byte.loss_per_example,
            "query_nll": qnll,
            "query_marker_accuracy": correct[row, q, 0],
            "query_marker_nll": all_ce[row, q, 0],
            "missing_bytes": byte.missing_count,
            "extra_bytes": byte.extra_count,
            "committed_structure_exact": jnp.ones((len(h),), bool),
            "extent_class_support": (ell[:, :, None] == jnp.arange(config.carrier.L_max + 1)).sum(
                1
            ),
            "extent_class_correct": (
                (ell[:, :, None] == jnp.arange(config.carrier.L_max + 1)) & (lp == ell)[:, :, None]
            ).sum(1),
            "query_accuracy": qcorrect.mean(1),
            "query_exact": jnp.all(qcorrect, 1),
            "nonquery_nll": jnp.sum(jnp.where(nmask, all_ce, 0), (1, 2)) / jnp.maximum(ncount, 1),
            "nonquery_accuracy": jnp.sum(correct & nmask, (1, 2)) / jnp.maximum(ncount, 1),
            "nonquery_count": ncount,
            "complement_nll": jnp.sum(jnp.where(complement, all_ce, 0), (1, 2))
            / jnp.maximum(cn, 1),
            "complement_accuracy": jnp.sum(correct & complement, (1, 2)) / jnp.maximum(cn, 1),
            "byte_accuracy": byte.accuracy_per_example,
            "byte_nll": byte.overlap_nll_per_example,
            "byte_exact": byte.exact_per_example,
            "boundary_accuracy": (bp[:, :-1] == boundary[:, :-1]).mean(1),
            "boundary_exact": jnp.all(bp == boundary, 1),
            "extent_accuracy": (lp == ell).mean(1),
            "extent_exact": jnp.all(lp == ell, 1),
            "total_length_accuracy": lp.sum(1) == ell.sum(1),
            "structure_exact": jnp.all(bp == boundary, 1) & jnp.all(lp == ell, 1),
            "boundary_tp": tp,
            "boundary_fp": fp,
            "boundary_fn": fn,
            "partial_support": partial,
            "partial_correct": (lp[row, last] == ell[row, last]) & partial,
            "structural_churn": (
                (b != neutral_structure(len(h), config).c_b) | (ell != config.carrier.L_max)
            ).mean(1),
            "full_nonquery_count": full_mask.sum((1, 2)),
            "partial_nonquery_count": partial_mask.sum((1, 2)),
            "full_nonquery_nll_sum": jnp.where(full_mask, all_ce, 0).sum((1, 2)),
            "partial_nonquery_nll_sum": jnp.where(partial_mask, all_ce, 0).sum((1, 2)),
            "full_nonquery_correct": (correct & full_mask).sum((1, 2)),
            "partial_nonquery_correct": (correct & partial_mask).sum((1, 2)),
            "predicted_b": bp,
            "predicted_l": lp,
        }
        total = bce + lce + 0.5 * mse + byte.loss_per_example + qnll
        return total.mean(), {**metrics, "loss_per_example": total, "executed": jnp.asarray(True)}

    # Match types/shapes using abstract evaluation; no model execution in invalid branch.
    template = jax.eval_shape(execute, None)

    def skip(_):
        return jax.tree.map(lambda x: jnp.zeros(x.shape, x.dtype), template)

    return jax.lax.cond(valid, execute, skip, operand=None)


def train_step(params: Any, moments: Any, step, batch, *, config):
    (loss, metrics), grads = jax.value_and_grad(
        lambda p: teacher_objective(p, batch, step, config=config), has_aux=True
    )(params)
    mask = stochastic_model_update_mask(params)
    masked = jax.tree.map(lambda g, e: jnp.where(e, g, 0.0), grads, mask)
    norm = jnp.sqrt(sum(jnp.sum(x * x) for x in jax.tree.leaves(masked)))
    finite = (
        jnp.isfinite(loss)
        & jnp.isfinite(norm)
        & jnp.all(jnp.stack([jnp.isfinite(x).all() for x in jax.tree.leaves(grads)]))
    )
    p, m, s, raw = fail_closed_adamw_step(
        params, grads, moments, step, metrics["executed"] & finite, mask
    )
    return (
        p,
        m,
        s,
        {
            **metrics,
            "loss": loss,
            "masked_raw_gradient_norm": norm,
            "applied_gradient_norm": jnp.minimum(raw, 1.0),
            "executed": metrics["executed"] & finite & (s == step + 1),
        },
    )
