"""G2 observation-only generated inference and once-owned structural loss.

Historical G/F interfaces and the validated score bridge are unchanged.
"""

from typing import Any, NamedTuple, cast

import jax
import jax.numpy as jnp

from .backends.torx import TorxOps, TorxOperatorConfig
from .generated_structure import (
    StructureEstimatorOwner,
    CommitPolicy,
    StructureKeyContext,
    commit_predicted_structure_core,
    full_commit_masks,
    score_corrected_structure_loss,
)
from .masking import draft_attention_mask
from .phase_g import (
    _compact_slots,
    generated_byte_metrics,
    neutral_structure,
    fail_closed_adamw_step,
)
from .phase_g0 import dynamic_capacity_routing
from .phase_g11_training import apply_observation, ce
from .state import CommittedStructure, PredictedStructure
from .training import stochastic_model_update_mask


MASTER_SEED = 12000


def named_key(namespace: str, seed: int | jax.Array = 0, root: int | jax.Array = 0):
    import hashlib

    code = int.from_bytes(hashlib.blake2s(namespace.encode(), digest_size=4).digest(), "little")
    key = jax.random.fold_in(jax.random.key(MASTER_SEED), code)
    key = jax.random.fold_in(key, jnp.asarray(seed, jnp.uint32))
    return jax.random.fold_in(key, jnp.asarray(root, jnp.uint32))


class Forward(NamedTuple):
    h_hat: jax.Array
    logits: jax.Array
    b_logits: jax.Array
    l_logits: jax.Array
    commit: Any
    reach: jax.Array
    capacity_cut: jax.Array
    executed: jax.Array
    carrier_to_m: jax.Array
    carrier_to_k: jax.Array


def reachability(route, *, config):
    m = route.metadata
    pos = jnp.maximum(m.carrier_to_m * config.packing.K + m.carrier_to_k, 0)
    mask = draft_attention_mask(m.block_id, m.query_mask, m.kv_mask)
    return mask[jnp.arange(len(pos))[:, None, None], pos[:, :, None], pos[:, None, :]]


def inference(
    params,
    source_h,
    prompt,
    global_ids,
    structure_key,
    operator_key,
    occurrence,
    *,
    policy,
    config,
    teacher: CommittedStructure | None = None,
):
    """B/C accept no target bytes, masks, teacher structure or clean target h.

    The optional teacher argument is rejected outside explicitly labelled A.
    """
    if policy not in ("A", "B", "C"):
        raise ValueError("unknown arm")
    if policy != "A" and teacher is not None:
        raise ValueError("teacher structure forbidden in generated inference")
    if policy == "A" and teacher is None:
        raise ValueError("oracle A requires explicit teacher adapter")
    h = jax.lax.stop_gradient(source_h)
    neutral = neutral_structure(len(h), config)
    ops = TorxOps.create(
        operator_key,
        config=TorxOperatorConfig(operator_stochasticity=True, lambda_op=0.0),
        optimizer_step=occurrence,
    )
    predictor_valid, (_, pb, pl, _) = apply_observation(
        params, h, prompt, neutral, ops, config=config
    )
    previous = teacher if policy == "A" else neutral
    assert previous is not None
    previous_check = dynamic_capacity_routing(
        previous, M_max=config.packing.M_max, K=config.packing.K, max_extent=config.carrier.L_max
    ).routing
    commit = commit_predicted_structure_core(
        previous,
        PredictedStructure(pb, pl),
        full_commit_masks(h.shape[:2]),
        policy=cast(CommitPolicy, {"A": "prediction_only", "B": "argmax", "C": "sample"}[policy]),
        max_extent=config.carrier.L_max,
        key_context=StructureKeyContext(structure_key, occurrence, global_ids, 0, 0, 0),
        temperature=1.0,
        estimator_owner=StructureEstimatorOwner.LOCAL_SCORE_BRIDGE
        if policy == "C"
        else StructureEstimatorOwner.NONE,
        allow_reactivation=True,
    )
    route = dynamic_capacity_routing(
        commit.committed,
        M_max=config.packing.M_max,
        K=config.packing.K,
        max_extent=config.carrier.L_max,
        upstream_valid=commit.valid & previous_check.valid,
        upstream_error=jnp.where(commit.valid, previous_check.error_code, commit.error_code),
    )
    execute = predictor_valid & route.routing.executed

    def replay(_):
        _, (hh, _, _, logits) = apply_observation(
            params, h, prompt, route.routing.structure, ops, config=config
        )
        return hh, logits

    def skip(_):
        return jnp.zeros_like(h), jnp.zeros((*h.shape[:2], config.carrier.L_max, 256), h.dtype)

    hh, logits = jax.lax.cond(execute, replay, skip, None)
    return Forward(
        hh,
        logits,
        pb,
        pl,
        commit,
        reachability(route.routing, config=config),
        route.capacity_cut,
        execute,
        route.routing.metadata.carrier_to_m,
        route.routing.metadata.carrier_to_k,
    )


def aligned_metrics(logits, ell, target_slots, target_mask, q):
    """Global emitted-byte alignment; labels/masks used strictly after inference."""
    slots = logits.shape[2]
    generated_mask = jnp.arange(slots)[None, None, :] < ell[:, :, None]
    compact = _compact_slots(logits, generated_mask)
    truth = _compact_slots(target_slots, target_mask).astype(jnp.int32)
    target_lengths = target_mask.sum(2)
    nt, ng = target_lengths.sum(1), ell.sum(1)
    positions = jnp.arange(truth.shape[1])[None, :]
    overlap = positions < jnp.minimum(nt, ng)[:, None]
    costs = ce(compact, truth)
    correct = (jnp.argmax(compact, -1) == truth) & overlap
    starts = jnp.cumsum(target_lengths, 1) - target_lengths
    start = starts[jnp.arange(len(q)), q]
    query_mask = (positions > start[:, None]) & (positions < start[:, None] + slots)
    query_present = (query_mask & overlap).sum(1)
    query_cost = jnp.where(query_mask & overlap, costs, 0).sum(1)
    query_missing = (slots - 1) - query_present
    nonquery = (positions < nt[:, None]) & ~(
        (positions >= start[:, None]) & (positions < start[:, None] + slots)
    )
    nonquery_count = nonquery.sum(1)
    nonquery_present = (nonquery & overlap).sum(1)
    extra = jnp.maximum(ng - nt, 0)
    nonquery_cost = jnp.where(nonquery & overlap, costs, 0).sum(1)
    byte = generated_byte_metrics(logits, ell, target_slots, target_mask)
    return {
        "query_loss": (query_cost + jnp.log(256.0) * query_missing) / (slots - 1),
        "query_nll": query_cost / jnp.maximum(query_present, 1),
        "query_nll_support": query_present,
        "query_accuracy": (correct & query_mask).sum(1) / (slots - 1),
        "query_exact": ((correct & query_mask).sum(1) == slots - 1),
        "missing_query": query_missing,
        "nonquery_loss": (
            nonquery_cost + jnp.log(256.0) * (nonquery_count - nonquery_present + extra)
        )
        / jnp.maximum(nonquery_count, 1),
        "nonquery_nll": nonquery_cost / jnp.maximum(nonquery_present, 1),
        "nonquery_nll_support": nonquery_present,
        "nonquery_accuracy": (correct & nonquery).sum(1) / jnp.maximum(nonquery_count + extra, 1),
        "byte_loss": byte.loss_per_example,
        "byte_nll": byte.overlap_nll_per_example,
        "byte_nll_support": byte.overlap_count,
        "byte_accuracy": byte.accuracy_per_example,
        "byte_exact": byte.exact_per_example,
        "missing": byte.missing_count,
        "extra": byte.extra_count,
        "emitted_length": ng,
        "true_length": nt,
    }


def objective(
    params, batch, occurrence, structure_key, operator_key, *, policy, config, detach_score=False
):
    truth = CommittedStructure(batch["boundary"], batch["length"] > 0, batch["length"])
    out = inference(
        params,
        batch["source_h"],
        batch["prompt"],
        batch["global_example_id"],
        structure_key,
        operator_key,
        occurrence,
        policy=policy,
        config=config,
        teacher=truth if policy == "A" else None,
    )

    def compute(_):
        ell = out.commit.committed.length
        assert ell is not None
        b = out.commit.committed.c_b
        metrics = aligned_metrics(out.logits, ell, batch["slots"], batch["slot_mask"], batch["q"])
        bce = ce(out.b_logits[:, :-1], batch["boundary"][:, :-1]).mean(1)
        lce = ce(out.l_logits, batch["length"]).mean(1)
        mse = ((out.h_hat - jax.lax.stop_gradient(batch["target_h"])) ** 2).mean((1, 2))
        descendant = 0.5 * mse + metrics["byte_loss"] + metrics["query_loss"]
        corrected = (
            score_corrected_structure_loss(descendant, out.commit)
            if policy == "C" and not detach_score
            else descendant
        )
        loss = (bce + lce + corrected).mean()
        active = batch["length"][:, :-1] > 0
        labels = batch["boundary"][:, :-1]
        chosen = b[:, :-1]
        tp = ((chosen == 1) & (labels == 1) & active).sum(1)
        fp = ((chosen == 1) & (labels == 0) & active).sum(1)
        fn = ((chosen == 0) & (labels == 1) & active).sum(1)
        lp = out.commit.sampled.descendant_log_prob
        predicted_b = jnp.argmax(out.b_logits, -1).at[:, -1].set(1)
        predicted_ell = jnp.argmax(out.l_logits, -1)
        classes = jnp.arange(config.carrier.L_max + 1)
        support = batch["length"][:, :, None] == classes
        partial = (batch["length"] > 0) & (batch["length"] < config.carrier.L_max)

        def entropy(x):
            return -jnp.sum(jax.nn.softmax(x) * jax.nn.log_softmax(x), -1)

        metrics = {
            **metrics,
            "b_ce": bce,
            "ell_ce": lce,
            "h_mse": mse,
            "descendant": descendant,
            "loss_per_example": bce + lce + descendant,
            "b_accuracy": ((chosen == labels) & active).sum(1) / jnp.maximum(active.sum(1), 1),
            "b_support": active.sum(1),
            "b_tp": tp,
            "b_fp": fp,
            "b_fn": fn,
            "extent_accuracy": (ell == batch["length"]).mean(1),
            "extent_class_support": support.sum(1),
            "extent_class_correct": (support & (ell == batch["length"])[:, :, None]).sum(1),
            "partial_support": partial.sum(1),
            "partial_correct": ((ell == batch["length"]) & partial).sum(1),
            "prediction_b_accuracy": ((predicted_b[:, :-1] == labels) & active).sum(1)
            / jnp.maximum(active.sum(1), 1),
            "prediction_extent_accuracy": (predicted_ell == batch["length"]).mean(1),
            "prediction_extent_exact": jnp.all(predicted_ell == batch["length"], 1),
            "extent_exact": jnp.all(ell == batch["length"], 1),
            "length_accuracy": ell.sum(1) == batch["length"].sum(1),
            "structure_exact": jnp.all(ell == batch["length"], 1)
            & jnp.all(b == batch["boundary"], 1),
            "peer_reach": out.reach[jnp.arange(len(ell)), batch["q"], batch["peer"]]
            & (ell[jnp.arange(len(ell)), batch["peer"]] > 0),
            "entropy": entropy(out.b_logits[:, :-1]).sum(1) + entropy(out.l_logits).sum(1),
            "log_prob": lp,
            "valid": out.commit.valid,
            "b": b,
            "ell": ell,
            "capacity_cut": out.capacity_cut,
            "pack_m": out.carrier_to_m,
            "pack_k": out.carrier_to_k,
            "prediction_b": predicted_b,
            "prediction_ell": predicted_ell,
            "commit_change": out.commit.changed.mean(1),
            "executed": out.executed,
        }
        return loss, metrics

    template = jax.eval_shape(compute, None)
    return jax.lax.cond(
        out.executed,
        compute,
        lambda _: jax.tree.map(lambda x: jnp.zeros(x.shape, x.dtype), template),
        None,
    )


def train_step(params, moments, step, batch, structure_key, operator_key, *, policy, config):
    (loss, metrics), grads = jax.value_and_grad(
        lambda p: objective(
            p, batch, step, structure_key, operator_key, policy=policy, config=config
        ),
        has_aux=True,
    )(params)
    mask = stochastic_model_update_mask(params)
    finite = jnp.isfinite(loss) & jnp.all(
        jnp.stack([jnp.all(jnp.isfinite(x)) for x in jax.tree.leaves(grads)])
    )
    p, m, s, norm = fail_closed_adamw_step(
        params, grads, moments, step, metrics["executed"] & finite, mask
    )
    return (
        p,
        m,
        s,
        {
            **metrics,
            "loss": loss,
            "raw_gradient_norm": norm,
            "clipped_gradient_norm": jnp.minimum(norm, 1.0),
            "executed": metrics["executed"] & finite & (s == step + 1),
        },
    )
