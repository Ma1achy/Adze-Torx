"""G.0.1 length probes and faithful codec readout/representation remediation.

Probe MLPs are diagnostic instruments only. They are never installed in Adze.
The candidate keeps the original target encoder, linear structural heads and
Mamba decoder. Semantic sampling and capacity routing are untouched.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from .config import REFERENCE_SMALL_V0 as CONFIG
from .corruption import corrupt_h
from .decoder import apply_decoder
from .model import apply_target_codec
from .objectives import adamw_step, cross_entropy
from .phase_g import generated_structure_truth


VERSION = "PHASE_G_0_1_LENGTH_RECOVERY_V1"
LEVELS = (0.025, 0.10, 0.25)


def balanced_extent_ce(logits: jax.Array, labels: jax.Array) -> jax.Array:
    """Equal mass per present extent class, not dominated by inactive sites."""
    lp = jax.nn.log_softmax(logits, -1)
    ce = -jnp.take_along_axis(lp, labels[..., None], -1)[..., 0]
    one_hot = jax.nn.one_hot(labels, logits.shape[-1])
    counts = one_hot.sum((0, 1))
    sums = (one_hot * ce[..., None]).sum((0, 1))
    return jnp.sum(jnp.where(counts > 0, sums / jnp.maximum(counts, 1), 0)) / jnp.maximum(
        jnp.sum(counts > 0), 1
    )


def length_labels(extent: jax.Array) -> jax.Array:
    total = extent.sum(1)
    emitting = (extent > 0).sum(1)
    site = jnp.max(jnp.where(extent > 0, jnp.arange(extent.shape[1])[None], 0), 1)
    last = extent[jnp.arange(len(extent)), site]
    partial = jnp.where((last > 0) & (last < 4), last, 0)
    return jnp.stack((total - 8, emitting, partial), -1).astype(jnp.int32)


def initialize_probe(key: jax.Array, local_width: int, global_width: int, mlp: bool) -> Any:
    keys = iter(jax.random.split(key, 4))

    def layer(i, o):
        return {"w": jax.random.normal(next(keys), (i, o)) * 0.02, "b": jnp.zeros((o,))}

    if mlp:
        return {
            "local_hidden": layer(local_width + 32, 64),
            "local": layer(64, 5),
            "global_hidden": layer(global_width, 64),
            "global": layer(64, 78),
        }
    return {"local": layer(local_width, 5), "global": layer(global_width, 78)}


def probe_apply(params: Any, local: jax.Array, global_x: jax.Array, mlp: bool):
    def affine(x, p):
        return x @ p["w"] + p["b"]

    if mlp:
        position = jnp.broadcast_to(jnp.eye(32)[None], (*local.shape[:2], 32))
        local = jax.nn.gelu(affine(jnp.concatenate((local, position), -1), params["local_hidden"]))
        global_x = jax.nn.gelu(affine(global_x, params["global_hidden"]))
    return affine(local, params["local"]), affine(global_x, params["global"])


def probe_step(params, moments, step, local, global_x, extents, mlp):
    labels = length_labels(extents)

    def objective(p):
        ell, glob = probe_apply(p, local, global_x, mlp)
        ce = balanced_extent_ce(ell, extents)
        for start, end, column in ((0, 57, 0), (57, 74, 1), (74, 78, 2)):
            ce += cross_entropy(
                glob[:, start:end], labels[:, column], jnp.ones((len(labels),), bool)
            )
        return ce

    loss, grad = jax.value_and_grad(objective)(params)
    updated, moments, norm = adamw_step(
        params, grad, moments, step, learning_rate=1e-3, weight_decay=0.01, clip_norm=1.0
    )
    return updated, moments, loss, norm


def candidate_step(params, moments, step, values, masks, noise_key, readout_only):
    def objective(p):
        output = apply_target_codec(p, values, masks, config=CONFIG)
        target = output["target"]
        teacher = target["teacher"]
        truth = generated_structure_truth(teacher.slot_bytes, teacher.length, K=CONFIG.packing.K)
        noise_key1, noise_key2 = jax.random.split(noise_key)
        nu = jax.random.uniform(noise_key1, (len(values),), minval=0.025, maxval=0.25)
        noisy = corrupt_h(target["h0"], nu, jax.random.normal(noise_key2, target["h0"].shape))
        head = p["encoder"]["target_l"]
        noisy_logits = noisy @ head["weight"] + head["bias"]
        byte = cross_entropy(output["codec_logits"], teacher.slot_bytes, teacher.slot_mask)
        boundary = cross_entropy(
            target["b_logits"][:, :-1], truth.c_b[:, :-1], jnp.ones_like(truth.c_b[:, :-1], bool)
        )
        clean_extent = balanced_extent_ce(target["l_logits"], teacher.length)
        noisy_extent = balanced_extent_ce(noisy_logits, teacher.length)
        return byte + boundary + clean_extent + noisy_extent, (
            byte,
            boundary,
            clean_extent,
            noisy_extent,
        )

    (loss, components), grad = jax.value_and_grad(objective, has_aux=True)(params)
    mask = jax.tree.map(lambda _: jnp.asarray(not readout_only), params)
    if readout_only:
        for name in ("target_b", "target_l"):
            mask["encoder"][name] = jax.tree.map(
                lambda _: jnp.asarray(True), params["encoder"][name]
            )
    finite = jnp.isfinite(loss) & jnp.all(
        jnp.stack([jnp.all(jnp.isfinite(x)) for x in jax.tree.leaves(grad)])
    )

    def update():
        return adamw_step(
            params,
            grad,
            moments,
            step,
            learning_rate=3e-4,
            weight_decay=0.01,
            clip_norm=1.0,
            update_mask=mask,
        )

    updated, moments, norm = jax.lax.cond(
        finite, update, lambda: (params, moments, jnp.asarray(0.0))
    )
    return (
        updated,
        moments,
        {
            "loss": loss,
            "finite": finite & jnp.isfinite(norm),
            "norm": norm,
            "components": jnp.stack(components),
        },
    )


def decode_inferred_extents(h0: jax.Array, head: Any, decoder: Any):
    """Inference accepts h0 and parameters ONLY; no teacher lengths/masks."""
    lengths = jnp.argmax(h0 @ head["weight"] + head["bias"], -1)
    logits, mask = apply_decoder(h0, lengths, decoder, CONFIG)
    return lengths, logits, mask


def extent_report(
    prediction: np.ndarray, truth: np.ndarray, direct: np.ndarray | None = None
) -> dict:
    lengths = truth.sum(1)

    def group(p, t):
        correct = p == t
        partial = (t > 0) & (t < 4)
        zero = t == 0
        confusion = np.zeros((5, 5), np.int64)
        np.add.at(confusion, (t.reshape(-1), p.reshape(-1)), 1)
        recalls = [
            None if confusion[c].sum() == 0 else float(confusion[c, c] / confusion[c].sum())
            for c in range(5)
        ]
        return {
            "examples": len(t),
            "exact_total_length_accuracy": float((p.sum(1) == t.sum(1)).mean()),
            "emitting_carrier_count_accuracy": float(((p > 0).sum(1) == (t > 0).sum(1)).mean()),
            "per_carrier_extent_accuracy": float(correct.mean()),
            "exact_extent_structure_accuracy": float(correct.all(1).mean()),
            "partial_extent_accuracy": None
            if not partial.any()
            else float(correct[partial].mean()),
            "partial_support": int(partial.sum()),
            "zero_detection_recall": None if not zero.any() else float((p[zero] == 0).mean()),
            "nonzero_detection_recall": None if zero.all() else float((p[~zero] > 0).mean()),
            "per_class_recall": recalls,
            "confusion_true_by_pred": confusion.tolist(),
        }

    result = group(prediction, truth)
    result["by_length"] = [
        {"length": n, **group(prediction[lengths == n], truth[lengths == n])} for n in range(8, 65)
    ]
    result["by_partial_class"] = []
    for c in (1, 2, 3):
        selected = lengths % 4 == c
        result["by_partial_class"].append(
            {"class": c, **group(prediction[selected], truth[selected])}
        )
    if direct is not None:
        labels = np.asarray(length_labels(jnp.asarray(truth)))
        result["direct_global_probes"] = {
            "total_length_accuracy": float((direct[:, 0] == labels[:, 0]).mean()),
            "emitting_count_accuracy": float((direct[:, 1] == labels[:, 1]).mean()),
            "partial_class_accuracy": float((direct[:, 2] == labels[:, 2]).mean()),
        }
    return result


def structurally_acceptable(report: dict, *, noisy: bool = False) -> bool:
    required = 0.95 if noisy else 0.99
    return (
        report["exact_total_length_accuracy"] >= required
        and report["exact_extent_structure_accuracy"] >= required
        and report["per_carrier_extent_accuracy"] >= (0.99 if noisy else 0.995)
        and report["partial_extent_accuracy"] >= required
        and min(g["partial_extent_accuracy"] for g in report["by_partial_class"])
        >= (0.95 if noisy else 0.98)
        and report["zero_detection_recall"] >= 0.99
        and report["nonzero_detection_recall"] >= 0.99
        and (noisy or min(g["exact_total_length_accuracy"] for g in report["by_length"]) >= 0.95)
    )
