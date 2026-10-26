"""Supplement G.1 with saved-array analysis and frozen encoder-only probes."""

import json
import pickle

import jax
import jax.numpy as jnp
import numpy as np

import run_phase_g1 as g


def ce(logits, labels):
    x = logits - np.max(logits, -1, keepdims=True)
    return np.log(np.exp(x).sum(-1)) - np.take_along_axis(x, labels[..., None], -1)[..., 0]


def main():
    result = json.loads((g.OUT / "diagnosis.json").read_text())
    assert g.sha(g.RAW) == result["raw_sha256"]
    assert g.sha(g.CODEC) == result["codec_sha256"]
    raw = np.load(g.RAW, allow_pickle=False)
    data, q, _, _ = g.fixtures()
    index = np.arange(16)
    labels = [np.asarray(d["target"]).reshape(16, 16, 4)[index, np.asarray(q)] for d in data]
    local = {}
    for route in ("correct", "excluded_peer"):
        a, b = (raw[f"fixture_{route}_target_logits_{s}"] for s in ("A", "B"))
        local[route] = {
            "own_label_nll_per_pair": 0.5 * (ce(a, labels[0]).mean(1) + ce(b, labels[1]).mean(1)),
            "counterfactual_matching_gain_per_pair": 0.5
            * (
                ce(a, labels[1]).mean(1)
                + ce(b, labels[0]).mean(1)
                - ce(a, labels[0]).mean(1)
                - ce(b, labels[1]).mean(1)
            ),
        }
    advantage = (
        local["excluded_peer"]["own_label_nll_per_pair"]
        - local["correct"]["own_label_nll_per_pair"]
    )
    with g.CODEC.open("rb") as stream:
        codec = jax.tree.map(jnp.asarray, pickle.load(stream))
    encode = jax.jit(lambda t, m: g.apply_clean_target_teacher(codec, t, m, config=g.CFG))
    isolated_decoder = jax.jit(g.decode_local)
    representations, peer_scores = [], []
    for d, target in zip(data, labels, strict=True):
        encoded = encode(d["target"], d["target_mask"])
        h, ell = encoded["target"]["h0"], encoded["target"]["teacher"].length
        representations.append(np.asarray(h))
        logits = np.asarray(isolated_decoder(h, ell, codec["decoder"]))[index, np.asarray(q) + 1]
        peer_scores.append(
            {
                "accuracy": float((logits.argmax(-1) == target).mean()),
                "nll": float(ce(logits, target).mean()),
            }
        )
    delta = representations[0] - representations[1]
    outside = np.arange(32)[None, :] > np.asarray(q)[:, None] + 1
    rms = np.sqrt(np.mean(delta**2, -1))
    original = g.dataset(256, 951, global_id_offset=1024)
    target = np.pad(np.asarray(original["target"]), ((0, 0), (0, 64))).reshape(256, 32, 4)
    ell = raw["length"]
    active = ell > 0
    boundaries = raw["semantic_correct"]
    blocks = np.cumsum(np.pad(boundaries[:, :-1], ((0, 0), (1, 0))), axis=1)
    signatures = target[:, :, 0] <= 16
    outside_matching = (
        (signatures[:, :, None] == signatures[:, None, :])
        & (blocks[:, :, None] != blocks[:, None, :])
        & active[:, :, None]
        & active[:, None, :]
    )
    exact_payload_match = (
        np.all(target[:, :, None, :] == target[:, None, :, :], -1)
        & (ell[:, :, None] == 4)
        & (ell[:, None, :] == 4)
        & ~np.eye(32, dtype=bool)[None]
    )
    partitions = {}
    identity = {}
    for route in ("correct", "shifted_cuts"):
        pack = raw[route + "_pack"]
        flat = pack.reshape(256, -1)
        inverse = raw[route + "_unpool_m"] * 8 + raw[route + "_unpool_k"]
        identity[route] = bool(np.all(np.take_along_axis(flat, inverse, axis=1) == np.arange(32)))
        partitions[route] = [
            [row[row >= 0].tolist() for row in pack[i] if (row >= 0).any()] for i in range(8)
        ]
    metrics = {}
    for name in ("unpooled_rms_difference", "h_hat_rms_difference", "logits_rms_difference"):
        metrics[name] = {
            k: float(raw[name][selection].mean()) if selection.any() else None
            for k, selection in {
                "emitting": active,
                "inactive": ~active,
                "lost_future_emitting": (raw["lost_later_peers"] > 0) & active,
            }.items()
        }
    value = {
        "diagnostic_only": True,
        "source_sha256": g.sha(__file__),
        "raw_sha256": g.sha(g.RAW),
        "counterfactual_local": local,
        "paired_nll_advantage": {
            "values": advantage,
            "mean": float(advantage.mean()),
            "sd": float(advantage.std(ddof=1)),
            "positive_pairs": int((advantage > 0).sum()),
            "pairs": 16,
        },
        "isolated_clean_peer_payload_readout": peer_scores,
        "encoder_outside_query_block_rms_by_site": rms,
        "encoder_outside_query_block_changed_sites": int(((rms > 1e-6) & outside).sum()),
        "encoder_outside_query_block_max_rms": float(rms[outside].max()),
        "original_emitting_sites": int(active.sum()),
        "original_sites_with_same_binary_signature_in_other_block": int(
            outside_matching.any(-1).sum()
        ),
        "original_sites_with_identical_full_payload_elsewhere": int(
            exact_payload_match.any(-1).sum()
        ),
        "identity_roundtrip_both_routes": identity,
        "first8_partitions": partitions,
        "identity_aligned_numerical_changes": metrics,
        "original_label_note": "binary first-byte boundary signature is not an independent repeated block payload",
    }
    g.dump(g.OUT / "supplement.json", value)


if __name__ == "__main__":
    main()
