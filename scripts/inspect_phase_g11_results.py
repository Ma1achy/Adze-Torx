"""Export deterministic routing traces and descriptive statistics; no neural run."""

import json

import jax
import jax.numpy as jnp
import numpy as np

import run_phase_g11_validation as v
from adze_t.state import CommittedStructure
from adze_t.phase_g0 import dynamic_capacity_routing
from adze_t.masking import draft_attention_mask
from adze_t.packing import pack_values
from adze_t.unpool import unpool_values


def main():
    data = v.make_examples(1141, 16, id_offset=10000)
    selected = jax.random.permutation(jax.random.key(1143), 912)[:64]
    d = v.take(data, selected)
    expected = json.loads((v.OUT / "flow_selection.json").read_text())["ids"]
    assert np.array_equal(np.asarray(d["global_example_id"]), expected)
    raw = {k: np.asarray(d[k]) for k in ("q", "peer", "global_example_id", "boundary", "length")}
    rows = {}
    for name, b in (("A", d["boundary"]), ("B", v.cut_after_query(d))):
        route = dynamic_capacity_routing(
            CommittedStructure(b, d["length"] > 0, d["length"]), M_max=32, K=8, max_extent=4
        )
        assert route.routing.executed
        m = route.routing.metadata
        pos = m.carrier_to_m * 8 + m.carrier_to_k
        mask = draft_attention_mask(m.block_id, m.query_mask, m.kv_mask)
        reach = mask[jnp.arange(64)[:, None, None], pos[:, :, None], pos[:, None, :]]
        x = jnp.broadcast_to(jnp.arange(32)[None, :, None], (64, 32, 1))
        exact = bool(jnp.array_equal(unpool_values(pack_values(x, m), m, C=32), x))
        ancestors = jnp.any(reach & (jnp.arange(32)[None] <= d["q"][:, None])[:, :, None], axis=1)
        label_region = jnp.arange(32)[None] >= d["peer"][:, None]
        for k, value in {
            "semantic_b": b,
            "mechanical_b": route.routing.structure.c_b,
            "capacity_cuts": route.capacity_cut,
            "pack": m.packed_to_carrier,
            "unpool_m": m.carrier_to_m,
            "unpool_k": m.carrier_to_k,
            "query_mask": m.query_mask,
            "kv_mask": m.kv_mask,
            "reach": reach,
            "decoder_prefix_ancestors": ancestors,
        }.items():
            raw[name + "_" + k] = np.asarray(value)
        peer = reach[jnp.arange(64), d["q"], d["peer"]]
        for model in ("fresh_F1_G01", "teacher100"):
            flow = json.loads((v.OUT / ("flow_" + model + ".json")).read_text())
            assert np.array_equal(
                np.concatenate([r["peer_reachable"] for r in flow["rows"][name]]), np.asarray(peer)
            )
        rows[name] = {
            "pack_unpool_identity": exact,
            "peer_reachable": int(peer.sum()),
            "forced_capacity_cuts": int(route.capacity_cut.sum()),
            "query_active_sites": int(m.query_mask.sum()),
            "kv_active_sites": int(m.kv_mask.sum()),
            "label_region_reachable_to_decoder_prefix": int(
                jnp.any(ancestors & label_region, axis=1).sum()
            ),
            "states_after_peer_reachable_to_decoder_prefix": int(
                jnp.any(ancestors & (jnp.arange(32)[None] > d["peer"][:, None]), axis=1).sum()
            ),
        }
    path = v.ROOT / "results/runs/phase_g/g1_1/validation_v0/routing_trace.npz"
    if path.exists():
        raise RuntimeError("refusing trace overwrite")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, allow_pickle=False, **raw)
    codec = json.loads((v.OUT / "codec_identifiability.json").read_text())
    buckets = codec["length_buckets"]
    micro = sum(int(k) * x["examples"] * x["accuracy"] for k, x in buckets.items()) / sum(
        int(k) * x["examples"] for k, x in buckets.items()
    )
    # No fitted probe: counts-only population-prior baselines from training labels.
    train = v.make_examples(1140, 32)
    train_y = np.asarray(train["target"]).reshape(-1, 16, 4)[
        np.arange(1824), np.asarray(train["q"]), 1:
    ]
    val_y = np.asarray(data["target"]).reshape(-1, 16, 4)[np.arange(912), np.asarray(data["q"]), 1:]
    majority = np.stack([np.bincount(train_y[:, k], minlength=33).argmax() for k in range(3)])
    prior_acc = float((val_y == majority).mean())
    positional = np.zeros_like(val_y)
    for q in range(15):
        subset = train_y[np.asarray(train["q"]) == q]
        pred = (
            np.stack([np.bincount(subset[:, k], minlength=33).argmax() for k in range(3)])
            if len(subset)
            else majority
        )
        positional[np.asarray(data["q"]) == q] = pred
    v.dump(
        v.OUT / "routing_trace.json",
        {
            "postprocessing_only": True,
            "source_sha256": v.sha(__file__),
            "rows": rows,
            "query_ids": d["q"],
            "peer_ids": d["peer"],
            "global_ids": d["global_example_id"],
            "gaps": d["peer"] - d["q"],
            "raw_path": str(path.relative_to(v.ROOT)),
            "raw_sha256": v.sha(path),
            "codec_micro_byte_accuracy": micro,
            "majority_class_validation_payload_accuracy": prior_acc,
            "position_prior_validation_payload_accuracy": float((positional == val_y).mean()),
            "population_chance": 1 / 28,
            "conditional_uniform_NLL": float(np.log(28)),
            "exact_three_byte_chance": 28**-3,
            "note": "Counts-only prior diagnostics; no neural fitting/model execution or changed task gates",
        },
    )


if __name__ == "__main__":
    main()
