"""Exploratory isolation of attention masks from committed positional codes."""

import json
import pickle

import jax
import jax.numpy as jnp
import numpy as np

import run_phase_g1 as g


class GeometryOps(g.DiagnosticOps):
    def __init__(self, variant="full"):
        self.zero_geometry = variant in ("zero_geometry", "zero_geometry_noattention")
        super().__init__("no_attention" if variant == "zero_geometry_noattention" else variant)

    def embedding(self, indices, params, *, name):
        result = super().embedding(indices, params, name=name)
        if self.zero_geometry and name in ("dit.block_embed", "dit.within_embed"):
            result = jnp.zeros_like(result)
        return result


def main():
    before = g.sha(g.CHECKPOINT)
    original = json.loads((g.OUT / "diagnosis.json").read_text())
    assert before == original["checkpoint_sha256"]
    with g.CHECKPOINT.open("rb") as stream:
        params = jax.tree.map(
            jnp.asarray, g.torx_means_to_deterministic(pickle.load(stream)["params"])
        )
    with g.CODEC.open("rb") as stream:
        codec = jax.tree.map(jnp.asarray, pickle.load(stream))
    assert g.sha(g.CODEC) == original["codec_sha256"]
    # Diagnostic interpreter-local backend selection; production source is unchanged.
    g.DiagnosticOps = GeometryOps
    run = jax.jit(g.forward, static_argnames=("variant", "torx"))
    teacher = jax.jit(lambda t, m: g.apply_clean_target_teacher(codec, t, m, config=g.CFG))
    data = g.with_nu(g.dataset(256, 951, global_id_offset=1024), 0)
    structures = g._control_structures(data)
    rows = {}
    maximum = 0.0
    for start in range(0, 256, 8):
        batch = g.take_batch(data, jnp.arange(start, start + 8))
        clean = teacher(batch["target"], batch["target_mask"])
        h, _ = g.make_initial_corruption(
            clean["target"]["h0"],
            batch["nu"],
            g.stochastic_root("diffusion"),
            batch["global_example_id"],
            optimizer_step=0,
        )
        ell = clean["target"]["teacher"].length
        slots, mask = clean["target"]["teacher"].slot_bytes, clean["target"]["teacher"].slot_mask
        for variant in ("zero_geometry", "zero_geometry_noattention"):
            outputs = {}
            for route in ("correct", "shifted_cuts"):
                value = run(
                    params,
                    h,
                    batch["nu"],
                    structures[route].c_b[start : start + 8],
                    ell,
                    variant=variant,
                )
                assert bool(value["valid"])
                outputs[route] = value["logits"]
                m = g.generated_byte_metrics(value["logits"], ell, slots, mask)
                rows.setdefault(variant + "_" + route, []).append(
                    {
                        "accuracy": np.asarray(m.accuracy_per_example),
                        "nll": np.asarray(m.overlap_nll_per_example),
                        "exact": np.asarray(m.exact_per_example),
                    }
                )
            if variant == "zero_geometry_noattention":
                maximum = max(
                    maximum, float(jnp.max(jnp.abs(outputs["correct"] - outputs["shifted_cuts"])))
                )
    summary = {}
    for name, values in rows.items():
        joined = {k: np.concatenate([v[k] for v in values]) for k in values[0]}
        summary[name] = {
            "means": {k: float(v.mean()) for k, v in joined.items()},
            "per_example": joined,
        }
    assert g.sha(g.CHECKPOINT) == before and g.sha(g.CODEC) == original["codec_sha256"]
    g.dump(
        g.OUT / "geometry.json",
        {
            "source_sha256": g.sha(__file__),
            "protocol_sha256": g.sha(g.OUT / "GEOMETRY_FOLLOWUP.md"),
            "exploratory": True,
            "rows": summary,
            "both_zero_correct_shift_max_logit_error": maximum,
            "identity_check_passed": maximum <= 1e-5,
            "checkpoint_unchanged": True,
        },
    )
    if maximum > 1e-5:
        raise RuntimeError("unexpected residual geometry dependence")


if __name__ == "__main__":
    main()
