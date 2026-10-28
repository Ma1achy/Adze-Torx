"""Post-training host/device boundary fix; frozen teacher100 source stays intact.

The first endpoint attempt failed on eager indexing of pickled NumPy parameters.
This wrapper converts their container to JAX without changing a single value.
It verifies the original execution binding rather than creating a new training
protocol. It never invokes an optimizer or modifies a checkpoint.
"""

import hashlib
import importlib.metadata
import json

import jax
import jax.numpy as jnp

import run_phase_g11_teacher as frozen
from adze_t.phase_g11_records import Journal, sha


def main():
    binding = json.loads((frozen.OUT / "protocol_binding.json").read_text())
    identity = binding["sha256"]
    protocol = binding["protocol"]
    assert hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest() == identity
    for name, expected in protocol["source"].items():
        path = (
            frozen.OUT / "PROSPECTIVE_CLARIFICATION.md"
            if name == "clarification"
            else frozen.ROOT / name
        )
        if sha(path) != expected:
            raise RuntimeError(f"frozen execution source mismatch:{name}")
    for path, expected in (
        (frozen.v.F1, protocol["f1_sha256"]),
        (frozen.v.CODEC, protocol["codec_sha256"]),
    ):
        if sha(path) != expected:
            raise RuntimeError("parent/codec mismatch")
    for package, version in protocol["runtime"]["packages"].items():
        if importlib.metadata.version(package) != version:
            raise RuntimeError("runtime changed")
    if (
        jax.default_backend() != protocol["runtime"]["backend"]
        or bool(jax.config.read("jax_enable_x64")) != protocol["runtime"]["x64"]
    ):
        raise RuntimeError("backend/precision changed")
    direct = json.loads(
        importlib.metadata.distribution("extro-torx").read_text("direct_url.json") or "{}"
    )
    if direct.get("vcs_info", {}).get("commit_id") != "f1fc858ed950ecd41935d15c06d0ec7c5e0674ae":
        raise RuntimeError("Torx revision changed")
    baseline = json.loads(
        (
            frozen.ROOT / "results/phase_g/preflight_g01_v1/f571ab40659d/historical_baseline.json"
        ).read_text()
    )["files"]
    if not all(sha(p) == h for p, h in baseline.items()):
        raise RuntimeError("historical artifact changed")
    journal = Journal(frozen.RUN / "official", identity)
    state = journal.recover()
    if state is None or state["step"] != 100:
        raise RuntimeError("exact endpoint missing")
    before = frozen.tree_hash(state["params"])
    state["params"] = jax.tree.map(jnp.asarray, state["params"])
    if frozen.tree_hash(state["params"]) != before:
        raise RuntimeError("conversion changed parameter bits")
    codec = frozen.load_pickle(frozen.v.CODEC)
    _, validation = frozen.prepare(codec)
    frozen.dump(
        "endpoint_array_conversion.json",
        {
            "protocol": identity,
            "checkpoint_model_tree": before,
            "conversion": "NumPy to JAX; shape/dtype/bytes identical",
            "driver_sha256": sha(__file__),
            "original_source_unchanged": True,
            "Torx_direct_url": direct,
            "additional_updates": 0,
        },
    )
    frozen.endpoint(state, codec, identity, validation)


if __name__ == "__main__":
    main()
