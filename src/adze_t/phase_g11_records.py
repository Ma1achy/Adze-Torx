"""Strict bounded-run journal and prospectively fixed paired decision statistics."""

from pathlib import Path
import hashlib
import json
import pickle

import jax
import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value, *, replace=False):
    path = Path(path)
    if path.exists() and not replace:
        if json.loads(path.read_text()) != json.loads(json.dumps(value)):
            raise ValueError(f"immutable record differs: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        import os

        os.fsync(stream.fileno())
    tmp.replace(path)


def finite(tree):
    return all(np.isfinite(np.asarray(x)).all() for x in jax.tree.leaves(tree))


class Journal:
    """Only a hash-verified contiguous receipt chain counts completed updates."""

    def __init__(self, root, protocol):
        self.root = Path(root)
        self.protocol = protocol

    def recover(self):
        receipts = sorted((self.root / "receipts").glob("*.json"))
        state = None
        parent = None
        for step, path in enumerate(receipts):
            r = json.loads(path.read_text())
            if path.stem != f"{step:03d}" or r["step"] != step or r["completed"] is not True:
                raise ValueError("noncontiguous completed journal")
            if r["protocol"] != self.protocol or r["parent"] != parent:
                raise ValueError("protocol/parent mismatch")
            payload = self.root / r["payload"]
            if sha(payload) != r["sha256"]:
                raise ValueError("state hash mismatch")
            with payload.open("rb") as stream:
                state = pickle.load(stream)
            if state["step"] != step or state["protocol"] != self.protocol:
                raise ValueError("state identity mismatch")
            if not finite((state["params"], state["moments"])):
                raise ValueError("nonfinite saved state")
            parent = r["sha256"]
        if state is not None:
            write_json(
                self.root / "LATEST.json", {"step": state["step"], "sha256": parent}, replace=True
            )
        return state

    def append(self, state, metrics):
        prior = self.recover()
        step = 0 if prior is None else prior["step"] + 1
        if state["step"] != step or state["protocol"] != self.protocol:
            raise ValueError("invalid attempted update identity")
        if not finite((state["params"], state["moments"], metrics)):
            raise ValueError("invalid attempt cannot complete")
        parent = (
            None if prior is None else json.loads((self.root / "LATEST.json").read_text())["sha256"]
        )
        directory = self.root / "payloads"
        directory.mkdir(parents=True, exist_ok=True)
        attempt = len(list(directory.glob(f"{step:03d}_*.pkl")))
        path = directory / f"{step:03d}_{attempt:03d}.pkl"
        # Exclusive, immutable payload; interruption leaves an uncounted orphan.
        with path.open("xb") as stream:
            pickle.dump(jax.device_get(state), stream, protocol=pickle.HIGHEST_PROTOCOL)
            stream.flush()
            import os

            os.fsync(stream.fileno())
        record = {
            "step": step,
            "protocol": self.protocol,
            "parent": parent,
            "payload": str(path.relative_to(self.root)),
            "sha256": sha(path),
            "metrics": metrics,
            "completed": True,
        }
        write_json(self.root / "receipts" / f"{step:03d}.json", record)
        write_json(
            self.root / "LATEST.json", {"step": step, "sha256": record["sha256"]}, replace=True
        )
        return record


def paired_bootstrap(a, b, lengths, *, seed, code=0):
    """2000 whole-example pairs within 57 length strata; no byte resampling."""
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape or a.ndim != 1 or len(a) != len(lengths):
        raise ValueError("unpaired observations")
    groups = [np.flatnonzero(np.asarray(lengths) == n) for n in range(8, 65)]
    if any(len(g) != 16 for g in groups):
        raise ValueError("frozen balanced validation strata required")
    key = jax.random.fold_in(jax.random.key(seed), code)
    keys = jax.random.split(key, len(groups))
    indices = np.concatenate(
        [
            g[np.asarray(jax.random.randint(k, (2000, 16), 0, 16))]
            for g, k in zip(groups, keys, strict=True)
        ],
        axis=1,
    )
    delta = (a[indices] - b[indices]).mean(1)
    denominators = b[indices].mean(1)
    # Accuracy controls can correctly have zero hits. Their absolute paired
    # effect remains defined; a relative effect with zero support does not.
    ratio = a[indices].mean(1) / denominators - 1 if np.all(denominators > 0) else None
    return {
        "delta": float((a - b).mean()),
        "delta_ci95": np.percentile(delta, [2.5, 97.5]).tolist(),
        "relative": float(a.mean() / b.mean() - 1) if b.mean() > 0 else None,
        "relative_ci95": np.percentile(ratio, [2.5, 97.5]).tolist() if ratio is not None else None,
        "replicates": 2000,
        "seed": seed,
        "control_code": code,
    }


def nonquery_guard(current, baseline):
    accuracy = current["nonquery_accuracy"] >= baseline["nonquery_accuracy"] - 0.02
    nll = current["nonquery_nll"] <= 1.05 * baseline["nonquery_nll"]
    return {
        "accuracy_pass": bool(accuracy),
        "nll_pass": bool(nll),
        "passed": bool(accuracy and nll),
    }
