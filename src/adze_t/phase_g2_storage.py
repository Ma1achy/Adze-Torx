"""Bounded-space atomic resume journal for the prospectively declared G2 run."""

import json
from pathlib import Path
import pickle
import os
import hashlib

import jax
import jax.numpy as jnp
import numpy as np

from .phase_g11_records import sha, write_json


class RollingJournal:
    def __init__(self, root, identity):
        self.root = Path(root)
        self.identity = identity
        self.tip = None

    def recover(self):
        parent = None
        paths = sorted((self.root / "receipts").glob("*.json"))
        for n, path in enumerate(paths):
            record = json.loads(path.read_text())
            content = {k: v for k, v in record.items() if k != "receipt_digest"}
            if hashlib.sha256(
                json.dumps(content, sort_keys=True).encode()
            ).hexdigest() != record.get("receipt_digest"):
                raise ValueError("receipt content checksum mismatch")
            if (
                path.stem != f"{n:05d}"
                or record["step"] != n
                or record["identity"] != self.identity
                or record["parent"] != parent
                or record["completed"] is not True
            ):
                raise ValueError("corrupt/mixed/noncontiguous receipt chain")
            parent = record["state_sha256"]
        if not paths:
            self._preserve_uncommitted_slots(-1)
            self._validate_snapshots(-1)
            return None
        self.tip = json.loads(paths[-1].read_text())
        path = self.root / self.tip["slot"]
        if sha(path) != self.tip["state_sha256"]:
            raise ValueError("durable tip payload mismatch")
        with path.open("rb") as stream:
            state = pickle.load(stream)
        if state["step"] != self.tip["step"] or state["identity"] != self.identity:
            raise ValueError("wrong durable state identity")
        if not all(
            np.isfinite(x).all() for x in jax.tree.leaves((state["params"], state["moments"]))
        ):
            raise ValueError("nonfinite saved state")
        write_json(self.root / "LATEST.json", self.tip, replace=True)
        self._preserve_uncommitted_slots(self.tip["step"])
        self._validate_snapshots(self.tip["step"])
        return jax.tree.map(lambda x: jnp.asarray(x) if isinstance(x, np.ndarray) else x, state)

    def _preserve_uncommitted_slots(self, tip_step):
        for path in self.root.glob("slot_*.pkl"):
            if self.tip is not None and path.name == self.tip["slot"]:
                continue
            with path.open("rb") as stream:
                candidate = pickle.load(stream)
            if candidate["identity"] != self.identity:
                raise ValueError("uncommitted slot has incompatible provenance")
            if candidate["step"] > tip_step:
                orphan = (
                    self.root
                    / f"orphan_{candidate['step']:05d}_{len(list(self.root.glob('orphan_*'))):03d}.pkl"
                )
                path.rename(orphan)

    def _validate_snapshots(self, tip_step):
        for path in (self.root / "snapshots").glob("update_*.pkl"):
            step = int(path.stem.split("_")[-1])
            if step > tip_step:
                path.rename(
                    self.root
                    / f"orphan_snapshot_{step:05d}_{len(list(self.root.glob('orphan_*'))):03d}.pkl"
                )
                continue
            receipt = json.loads((self.root / "receipts" / f"{step:05d}.json").read_text())
            if sha(path) != receipt["state_sha256"]:
                raise ValueError("immutable milestone snapshot corruption")

    def append(self, state, metrics, *, snapshot=False):
        step = state["step"]
        expected = 0 if self.tip is None else self.tip["step"] + 1
        if step != expected or state["identity"] != self.identity:
            raise ValueError("not the next matching completed update")
        if step > 0 and not bool(metrics.get("executed", False)):
            raise ValueError("nonexecuted update cannot get a completed receipt")
        if "executed" in metrics and not bool(metrics["executed"]):
            raise ValueError("failed attempt cannot complete")
        if not all(
            np.isfinite(x).all()
            for x in jax.tree.leaves((state["params"], state["moments"], metrics))
            if x is not None and not isinstance(x, str)
        ):
            raise ValueError("failed update cannot persist as completed")
        self.root.mkdir(parents=True, exist_ok=True)
        slot = f"slot_{step % 2}.pkl"
        path = self.root / slot
        # Never overwrite the committed tip slot. The old alternate is transient.
        if self.tip is not None and self.tip["slot"] == slot:
            raise ValueError("attempt to overwrite committed tip")
        tmp = path.with_suffix(".tmp")
        if tmp.exists():
            orphan = (
                self.root / f"orphan_{step:05d}_{len(list(self.root.glob('orphan_*'))):03d}.pkl"
            )
            tmp.rename(orphan)
        with tmp.open("wb") as stream:
            pickle.dump(jax.device_get(state), stream, protocol=pickle.HIGHEST_PROTOCOL)
            stream.flush()
            os.fsync(stream.fileno())
        digest = sha(tmp)
        tmp.replace(path)
        if snapshot:
            snap = self.root / "snapshots" / f"update_{step:05d}.pkl"
            snap.parent.mkdir(parents=True, exist_ok=True)
            if snap.exists():
                if sha(snap) != digest:
                    raise ValueError("snapshot mismatch; refuse overwrite")
            else:
                # A hardlink remains immutable across atomic replacement of slots.
                os.link(path, snap)
        receipt = {
            "step": step,
            "identity": self.identity,
            "slot": slot,
            "state_sha256": digest,
            "parent": None if self.tip is None else self.tip["state_sha256"],
            "metrics": metrics,
            "completed": True,
            "snapshot": snapshot,
        }
        receipt["receipt_digest"] = hashlib.sha256(
            json.dumps(receipt, sort_keys=True).encode()
        ).hexdigest()
        write_json(self.root / "receipts" / f"{step:05d}.json", receipt)
        self.tip = receipt
        write_json(self.root / "LATEST.json", receipt, replace=True)
        return receipt
