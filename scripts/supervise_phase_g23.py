"""Detached direct-file G job supervision; no model imports or scientific edits."""

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

from collect_phase_g23_records import ROOT, sha, write_identical


def command(stage):
    if stage == "validation16":
        script, args = "complete_phase_g23_validation.py", []
    elif stage in ("campaign", "science", "close"):
        script, args = "run_phase_g23_v1.py", ["--stage", stage]
    else:
        raise ValueError("unsupported scientific stage")
    return ["nice", "-n", "10", sys.executable, str(ROOT / "scripts" / script), *args]


def protocol_identity():
    path = ROOT / "results/phase_g/g2/v1/G2_G3_PROTOCOL_IDENTITY.json"
    record = json.loads(path.read_text())
    binding = record["binding"]
    if (
        hashlib.sha256(json.dumps(binding, sort_keys=True).encode()).hexdigest()
        != record["identity"]
    ):
        raise ValueError("protocol binding checksum mismatch")
    for name, digest in {**binding["source"], **binding["protocols"]}.items():
        if sha(ROOT / name) != digest:
            raise ValueError("frozen source/protocol mismatch:" + name)
    return record["identity"]


def run(stage, attempt):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", attempt):
        raise ValueError("attempt must be a unique simple project-local name")
    argv = command(stage)
    identity = protocol_identity()
    base = ROOT / "results/runs/phase_g/g23/v1/process_supervision"
    base.mkdir(parents=True, exist_ok=True)
    with (base / "supervisor.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        directory = base / attempt
        directory.mkdir()  # Refuse any reused attempt or overwritten log.

        def dump(name, value):
            write_identical(directory / name, (json.dumps(value, indent=2) + "\n").encode())

        metadata = {
            "identity": identity,
            "stage": stage,
            "attempt": attempt,
            "command": argv,
            "source_head": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip(),
            "supervisor_source_sha256": sha(Path(__file__)),
            "worker_source_sha256": sha(Path(argv[4])),
            "supervisor_pid": os.getpid(),
            "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "stdout_is_regular_project_file": True,
            "child_new_session": True,
            "PYTHONPATH": os.environ.get("PYTHONPATH", ""),
            "optimizer_or_protocol_modifications": False,
        }
        with (directory / "worker.log").open("xb") as log:
            try:
                child = subprocess.Popen(
                    argv,
                    cwd=ROOT,
                    stdin=subprocess.DEVNULL,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
            except Exception as error:
                dump("LAUNCH_FAILURE.json", {**metadata, "exception": repr(error)})
                raise
            dump("LAUNCH.json", {**metadata, "worker_pid": child.pid})
            code = child.wait()
            log.flush()
            os.fsync(log.fileno())
        dump(
            "EXIT.json",
            {
                "identity": identity,
                "stage": stage,
                "worker_pid": child.pid,
                "returncode": code,
                "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "worker_log_sha256": sha(directory / "worker.log"),
                "launch_sha256": sha(directory / "LAUNCH.json"),
            },
        )
        return code


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage", choices=("campaign", "validation16", "science", "close"), required=True
    )
    parser.add_argument("--attempt", required=True)
    args = parser.parse_args()
    raise SystemExit(0 if run(args.stage, args.attempt) == 0 else 1)
