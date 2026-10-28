"""Direct-file supervision tests use a tiny Python child, never a JAX model."""

import hashlib
import json
import sys

import pytest

import supervise_phase_g23 as supervisor


def test_stage_commands_cannot_override_optimizer_or_budget():
    for stage in ("campaign", "science", "close", "validation16"):
        argv = supervisor.command(stage)
        assert argv[:4] == ["nice", "-n", "10", sys.executable]
        assert argv[5:] == ([] if stage == "validation16" else ["--stage", stage])
    with pytest.raises(ValueError):
        supervisor.command("campaign --max-updates 2000")


@pytest.mark.parametrize("name", ["../outside", "/tmp/other", "", "bad/name"])
def test_attempt_path_escape_is_rejected(name):
    with pytest.raises(ValueError, match="simple"):
        supervisor.run("campaign", name)


def test_direct_regular_file_child_and_durable_exit(tmp_path, monkeypatch):
    monkeypatch.setattr(supervisor, "ROOT", tmp_path)
    binding = {"source": {}, "protocols": {}}
    identity = hashlib.sha256(json.dumps(binding, sort_keys=True).encode()).hexdigest()
    protocol = tmp_path / "results/phase_g/g2/v1/G2_G3_PROTOCOL_IDENTITY.json"
    protocol.parent.mkdir(parents=True)
    protocol.write_text(json.dumps({"identity": identity, "binding": binding}))
    script = tmp_path / "child.py"
    script.write_text(
        "import os,stat\nprint(stat.S_ISREG(os.fstat(1).st_mode),os.getpid()==os.getsid(0),flush=True)\n"
    )
    monkeypatch.setattr(
        supervisor, "command", lambda stage: [sys.executable, "-I", "-u", "-B", str(script)]
    )
    monkeypatch.setattr(
        supervisor.subprocess, "check_output", lambda *a, **kw: "fixture-source-head\n"
    )
    assert supervisor.run("campaign", "test") == 0
    directory = tmp_path / "results/runs/phase_g/g23/v1/process_supervision/test"
    assert (directory / "worker.log").read_text() == "True True\n"
    result = json.loads((directory / "EXIT.json").read_text())
    assert result["returncode"] == 0 and result["identity"] == identity
    assert result["worker_log_sha256"] == supervisor.sha(directory / "worker.log")
    assert result["launch_sha256"] == supervisor.sha(directory / "LAUNCH.json")
    with pytest.raises(FileExistsError):
        supervisor.run("campaign", "test")


def test_changed_bound_source_is_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(supervisor, "ROOT", tmp_path)
    file = tmp_path / "source.py"
    file.write_text("before")
    binding = {"source": {"source.py": supervisor.sha(file)}, "protocols": {}}
    identity = hashlib.sha256(json.dumps(binding, sort_keys=True).encode()).hexdigest()
    path = tmp_path / "results/phase_g/g2/v1/G2_G3_PROTOCOL_IDENTITY.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"identity": identity, "binding": binding}))
    file.write_text("after")
    with pytest.raises(ValueError, match="frozen source"):
        supervisor.protocol_identity()
