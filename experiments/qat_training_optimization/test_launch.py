"""Mock remote process creation and fail-closed resident-worker retry guards."""

import json
from types import SimpleNamespace

import pytest

from experiments.qat_training_optimization import launch


def fixture_worker(tmp_path, monkeypatch, *, live):
    monkeypatch.chdir(tmp_path)
    config = tmp_path / "experiments/qat_training_optimization/config.yaml"
    config.parent.mkdir(parents=True)
    config.write_text("experiment: qat-training-optimization01\n")
    output = tmp_path / "results/qat-training-optimization01"
    output.mkdir(parents=True)
    (output / "worker.pid").write_text("999999")
    (output / "status.json").write_text(json.dumps({"phase": "failed"}))
    log = tmp_path / "logs/runpod/qat-training-optimization01/worker.log"
    log.parent.mkdir(parents=True)
    log.write_text("original failure")
    monkeypatch.setattr("sys.argv", ["launch", "--retry-failed"])

    def kill(pid, signal):
        assert pid == 999999 and signal == 0
        if not live:
            raise ProcessLookupError

    monkeypatch.setattr(launch.os, "kill", kill)
    return output, log


def test_failed_worker_retry_preserves_original_receipts(tmp_path, monkeypatch):
    output, log = fixture_worker(tmp_path, monkeypatch, live=False)
    monkeypatch.setattr(launch, "training_environment", lambda *a, **k: {})
    calls = []

    def spawn(args, **kwargs):
        calls.append((args, kwargs))
        return SimpleNamespace(pid=42)

    monkeypatch.setattr(launch.subprocess, "Popen", spawn)
    launch.main()
    archive = output / "failed_attempt_999999"
    assert (archive / "worker.log").read_text() == "original failure"
    assert json.loads((archive / "status.json").read_text())["phase"] == "failed"
    assert (output / "worker.pid").read_text() == "42"
    assert (output / "worker_source.py").is_file()
    assert log.is_file()
    assert len(calls) == 1 and calls[0][1]["start_new_session"]


def test_retry_requires_explicit_failed_attempt(tmp_path, monkeypatch):
    output, _ = fixture_worker(tmp_path, monkeypatch, live=False)
    monkeypatch.setattr("sys.argv", ["launch"])
    with pytest.raises(RuntimeError, match="existing worker"):
        launch.main()
    assert (output / "worker.pid").read_text() == "999999"
