"""The systems campaign preserves workload identity and the allocation deadline."""

import datetime
import json
from pathlib import Path

import pytest

from experiments.fp4_inference import run
from experiments.fp4_inference.run import condition_config, remaining_seconds


def test_deadline_leaves_ten_minutes_before_gpu_expiry():
    expiry = datetime.datetime(2026, 9, 29, 23, 39, 21, tzinfo=datetime.UTC)
    assert remaining_seconds(expiry) == -600
    assert remaining_seconds(expiry - datetime.timedelta(minutes=20)) == 600


@pytest.mark.parametrize("name", ["../baseline", "baseline.json", "", "/tmp/config"])
def test_condition_resolution_rejects_path_traversal(name):
    with pytest.raises(ValueError):
        condition_config(name)


def test_blackwell_baseline_preserves_historical_serving_contract():
    historical = json.loads(
        Path("experiments/local_inference/iteration32.json").read_text()
    )
    blackwell = json.loads(condition_config("baseline").read_text())
    blackwell.pop("log_dir")
    assert blackwell.pop("output") != historical.pop("output")
    assert blackwell == historical


def test_launcher_creates_parent_and_records_process_result(tmp_path, monkeypatch):
    output = tmp_path / "new_campaign" / "baseline"
    config = tmp_path / "baseline.json"
    config.write_text(json.dumps({"output": str(output)}))
    monkeypatch.setattr(run, "condition_config", lambda name: config)
    monkeypatch.setattr(run, "remaining_seconds", lambda: 600)
    monkeypatch.setattr(run, "ROOT", tmp_path / "receipts")
    monkeypatch.setattr(run.sys, "argv", ["run", "--condition", "baseline"])

    class Process:
        def __init__(self, command, **kwargs):
            assert output.parent.is_dir()
            assert kwargs["start_new_session"]
            self.pid = 123

        def wait(self, timeout):
            assert timeout == 600
            return 0

    monkeypatch.setattr(run.subprocess, "Popen", Process)
    run.main()
    receipt = json.loads((tmp_path / "receipts/baseline_execution.json").read_text())
    assert receipt["returncode"] == 0
    assert not receipt["deadline_reached"]
