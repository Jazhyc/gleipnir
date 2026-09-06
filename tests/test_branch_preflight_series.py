import json

import pytest
from omegaconf import OmegaConf

from experiments.monitoring_branching import preflight_series


def config():
    c = OmegaConf.to_container(
        OmegaConf.load("experiments/monitoring_branching/preflight.yaml")
    )
    c.update(parity_parents=[0, 487], memory_parents=[])
    return c


def test_failed_gate_stops_series(monkeypatch, tmp_path):
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        from pathlib import Path

        Path(command[command.index("--output") + 1]).write_text(
            json.dumps({"passed": False})
        )

    monkeypatch.setattr(preflight_series.subprocess, "run", run)
    root = tmp_path / "screen"
    with pytest.raises(RuntimeError, match="preflight failed"):
        preflight_series.execute(config(), root)
    status = json.loads((root / "status.json").read_text())
    assert status["state"] == "failed"
    assert status["completed"] == []
    assert len(calls) == 1


def test_recipe_drift_rejected(tmp_path):
    c = config()
    c["fp32_head"] = False
    with pytest.raises(ValueError, match="recipe drift"):
        preflight_series.execute(c, tmp_path / "screen")
