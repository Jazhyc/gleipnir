"""Exercise evaluation recovery without repeating completed campaign stages."""

import os

import pytest

from experiments.monitoring_hard_labels import run as campaign


def arrange(monkeypatch, tmp_path):
    monkeypatch.setattr(campaign, "ROOT", tmp_path)
    monkeypatch.setattr(campaign, "OUTPUT", tmp_path / "results")
    monkeypatch.setattr(campaign, "verify_preparation", lambda: {})
    monkeypatch.setattr(campaign, "training_environment", lambda *_: {})
    monkeypatch.setenv("PATH", "/usr/bin")
    monkeypatch.setenv("CUDA_HOME", "/usr/local/cuda")
    calls = []
    monkeypatch.setattr(
        campaign.subprocess, "run", lambda argv, **kwargs: calls.append((argv, kwargs))
    )
    return calls


def test_resume_never_runs_training_or_reference(monkeypatch, tmp_path):
    calls = arrange(monkeypatch, tmp_path)
    for variant in campaign.FRACTIONS:
        directory = campaign.OUTPUT / "4b" / variant
        directory.mkdir(parents=True)
        for name in ("complete.json", "parity_reference.json"):
            (directory / name).write_text("{}")
    campaign.run(resume_evaluation=True)
    assert [argv[2:] for argv, _ in calls] == [
        ["experiments.monitoring_hard_labels.evaluate", "--backend", "vllm"],
        ["experiments.monitoring_hard_labels.summarize"],
    ]
    assert calls[0][1]["env"]["PATH"].split(os.pathsep) == [
        str(tmp_path / ".venv/bin"), "/usr/local/cuda/bin", "/usr/bin"
    ]


def test_resume_rejects_missing_reference_before_launch(monkeypatch, tmp_path):
    calls = arrange(monkeypatch, tmp_path)
    directory = campaign.OUTPUT / "4b" / "hard000"
    directory.mkdir(parents=True)
    (directory / "complete.json").write_text("{}")
    with pytest.raises(ValueError, match="parity_reference"):
        campaign.run(resume_evaluation=True)
    assert calls == []
