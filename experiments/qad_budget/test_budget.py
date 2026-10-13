"""Subset identity/nesting, source derivation and remote launch fail closed."""

from collections import Counter
from pathlib import Path

import pytest

from experiments.qad_budget import launch
from experiments.qad_budget.generate import transform
from experiments.qad_budget.prepare import select


def rows():
    workload = [
        {
            "id": str(i),
            "dataset": "train",
            "source_dataset": str(i % 2),
            "label": i % 2,
            "prompt_tokens": 100 + 100 * (i % 4),
            "prompt_sha256": str(i),
        }
        for i in range(64)
    ]
    teachers = [
        {
            "index": r["id"],
            "dataset": r["dataset"],
            "label": r["label"],
            "soft_target": 0.8 if r["label"] else 0.2,
            "rendered_prompt_sha256": r["prompt_sha256"],
        }
        for r in workload
    ]
    return workload, teachers


def test_nested_quota_stream_is_unique_deterministic_and_balanced():
    workload, teachers = rows()
    short, _, _ = select(workload, teachers, 16, 0)
    long, strata, _ = select(workload, teachers, 32, 0)
    assert short == long[:16]
    assert len(set(long)) == 32
    assert select(workload, list(reversed(teachers)), 32, 0)[0] == long
    counts = Counter(strata[i] for i in long)
    assert all(value == 8 for value in counts.values())


@pytest.mark.parametrize("invalid", ["label", "prompt", "duplicate"])
def test_invalid_teacher_join_is_rejected(invalid):
    workload, teachers = rows()
    if invalid == "label":
        teachers[0]["label"] = 1
    elif invalid == "prompt":
        teachers[0]["rendered_prompt_sha256"] = "changed"
    else:
        teachers[1] = teachers[0]
    with pytest.raises(ValueError):
        select(workload, teachers, 16, 0)


@pytest.mark.parametrize("stage", ["train", "reference", "evaluate"])
def test_short_qad_derivative_is_exactly_reversible(stage):
    source = Path(f"experiments/quantization_recovery/{stage}.py").read_text()
    generated, changes = transform(
        source,
        stage,
        config_path="resolved/arm32.yaml",
        runtime_module="experiments.qad_budget.runtime",
    )
    for change in reversed(changes):
        generated = generated.replace(change["new"], change["old"])
    assert generated == source


def test_launch_refuses_active_gpu(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    p = tmp_path / "experiments/qad_budget/config.yaml"
    p.parent.mkdir(parents=True)
    p.write_text("experiment: test\n")
    monkeypatch.setattr(launch.subprocess, "check_output", lambda *a, **k: "123\n")
    with pytest.raises(RuntimeError, match="active compute"):
        launch.main()
