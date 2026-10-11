"""New default, frozen history, identity joins and effective training exposure."""

import json
from dataclasses import replace
from pathlib import Path

import pytest
import yaml

from gleipnir.campaigns.monitoring.contract import Campaign
from gleipnir.campaigns.monitoring.filtering import resolve_filter, retained_indices
from gleipnir.data.monitoring import file_hash, read_rows, write_rows

EXAMPLE = (
    Path(__file__).resolve().parents[2] / "experiments/monitoring_campaign/config.yaml"
)


@pytest.fixture
def source(tmp_path, monkeypatch):
    config = yaml.safe_load(EXAMPLE.read_text())
    config["augmentation_audit"] = False
    config["training_rows"] = 4
    rows = [
        {
            "dataset": "tool_trajectory/test",
            "index": i,
            "label": label,
            "teacher_rendered_prompt_sha256": f"prompt-{i}",
        }
        for i, label in enumerate([0, 1, 0, 1])
    ]
    targets = [
        {
            "dataset": row["dataset"],
            "index": row["index"],
            "label": row["label"],
            "rendered_prompt_sha256": f"prompt-{i}",
            "soft_target": score,
        }
        for i, (row, score) in enumerate(zip(rows, [0.1, 0.49, 0.5, 0.5], strict=True))
    ]
    for name, population in [("training", rows), ("teacher", list(reversed(targets)))]:
        path = tmp_path / f"{name}.jsonl"
        write_rows(path, population)
        config["inputs"][name] = {"path": path.name, "sha256": file_hash(path)}
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config))
    monkeypatch.setattr(Campaign, "inventory", lambda *a, **kw: {"passed": True})
    monkeypatch.setattr(
        "gleipnir.campaigns.monitoring.filtering.token_lengths",
        lambda ctx, rows: [10 + r["index"] for r in rows],
    )
    return Campaign.load(tmp_path, path)


def test_default_materializes_strict_selection_and_recounts_exposure(source):
    original = source.config_path.read_bytes()
    resolved = resolve_filter(source)
    assert [r["index"] for r in read_rows(resolved.input("training"))] == [0, 3]
    assert [r["index"] for r in read_rows(resolved.input("teacher"))] == [3, 0]
    assert resolved.config["training_rows"] == 2
    assert resolved.config["expected_steps"] == 1
    assert resolved.config["expected_training_tokens"] == 23
    assert resolved.job()["student_rows"] == str(resolved.input("training"))
    receipt = json.loads(resolved.input("filter").read_text())
    assert receipt["removed"] == receipt["retained"] == 2
    assert source.config_path.read_bytes() == original
    assert resolved.input("unfiltered_config") == source.config_path
    assert resolve_filter(source).config_path == resolved.config_path
    assert resolve_filter(resolved) is resolved
    source.data.joinpath("manifest.json").write_text("{}")
    assert resolve_filter(source).config_path == resolved.config_path
    assert resolved.input("training").read_bytes() == b"".join(
        line
        for line in source.input("training").read_bytes().splitlines(keepends=True)
        if json.loads(line)["index"] in (0, 3)
    )


def test_opt_out_and_historical_preparation_are_preserved(source):
    assert (
        resolve_filter(
            replace(source, config={**source.config, "teacher_filter": "none"})
        ).config_path
        == source.config_path
    )
    source.data.mkdir(parents=True)
    (source.data / "manifest.json").write_text("{}")
    assert resolve_filter(source) is source


def test_resolved_resume_still_binds_original_config(source):
    resolve_filter(source)
    source.config_path.write_text(source.config_path.read_text() + "\n# changed\n")
    with pytest.raises(ValueError, match="configuration changed"):
        resolve_filter(source)


def test_missing_teacher_or_changed_prompt_fails_closed(source):
    rows, targets = (
        read_rows(source.input("training")),
        read_rows(source.input("teacher")),
    )
    with pytest.raises(ValueError, match="coverage"):
        retained_indices(rows, targets[:-1])
    targets[0]["rendered_prompt_sha256"] = "changed"
    with pytest.raises(ValueError, match="provenance"):
        retained_indices(rows, targets)


def test_reject_unknown_policy(source):
    with pytest.raises(ValueError, match="teacher_filter"):
        resolve_filter(
            replace(source, config={**source.config, "teacher_filter": "typo"})
        )
