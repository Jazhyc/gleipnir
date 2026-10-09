"""Protect configurable follow-ups and ID-only completion counts."""

import json
from types import SimpleNamespace

import pytest

from experiments.b200_augmented_bf16_optimized import run


def test_explicit_config_freezes_existing_merge_references(tmp_path, monkeypatch):
    config = tmp_path / "config.yaml"
    output = tmp_path / "output"
    output.mkdir()
    sources = {}
    for name in ("completed_merge", "completed_merged_parity", "completed_canary"):
        sources[name] = tmp_path / name
        sources[name].write_bytes(name.encode())
    selected = []
    ctx = SimpleNamespace(
        output=output,
        prepare=lambda: selected.append("prepared"),
        input=lambda name: sources[name],
    )
    monkeypatch.setattr(
        run.Campaign, "load", lambda root, path: (selected.append(path), ctx)[1]
    )
    monkeypatch.setattr(
        "sys.argv", ["followup", "--stage", "prepare", "--config", str(config)]
    )
    run.main()
    assert selected == [config, "prepared"]
    assert (output / "merged_artifact.json").read_bytes() == sources[
        "completed_merge"
    ].read_bytes()
    assert (output / "merged_parity.json").read_bytes() == sources[
        "completed_merged_parity"
    ].read_bytes()


@pytest.mark.parametrize(
    "splits,expected",
    [(("id",), 3012), (("id", "benchmark", "honest_controls"), 12126)],
)
def test_completion_counts_only_requested_populations(
    tmp_path, monkeypatch, splits, expected
):
    output, serving = tmp_path / "output", tmp_path / "serving"
    output.mkdir()
    serving.mkdir()
    config = {
        "campaign_id": "followup",
        "evaluation": {
            "populations": {"id": 3012, "benchmark": 5184, "honest_controls": 3930}
        },
    }
    ctx = SimpleNamespace(
        root=tmp_path,
        config_path=tmp_path / "config",
        config=config,
        source_root=tmp_path,
        output=output,
        serving=serving,
        splits=splits,
        check=lambda: None,
    )
    monkeypatch.setattr(run.Campaign, "load", lambda root, path: ctx)
    monkeypatch.setattr(run, "Completed", lambda *args: ctx)

    async def completed(context):
        assert context is ctx

    monkeypatch.setattr(run, "optimized", completed)
    monkeypatch.setattr("sys.argv", ["followup", "--stage", "score"])
    run.main()
    assert json.loads((output / "status.json").read_text()) == {
        "stage": "complete",
        "rows": expected,
    }
