"""Bind newly selected controls without changing explicit historical references."""

import json

import pytest

from experiments.b200_inference_benchmark import run


def test_selected_baseline_binds_results_recipe_and_workload(tmp_path, monkeypatch):
    monkeypatch.setattr(run, "ROOT", tmp_path)
    monkeypatch.setattr(run, "EXPERIMENT", tmp_path)
    monkeypatch.setattr(run, "DATA", tmp_path)
    manifest = tmp_path / "manifest.json"
    manifest.write_text("frozen workload")
    summary = tmp_path / "control/summary.json"
    summary.parent.mkdir()
    summary.write_text(
        json.dumps({"status": "complete", "manifest_sha256": run.sha(manifest)})
    )
    recipe = tmp_path / "recipe.json"
    recipe.write_text("validated recipe")
    selection = {
        "results": "control",
        "summary_sha256": run.sha(summary),
        "manifest_sha256": run.sha(manifest),
        "validated_recipe_config": "recipe.json",
        "validated_recipe_config_sha256": run.sha(recipe),
    }
    (tmp_path / "baseline.json").write_text(json.dumps(selection))
    condition = {"name": "candidate"}
    resolved = run.resolve_kernel_baseline(condition)
    assert resolved["baseline"] == "control"
    assert "baseline" not in condition
    explicit = run.resolve_kernel_baseline({**condition, "baseline": "selected"})
    assert explicit == resolved
    for path in (summary, recipe, manifest):
        original = path.read_text()
        path.write_text(original + " ")
        with pytest.raises(ValueError, match="baseline identity drift"):
            run.resolve_kernel_baseline(condition)
        path.write_text(original)


def test_explicit_historical_baselines_are_preserved():
    condition = {"baseline": "results/historical/control"}
    assert run.resolve_kernel_baseline(condition) is condition
    assert run.resolve_kernel_baseline(None) is None
