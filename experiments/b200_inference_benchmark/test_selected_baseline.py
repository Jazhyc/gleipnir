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
    acceptance = tmp_path / "quality_acceptance.json"
    acceptance.write_text(json.dumps({"status": "user_accepted_finite"}))
    selection.update(
        quality_acceptance="quality_acceptance.json",
        quality_acceptance_sha256=run.sha(acceptance),
    )
    (tmp_path / "baseline.json").write_text(json.dumps(selection))
    resolved = run.resolve_kernel_baseline(condition)
    assert resolved["baseline_quality_acceptance_sha256"] == run.sha(acceptance)
    acceptance.write_text(json.dumps({"status": "unaccepted"}))
    with pytest.raises(ValueError, match="baseline identity drift"):
        run.resolve_kernel_baseline(condition)


def test_explicit_historical_baselines_are_preserved():
    condition = {"baseline": "results/historical/control"}
    assert run.resolve_kernel_baseline(condition) is condition
    assert run.resolve_kernel_baseline(None) is None


def test_resident_worker_reuse_allows_only_client_reference_changes():
    config = {
        "gleipnir_frost_fp4": {"kernel.py": "validated_hash"},
        "serving_condition": {
            "attention_precision": "mxfp8",
            "high_reference": "historical",
        },
    }
    command = [
        "python", "--max-num-seqs", "128", "--additional-config", json.dumps(config)
    ]
    config["serving_condition"]["high_reference"] = "selected"
    updated = [*command[:-1], json.dumps(config)]
    assert run.compatible_server_command(command, updated)
    changed_capacity = [*updated[:2], "64", *updated[3:]]
    assert not run.compatible_server_command(command, changed_capacity)
    config["serving_condition"]["attention_precision"] = "bf16"
    assert not run.compatible_server_command(
        command, [*command[:-1], json.dumps(config)]
    )
    config["serving_condition"]["attention_precision"] = "mxfp8"
    config["gleipnir_frost_fp4"]["kernel.py"] = "changed_hash"
    assert not run.compatible_server_command(
        command, [*command[:-1], json.dumps(config)]
    )
    assert not run.compatible_server_command(command, [*command[:-1], "invalid_json"])


def test_accepted_baseline_does_not_hide_strict_failure_or_accept_new_drift():
    limits = {
        "max_mean_absolute_difference": 0.02,
        "min_correlation": 0.995,
        "min_adapter_effect": 0.01,
    }
    comparisons = {
        "adapter": {"mean_absolute_difference": 0.03, "correlation": 0.992},
        "kernel_baseline": {"mean_absolute_difference": 0.001, "correlation": 0.999},
    }
    assert run.parity_status(comparisons, limits, 0.5, baseline_accepted=False) == (
        False,
        False,
    )
    assert run.parity_status(comparisons, limits, 0.5, baseline_accepted=True) == (
        False,
        True,
    )
    comparisons["kernel_baseline"]["mean_absolute_difference"] = 0.1
    assert run.parity_status(comparisons, limits, 0.5, baseline_accepted=True) == (
        False,
        False,
    )
    comparisons["kernel_baseline"]["mean_absolute_difference"] = 0.001
    assert run.parity_status(comparisons, limits, 0, baseline_accepted=True) == (
        False,
        False,
    )
