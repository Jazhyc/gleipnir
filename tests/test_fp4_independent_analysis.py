"""Independent engine summaries must retain configuration and gate failures."""

import json

import numpy as np
import pytest

from experiments.fp4_inference import independent_analysis as analysis
from experiments.fp4_inference.repeat_analysis import IDENTITY_FIELDS, ROW_FIELDS


def setup_runs(tmp_path, monkeypatch):
    baselines, candidates = [], []
    runs = {}
    for method, paths, time in (
        ("baseline", baselines, 10),
        ("candidate", candidates, 8),
    ):
        for index in range(2):
            root = tmp_path / f"{method}_{index}"
            root.mkdir()
            paths.append(root)
            (root / "launch_config.json").write_text(
                json.dumps({"engine": {"method": method}, "output": str(root)})
            )
            (root / "result.json").write_text("{}")
            result = dict.fromkeys(IDENTITY_FIELDS, "same")
            result.update(median_seconds=time + index, serving_parity_passed=True)
            rows = [{k: str(i) for k in ROW_FIELDS} for i in range(3)]
            scores = np.array([[0.1, 0.4, 0.8]]) + index * 0.01
            runs[root] = result, rows, scores, scores * 10
    manifests = [tmp_path / f"manifest_{i}.json" for i in range(2)]
    for path in manifests:
        path.write_text("{}")
    monkeypatch.setattr(analysis, "load_run", runs.__getitem__)
    monkeypatch.setattr(
        analysis,
        "report",
        lambda *args: {
            "full_pair": {"scoring_speedup": 1.25},
            "heldout_quality_gate": {"passed": True},
            "confirmation_gates_passed": True,
        },
    )
    return baselines, candidates, manifests, runs


def test_independent_ranges_and_gate_failure_are_preserved(tmp_path, monkeypatch):
    baselines, candidates, manifests, runs = setup_runs(tmp_path, monkeypatch)
    result = analysis.analyze(baselines, candidates, manifests)
    assert result["median_speedup"] == pytest.approx(10.5 / 8.5)
    assert result["methods"]["candidate"]["variation"]["independent_starts"] == 2
    assert result["methods"]["candidate"]["variation"]["max_score_range"] == (
        pytest.approx(0.01)
    )
    assert result["all_confirmation_gates_passed"]
    assert set(result["methods"]["candidate"]["variation_by_source"]) == {"0", "1", "2"}
    assert result["methods"]["candidate"]["variation_by_source"]["1"][
        "max_score_range"
    ] == pytest.approx(0.01)
    runs[baselines[1]][0]["serving_parity_passed"] = False
    assert not analysis.analyze(baselines, candidates, manifests)[
        "all_confirmation_gates_passed"
    ]


def test_independent_summary_rejects_recipe_identity_and_repeat_changes(
    tmp_path, monkeypatch
):
    baselines, candidates, manifests, runs = setup_runs(tmp_path, monkeypatch)
    config = candidates[1] / "launch_config.json"
    original = config.read_text()
    config.write_text('{"engine": {"method": "changed"}, "output": "changed"}')
    with pytest.raises(ValueError, match="recipe changed"):
        analysis.analyze(baselines, candidates, manifests)
    config.write_text(original)
    runs[candidates[1]][0]["subset_sha256"] = "changed"
    with pytest.raises(ValueError, match="result identity"):
        analysis.analyze(baselines, candidates, manifests)
    runs[candidates[1]][0]["subset_sha256"] = "same"
    result, rows, scores, margins = runs[candidates[1]]
    runs[candidates[1]] = result, rows, np.vstack([scores, scores]), margins
    with pytest.raises(ValueError, match="one full pass"):
        analysis.analyze(baselines, candidates, manifests)
    with pytest.raises(ValueError, match="distinct output"):
        analysis.analyze([baselines[0]] * 2, candidates, manifests)
    with pytest.raises(ValueError, match="matched independent"):
        analysis.analyze(baselines, candidates[:1], manifests)
