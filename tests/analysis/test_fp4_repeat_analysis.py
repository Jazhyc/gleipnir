"""Repeated comparisons preserve identities and reject inconsistent artifacts."""

import json

import pytest

from experiments.fp4_inference.repeat_analysis import IDENTITY_FIELDS, analyze


def write_run(root, *, seconds, scores):
    root.mkdir()
    repeats = []
    for i, score in enumerate(scores):
        predictions = [
            {
                "id": str(j),
                "source": "test",
                "label": j,
                "prompt_sha256": str(j),
                "tokens": 10,
                "score": value,
                "logit_margin": value,
            }
            for j, value in enumerate((score, 0.9))
        ]
        (root / f"predictions_{i}.json").write_text(json.dumps(predictions))
        repeats.append({"repeat": i, "seconds": seconds})
    result = dict.fromkeys(IDENTITY_FIELDS, "same")
    result.update(
        rows=2,
        prompt_tokens=20,
        repeats=repeats,
        median_seconds=seconds,
        median_scores=[sorted(scores)[len(scores) // 2], 0.9],
        serving_parity_passed=True,
        metrics={"macro": {"macro": {"auroc": 0.9, "brier": 0.1, "pauroc_at_20": 0.8}}},
    )
    (root / "result.json").write_text(json.dumps(result))


def test_repeated_medians_ranges_and_selection(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    write_run(a, seconds=10, scores=[0.49, 0.49, 0.49])
    write_run(b, seconds=8, scores=[0.49, 0.51, 0.53])
    report = analyze(a, b)
    assert report["median_speedup"] == 1.25
    assert report["score_drift"]["threshold_flips"] == 1
    assert report["stability"]["candidate"]["threshold_unstable_rows"] == 1
    assert report["stability"]["candidate"]["max_score_range"] == pytest.approx(0.04)
    assert report["development_selection_gates_passed"]


def test_repeats_reject_changed_predictions_and_medians(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    for root in (a, b):
        write_run(root, seconds=10, scores=[0.49, 0.49, 0.49])
    path = b / "predictions_1.json"
    rows = json.loads(path.read_text())
    rows[0]["prompt_sha256"] = "changed"
    path.write_text(json.dumps(rows))
    with pytest.raises(ValueError, match="Repeat prediction identity"):
        analyze(a, b)
    rows[0]["prompt_sha256"] = "0"
    path.write_text(json.dumps(rows))
    path = b / "result.json"
    result = json.loads(path.read_text())
    result["median_scores"][0] = 0.6
    path.write_text(json.dumps(result))
    with pytest.raises(ValueError, match="Stored score medians"):
        analyze(a, b)


def test_repeats_reject_unmatched_runtime(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    for root in (a, b):
        write_run(root, seconds=10, scores=[0.49, 0.49, 0.49])
    path = b / "result.json"
    result = json.loads(path.read_text())
    result["runner_sha256"] = "changed"
    path.write_text(json.dumps(result))
    with pytest.raises(ValueError, match="runner_sha256"):
        analyze(a, b)
