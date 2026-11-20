import json
from pathlib import Path

import pytest

from experiments.ood_recall_at_fpr.compute import evaluate, load_scores
from experiments.ood_recall_at_fpr.plot import frontier

CONFIG = json.loads((Path(__file__).parent / "config.json").read_text())


def test_config_lists_each_monitor_once_with_costs() -> None:
    names = [monitor["name"] for monitor in CONFIG["monitors"]]
    assert len(names) == len(set(names)) == 25
    for monitor in CONFIG["monitors"]:
        costs = monitor["cost_usd_per_1k"]
        assert costs == sorted(costs) and all(cost > 0 for cost in costs)
        assert len(costs) == (
            2 if monitor["name"] in ("Gleipnir 4B", "Gleipnir 27B") else 1
        )


def test_evaluate_averages_sources_with_equal_weight() -> None:
    # Source "a": 100 benign at 0.0-0.99, so 1% FPR allows one false positive.
    reference = {f"a{i}": (0, "a") for i in range(100)}
    scores = {f"a{i}": i / 100 for i in range(100)}
    reference |= {"a_pos_hi": (1, "a"), "a_pos_lo": (1, "a")}
    scores |= {"a_pos_hi": 0.985, "a_pos_lo": 0.5}
    # Source "b": too few benign rows for any false positive at 1%.
    reference |= {"b0": (0, "b"), "b1": (0, "b"), "b_pos": (1, "b")}
    scores |= {"b0": 0.1, "b1": 0.2, "b_pos": 0.9}

    result = evaluate(scores, reference, 0.01)

    assert result["per_source"]["a"]["recall"] == pytest.approx(0.5)
    assert result["per_source"]["a"]["achieved_fpr"] == pytest.approx(0.01)
    assert result["per_source"]["b"]["recall"] == pytest.approx(1.0)
    assert result["per_source"]["b"]["achieved_fpr"] == 0.0
    assert result["mean_recall"] == pytest.approx(0.75)


def test_load_scores_rejects_incomplete_coverage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "experiments.ood_recall_at_fpr.compute.REPOSITORY_ROOT", tmp_path
    )
    (tmp_path / "scores.jsonl").write_text('{"id": "x", "score": 0.5}\n')
    monitor = {"name": "m", "files": ["scores.jsonl"], "score_field": "score"}
    with pytest.raises(ValueError, match="cover"):
        load_scores(monitor, {"x": (0, "a"), "y": (1, "a")})


def test_frontier_keeps_nondominated_points() -> None:
    points = [
        (1.0, 0.5, "cheap"),
        (2.0, 0.4, "dominated"),
        (3.0, 0.9, "best"),
        (3.0, 0.8, "tie-cost"),
    ]
    assert [name for _, _, name in frontier(points)] == ["cheap", "best"]
