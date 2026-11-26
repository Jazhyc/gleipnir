import pytest

from experiments.gleipnir4b_full_specification.compare import summarize
from experiments.gleipnir4b_full_specification.score import check


def _truth() -> dict:
    truth = {}
    for source in ("a", "b"):
        for k in range(20):
            truth[f"{source}{k}"] = (source, int(k >= 10))
    return truth


def test_summarize_reports_difference_and_flips() -> None:
    truth = _truth()
    perfect = {i: 0.9 if label else 0.1 for i, (_, label) in truth.items()}
    flat = {i: 0.4 for i in truth}
    result = summarize(perfect, flat, truth, "ood")
    assert result["rows"] == 40
    assert result["full_specification"]["macro"] == pytest.approx(1.0)
    assert set(result["full_specification"]["sources"]) == {"a", "b"}
    assert result["macro_difference"] > 0
    assert result["threshold_flips"] == 20


def test_summarize_rejects_different_rows() -> None:
    truth = _truth()
    scores = {i: 0.5 for i in truth}
    with pytest.raises(ValueError):
        summarize(scores, dict(list(scores.items())[1:]), truth, "ood")


def test_check_requires_full_finite_coverage() -> None:
    workload = [{"id": "x"}, {"id": "y"}]
    check([{"id": "x", "score": 0.2}, {"id": "y", "score": 1.0}], workload, 2)
    with pytest.raises(ValueError):
        check([{"id": "x", "score": 0.2}], workload, 2)
    with pytest.raises(ValueError):
        check(
            [{"id": "x", "score": 0.2}, {"id": "y", "score": float("nan")}], workload, 2
        )
