"""Matched teacher diagnostic statistics."""

import pytest
from scipy.special import expit

from experiments.teacher_agreement.run import agreement


def rows(scale=1):
    return [
        dict(
            qwen_score=float(expit(m * scale)),
            kimi_score=float(expit(m)),
            qwen_margin=m * scale,
            kimi_margin=m,
            label=int(m > 0),
        )
        for m in [-2.0, -1.0, 1.0, 2.0]
    ]


def test_identical():
    result = agreement(rows())
    assert result["probability_mae"] == 0
    assert result["spearman"] == 1
    assert result["descriptive_qwen_margin_on_kimi_slope"] == pytest.approx(1)


def test_confidence_without_ranking_change():
    result = agreement(rows(2))
    assert result["spearman"] == 1
    assert result["decision_disagreement"] == 0
    assert result["qwen_mean_confidence"] > result["kimi_mean_confidence"]
    assert result["descriptive_qwen_margin_on_kimi_slope"] == pytest.approx(2)


def test_invalid_and_constant():
    with pytest.raises(ValueError):
        agreement([])
    records = rows()
    records[0]["qwen_score"] = float("nan")
    with pytest.raises(ValueError):
        agreement(records)
    assert agreement([rows()[0]])["spearman"] is None
