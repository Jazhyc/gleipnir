"""Paired logits remain informative when alarm thresholds saturate."""

import pytest

from experiments.caft_honest_steering.metrics import comparison


def row(identity: str, score: float, margin: float) -> dict:
    return dict(
        id=identity,
        prompt_sha256=identity,
        prompt_tokens=1,
        ground_truth=0,
        condition="clean",
        score=score,
        margin=margin,
    )


def test_fixed_threshold_and_continuous_effect_with_saturated_alarms() -> None:
    baseline = [row("a", 0.7, 1), row("b", 0.8, 2)]
    scored = [row("a", 0.8, 2), row("b", 0.9, 4)]
    result = comparison(scored, baseline, 0.2)
    assert result["fpr"] == 1
    assert result["paired_margin_shift_mean"] == 1.5
    assert result["paired_score_shift_mean"] == pytest.approx(0.1)
    assert result["auroc"] is None


def test_strict_clean_threshold_preserves_ties() -> None:
    baseline = [row("a", 0.2, -1), row("b", 0.3, -0.5)]
    result = comparison(baseline, baseline, 0.2)
    assert result["fpr"] == 0.5
    assert result["paired_margin_shift_mean"] == 0


def test_pairing_and_finite_guards() -> None:
    baseline = [row("a", 0.2, -1)]
    with pytest.raises(ValueError, match="identity"):
        comparison([row("b", 0.2, -1)], baseline, 0.2)
    with pytest.raises(ValueError, match="nonfinite"):
        comparison([row("a", 0.2, float("nan"))], baseline, 0.2)
