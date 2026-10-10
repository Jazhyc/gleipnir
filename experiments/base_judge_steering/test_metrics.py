"""A common A/B shift must not be mistaken for better preference judging."""

import math

import pytest

from experiments.base_judge_steering.metrics import judge_report


def population(shift: float) -> list[dict]:
    return [
        dict(
            id=f"{condition}-{label}",
            prompt_sha256=f"{condition}-{label}",
            prompt_tokens=1,
            label=label,
            condition=condition,
            pair_id="p",
            order=str(label),
            source="s",
            lineage_group="q",
            margin=(2 * label - 1) + shift,
            score=1 / (1 + math.exp(-((2 * label - 1) + shift))),
            pab=0.99,
            monitor_p01=0.001,
            monitor_margin=0.0,
        )
        for condition in ["clean", "preferred_injected", "disfavored_injected"]
        for label in [0, 1]
    ]


def test_common_b_bias_is_separate_from_correct_answer_margin():
    result = judge_report(population(2), population(0))
    paired = result["paired"]["all"]
    assert paired["mean_raw_b_minus_a_margin_shift"] == 2
    assert paired["mean_correct_margin_shift"] == 0
    assert paired["fraction_chose_b"] == 1
    assert result["metrics"]["pooled"]["accuracy"] == 0.5


def test_judging_pair_drift_is_rejected():
    changed = population(1)
    changed[0]["label"] = 1
    with pytest.raises(ValueError, match="identity"):
        judge_report(changed, population(0))
