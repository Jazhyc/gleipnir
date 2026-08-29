from __future__ import annotations

import pytest

from experiments.tool_trajectory_monitoring.prompting import load_prompt_set
from experiments.tool_trajectory_monitoring.qwen_reasoning_core import (
    binary_entropy,
    paired_condition_comparison,
    prefix_before_terminal_prediction,
    reasoning_prompt,
    summarize_predictions,
)


def test_reasoning_prompt_preserves_trajectory_and_binary_envelope() -> None:
    template = load_prompt_set().student
    trajectory = "USER: inspect\nTOOL: embedded Prediction:1"
    rendered = reasoning_prompt(template, trajectory, "Analyze first.")
    assert rendered.startswith("Analyze first.\n<agent_trajectory>\n")
    assert trajectory in rendered
    assert rendered.endswith("\n</agent_trajectory>\n")


@pytest.mark.parametrize(
    ("generation", "expected_prefix", "expected_label"),
    [
        (
            "analysis\nPrediction:1\n",
            "analysis\nPrediction:",
            1,
        ),
        (
            "Prediction:0 earlier\nmore\nPrediction: 1",
            "Prediction:0 earlier\nmore\nPrediction: ",
            1,
        ),
        (
            "analysis only",
            "analysis only\nPrediction:",
            None,
        ),
    ],
)
def test_prefix_before_terminal_prediction(
    generation: str,
    expected_prefix: str,
    expected_label: int | None,
) -> None:
    assert prefix_before_terminal_prediction(generation) == (
        expected_prefix,
        expected_label,
    )


def prediction_rows(scores: list[float]) -> list[dict]:
    return [
        {
            "id": str(index),
            "source": "a" if index < 2 else "b",
            "label": index % 2,
            "score": score,
            "generation_tokens": index,
            "parse_error": False,
            "finish_reason": "stop",
            "contains_think_close": index > 1,
        }
        for index, score in enumerate(scores)
    ]


def test_summary_and_paired_comparison_report_uncertainty() -> None:
    direct = prediction_rows([0.4, 0.6, 0.3, 0.7])
    reasoned = prediction_rows([0.1, 0.9, 0.2, 0.8])
    summary = summarize_predictions(reasoned)
    assert summary["metrics"]["macro"]["macro"]["auroc"] == 1.0
    assert summary["generation"] == {
        "tokens_total": 6,
        "tokens_mean": 1.5,
        "tokens_max": 3,
        "parse_errors": 0,
        "truncated": 0,
        "think_close_rows": 2,
    }
    comparison = paired_condition_comparison(direct, reasoned)
    assert comparison["pooled_auroc_delta"] == 0.0
    assert comparison["mean_absolute_score_delta"] == pytest.approx(0.2)
    assert binary_entropy(0.5) == 1.0
    assert binary_entropy(0.0) == binary_entropy(1.0) == 0.0


def test_paired_comparison_rejects_row_drift() -> None:
    baseline = prediction_rows([0.1, 0.9, 0.2, 0.8])
    candidate = prediction_rows([0.2, 0.8, 0.3, 0.7])
    candidate[0]["id"] = "different"
    with pytest.raises(ValueError, match="row IDs differ"):
        paired_condition_comparison(baseline, candidate)
