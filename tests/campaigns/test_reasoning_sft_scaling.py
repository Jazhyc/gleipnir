import pytest

from experiments.reasoning_sft_scaling.core import (
    REASONING_JUDGE_PROMPT,
    prefix_before_prediction,
    reasoning_student_prompt,
)
from experiments.reasoning_sft_scaling.evaluate import repository_path


def test_reasoning_prompt_preserves_only_example_body() -> None:
    direct = (
        "binary instructions\n\n<context>\nUSER: x\n</context>"
        "\n\n<output>\ny\n</output>"
    )
    result = reasoning_student_prompt(direct)
    assert result.startswith(REASONING_JUDGE_PROMPT)
    assert "binary instructions" not in result
    assert result.endswith("<output>\ny\n</output>")


def test_reasoning_prompt_requires_context_boundary() -> None:
    with pytest.raises(ValueError, match="context"):
        reasoning_student_prompt("missing example body")


def test_prefix_before_prediction_removes_only_final_decision() -> None:
    generation = "Prediction:0 was considered.\nPrediction:1"
    prefix, decision = prefix_before_prediction(generation)
    assert decision == 1
    assert prefix == "Prediction:0 was considered.\nPrediction:"


def test_prefix_before_prediction_has_parse_fallback() -> None:
    prefix, decision = prefix_before_prediction("unfinished reasoning ")
    assert decision is None
    assert prefix == "unfinished reasoning\nPrediction:"


def test_repository_path_makes_artifact_manifests_portable() -> None:
    assert repository_path("results/example").is_absolute()
    assert repository_path("results/example").as_posix().endswith(
        "/gleipnir/results/example"
    )
