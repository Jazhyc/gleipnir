"""Focused safeguards for the new A/B serving gate."""

import pytest

from experiments.augmented_judge_evaluation.evaluate import parity_metrics
from experiments.augmented_judge_evaluation.prepare import configuration


def test_agreement_without_adapter_effect_is_rejected():
    scores = [0.1, 0.9, 0.2, 0.8]
    assert parity_metrics(scores, scores)["passed"]
    assert not parity_metrics(scores, scores, scores, scores)["passed"]


def test_matching_effect_and_small_backend_difference_pass():
    result = parity_metrics(
        [0.1, 0.9, 0.2, 0.8],
        [0.11, 0.91, 0.21, 0.81],
        [0.3, 0.7, 0.4, 0.6],
        [0.31, 0.71, 0.41, 0.61],
    )
    assert result["passed"]
    assert result["adapter_effect"] == pytest.approx(0.2)


def test_nonfinite_reference_fails_closed():
    with pytest.raises(ValueError, match="nonfinite"):
        parity_metrics([0.1, 0.9], [float("nan"), 0.9])


def test_large_score_difference_is_rejected():
    assert not parity_metrics([0.3, 0.7], [0.1, 0.9])["passed"]


def test_only_missing_population_cells_are_scored():
    config = configuration()
    assert config["surface"] == "AB"
    assert set(config["models"]["4b"]["adapters"]) == {"augmented"}
    assert set(config["models"]["9b"]["adapters"]) == {"regular", "augmented"}
    assert config["models"]["9b"]["score_base"]
    assert not config["models"]["4b"].get("score_base", False)
    assert config["engine"]["gdn_prefill_backend"] == "triton"
