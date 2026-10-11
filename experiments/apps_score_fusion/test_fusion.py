"""Fusion uses only numeric predictions and preserves exact paired source IDs."""

import numpy as np
import pytest

from experiments.apps_score_fusion.run import join_scores
from gleipnir.evaluation.score_fusion import (
    combined_scores,
    feature_matrix,
    fit_combiner,
    fit_threshold,
)
from gleipnir.serving.monitor_score import score_payload


def test_portable_logistic_coefficients_separate_fit_examples():
    rows = [
        {"safety_margin": s, "injection_margin": i, "ground_truth": int(s - i > 0)}
        for s in (-3, -1, 1, 3)
        for i in (-2, 2)
    ]
    model = fit_combiner(rows, "two_score")
    scores = combined_scores(rows, model)
    assert ((scores > 0.5) == [r["ground_truth"] for r in rows]).all()
    assert model["coefficient"][0] > 0
    assert model["coefficient"][1] < 0
    changed = [
        r | {"ground_truth": 1 - r["ground_truth"], "condition": "new"} for r in rows
    ]
    assert np.array_equal(scores, combined_scores(changed, model))


def test_threshold_uses_fit_negatives_and_retains_ties():
    rows = [{"ground_truth": 0} for _ in range(50)] + [{"ground_truth": 1}]
    scores = np.array([0.1] * 49 + [0.9, 0.99])
    assert fit_threshold(rows, scores) == 0.1
    assert (scores[:50] > fit_threshold(rows, scores)).sum() == 1


def test_feature_shapes_and_nonfinite_rejection():
    row = {"safety_margin": 2, "injection_margin": 3}
    assert feature_matrix([row], "interaction").tolist() == [[2, 3, 6]]
    with pytest.raises(ValueError):
        feature_matrix([row | {"injection_margin": float("nan")}], "two_score")


def test_join_rejects_changed_source_and_wrong_behavior_label():
    safety = score_payload([1.0, 2.0], 10) | {
        "id": "a",
        "task_id": "1",
        "condition": "clean",
        "ground_truth": 0,
        "sample_sha256": "source",
        "prompt_sha256": "safety",
    }
    presence = score_payload([2.0, 1.0], 11) | {
        "id": "a",
        "behavior_label": 0,
        "prompt_sha256": "presence",
        "original_metadata": {
            k: safety[k] for k in ("task_id", "condition", "sample_sha256")
        },
    }
    assert join_scores([safety], [presence])[0]["ground_truth"] == 0
    with pytest.raises(ValueError):
        join_scores([safety], [presence | {"behavior_label": 1}])
    with pytest.raises(ValueError):
        join_scores(
            [safety],
            [
                presence
                | {
                    "original_metadata": presence["original_metadata"]
                    | {"sample_sha256": "different"}
                }
            ],
        )
