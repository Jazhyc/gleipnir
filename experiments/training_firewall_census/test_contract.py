"""Focused evidence-identity and descriptive-intersection checks."""

import pytest

from experiments.training_firewall_census.analyze import describe, join_predictions


def test_join_rejects_label_evidence_and_membership_drift() -> None:
    row = {
        "index": "a",
        "trajectory_sha256": "t",
        "student_prompt_sha256": "p",
        "label": 0,
        "view": "original",
    }
    pred = {
        "id": "a",
        "trajectory_sha256": "t",
        "student_prompt_sha256": "p",
        "label": 0,
        "score": 0.8,
        "log_odds": 1.4,
    }
    assert join_predictions([pred], [row])[0]["firewall_score"] == 0.8
    for field, value in [("trajectory_sha256", "other"), ("label", 1)]:
        with pytest.raises(ValueError, match="identity"):
            join_predictions([pred | {field: value}], [row])
    with pytest.raises(ValueError, match="membership"):
        join_predictions([pred, pred], [row])


def test_fixed_bands_keep_high_delta_low_concept_counterexamples() -> None:
    rows = [
        {
            "index": str(i),
            "firewall_score": score,
            "firewall_log_odds": odds,
            "delta_z20": delta,
            "delta_cos20": delta / 10,
            "trained_z20": delta,
            "soft_target": 0.8,
            "prompt_tokens": 100,
        }
        for i, score, odds, delta in [
            (0, 0.01, -4, 6),
            (1, 0.99, 4, 2),
            (2, 0.99, 4, 1),
        ]
    ]
    stats = describe(rows)
    assert stats["top_delta_bands"]["0.1"]["flagged"]["0.5"] == 0
    assert stats["thresholds"]["0.5"]["unflagged"]["delta_z20"]["mean"] == 6
    assert stats["spearman_with_log_odds"]["delta_z20"] < 0
