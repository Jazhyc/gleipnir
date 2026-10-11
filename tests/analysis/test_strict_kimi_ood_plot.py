"""Fail closed on mismatched populations and distinguish paired agreement."""

import copy

import pytest

from scripts.plot_strict_kimi_ood import describe, pair_scores


def samples():
    student = [
        {
            "id": str(i),
            "ground_truth": i,
            "source_dataset": "test",
            "trajectory_sha256": str(i),
            "score": float(i),
        }
        for i in (0, 1)
    ]
    teacher = [
        {
            "id": r["id"],
            "score": r["score"],
            "metadata": {
                k: r[k] for k in ("ground_truth", "source_dataset", "trajectory_sha256")
            },
        }
        for r in student
    ]
    return student, teacher


def test_identity_join_is_independent_of_cache_order():
    student, teacher = samples()
    rows = pair_scores(student, teacher[::-1])
    metrics = describe(rows)
    assert metrics["agreement_at_05"] == 1
    assert metrics["mean_absolute_score_difference"] == 0


@pytest.mark.parametrize(
    "field", ["ground_truth", "source_dataset", "trajectory_sha256"]
)
def test_label_source_or_trajectory_drift_fails(field):
    student, teacher = samples()
    teacher[0]["metadata"][field] = "changed"
    with pytest.raises(ValueError, match="drift"):
        pair_scores(student, teacher)


def test_duplicate_or_missing_teacher_ids_fail():
    student, teacher = samples()
    for bad in (teacher[:-1], teacher + [copy.deepcopy(teacher[0])]):
        with pytest.raises(ValueError, match="coverage"):
            pair_scores(student, bad)


def test_matching_marginal_histograms_do_not_imply_paired_agreement():
    student, teacher = samples()
    teacher[0]["score"], teacher[1]["score"] = 1.0, 0.0
    metrics = describe(pair_scores(student, teacher))
    assert metrics["histogram_overlap_bin_width_005"] == 1
    assert metrics["agreement_at_05"] == 0
    assert metrics["mean_absolute_score_difference"] == 1
