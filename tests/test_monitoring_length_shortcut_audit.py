from __future__ import annotations

import math

import pytest

from experiments.monitoring_length_shortcut_audit.core import (
    association_by_source,
    binary_metrics,
    length_matched_rows,
    materialize_counterfactual_rows,
    padded_prompt,
    paired_padding_summary,
    select_length_stratified,
)


def prompt_row(source: str, label: int, index: int, tokens: int) -> dict:
    prompt = "rubric\n<agent_trajectory>evidence</agent_trajectory>"
    return {
        "id": f"{source}-{label}-{index}",
        "prompt": prompt,
        "metadata": {
            "source_dataset": source,
            "ground_truth": label,
            "predicted_provider_prompt_tokens": tokens,
            "rendered_prompt_sha256": "original-hash",
        },
    }


def test_binary_metrics_handles_perfect_ranking_and_ties() -> None:
    perfect = binary_metrics([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9])
    assert perfect["auroc"] == pytest.approx(1.0)
    assert perfect["pauroc_at_20"] == pytest.approx(1.0)
    tied = binary_metrics([0, 1], [0.5, 0.5])
    assert tied["auroc"] == pytest.approx(0.5)
    assert tied["pauroc_at_20"] == pytest.approx(0.1)


def test_length_stratified_selection_is_balanced_and_deterministic() -> None:
    rows = [
        prompt_row(source, label, index, 100 + index)
        for source in ("a", "b")
        for label in (0, 1)
        for index in range(40)
    ]
    first = select_length_stratified(
        rows,
        quartiles=4,
        rows_per_stratum=2,
        maximum_tokens=1_000,
        seed=7,
    )
    second = select_length_stratified(
        rows,
        quartiles=4,
        rows_per_stratum=2,
        maximum_tokens=1_000,
        seed=7,
    )
    assert [row["id"] for row in first] == [row["id"] for row in second]
    counts = {}
    for row in first:
        key = (
            row["metadata"]["source_dataset"],
            row["metadata"]["ground_truth"],
            row["audit_length_quartile"],
        )
        counts[key] = counts.get(key, 0) + 1
    assert set(counts.values()) == {2}
    assert len(first) == 32


def test_padding_preserves_trajectory_and_materializes_conditions() -> None:
    row = prompt_row("a", 0, 0, 100)
    padded = padded_prompt(
        row["prompt"], marker="<agent_trajectory>", line="ignore", repetitions=3
    )
    assert padded.endswith("<agent_trajectory>evidence</agent_trajectory>")
    assert padded.count("ignore") == 3
    materialized = materialize_counterfactual_rows(
        [{**row, "audit_length_quartile": 0}],
        marker="<agent_trajectory>",
        line="ignore",
        conditions=[
            {"name": "original", "repetitions": 0},
            {"name": "pad", "repetitions": 3},
        ],
    )
    assert [item["id"] for item in materialized] == [
        "a-0-0::original",
        "a-0-0::pad",
    ]
    assert materialized[0]["prompt"] == row["prompt"]
    assert materialized[1]["metadata"]["audit_base_id"] == row["id"]


def test_length_matching_balances_each_source() -> None:
    rows = [
        {
            "id": f"{source}-{label}-{index}",
            "source": source,
            "label": label,
            "tokens": 100 * (index // 2) + label,
            "score": 0.2 + 0.6 * label,
        }
        for source in ("a", "b")
        for label in (0, 1)
        for index in range(8)
    ]
    matched = length_matched_rows(rows, bins=4)
    for source in ("a", "b"):
        subset = [row for row in matched if row["source"] == source]
        assert len(subset) == 16
        assert sum(row["label"] for row in subset) == 8
        for bin_index in range(4):
            length_bin = subset[bin_index * 4 : (bin_index + 1) * 4]
            assert sum(row["label"] for row in length_bin) == 2


def test_padding_summary_applies_frozen_materiality_thresholds() -> None:
    predictions = []
    for index, original in enumerate((0.1, 0.4, 0.6, 0.9)):
        predictions.extend(
            [
                {
                    "id": f"row-{index}::original",
                    "score": original,
                    "prompt_tokens": 100,
                },
                {
                    "id": f"row-{index}::pad",
                    "score": min(1.0, original + 0.03),
                    "prompt_tokens": 120,
                },
            ]
        )
    summary = paired_padding_summary(
        predictions,
        conditions=["original", "pad"],
        thresholds={
            "mean_absolute_score_difference": 0.02,
            "absolute_mean_signed_score_difference": 0.01,
            "p95_absolute_score_difference": 0.10,
            "threshold_flip_fraction": 0.02,
        },
    )["pad"]
    assert summary["material"] is True
    assert summary["mean_added_prompt_tokens"] == 20
    assert summary["mean_absolute_score_difference"] == pytest.approx(0.03)
    assert math.isfinite(summary["pearson_with_original"])


def test_association_reports_source_specific_direction() -> None:
    rows = [
        {"source": "a", "label": 0, "tokens": 1, "score": 0.1},
        {"source": "a", "label": 0, "tokens": 2, "score": 0.2},
        {"source": "a", "label": 1, "tokens": 3, "score": 0.8},
        {"source": "a", "label": 1, "tokens": 4, "score": 0.9},
        {"source": "b", "label": 0, "tokens": 4, "score": 0.1},
        {"source": "b", "label": 0, "tokens": 3, "score": 0.2},
        {"source": "b", "label": 1, "tokens": 2, "score": 0.8},
        {"source": "b", "label": 1, "tokens": 1, "score": 0.9},
    ]
    report = association_by_source(rows, value_key="score")
    assert report["sources"]["a"]["length_only_label_auroc"] == 1.0
    assert report["sources"]["b"]["length_only_label_auroc"] == 0.0
    assert report["pooled"]["length_only_label_auroc"] == 0.5
