from __future__ import annotations

import hashlib

import pytest

from experiments.tool_trajectory_monitoring.prompting import load_prompt_set
from experiments.tool_trajectory_monitoring.structured_output_canary import (
    compare_paired_scores,
    prepare_paired_prompts,
)


def input_rows() -> list[dict]:
    rows = []
    for index, (source, label) in enumerate(
        (("a", 0), ("a", 1), ("b", 0), ("b", 1))
    ):
        prompt = f"{load_prompt_set().teacher.cache_prefix}Prompt {index}\n"
        rows.append(
            {
                "id": str(index),
                "prompt": prompt,
                "metadata": {
                    "source_dataset": source,
                    "ground_truth": label,
                    "rendered_prompt_sha256": hashlib.sha256(
                        prompt.encode()
                    ).hexdigest(),
                },
            }
        )
    return rows


def score_rows(inputs: list[dict], scores: list[float]) -> list[dict]:
    rows = []
    for source, score in zip(inputs, scores, strict=True):
        rows.append(
            {
                "id": source["id"],
                "prompt_sha256": source["metadata"]["rendered_prompt_sha256"],
                "score": score,
                "text": str(int(score >= 0.5)),
                "label_logprobs": {"0": -score, "1": score},
                "provider": "Makora",
                "usage": {"cost": 0.01},
            }
        )
    return rows


def reference_rows(inputs: list[dict], scores: list[float]) -> list[dict]:
    rows = score_rows(inputs, scores)
    for source, row in zip(inputs, rows, strict=True):
        row["prompt_sha256"] = source["metadata"]["base_rendered_prompt_sha256"]
        row["text"] = f"Prediction:{row['text']}"
    return rows


def acceptance() -> dict:
    return {
        "complete_pairs": 4,
        "minimum_score_pearson": 0.999,
        "minimum_score_spearman": 0.999,
        "maximum_absolute_score_delta": 0.01,
        "maximum_absolute_auroc_delta": 0.01,
        "maximum_threshold_flips": 0,
        "required_provider": "Makora",
        "minimum_reference_spearman": 0.95,
        "maximum_reference_auroc_degradation": 0.05,
    }


def test_prepare_paired_prompts_preserves_policy_and_balanced_rows() -> None:
    original = input_rows()
    prepared = prepare_paired_prompts(original, total_rows=4)
    assert [row["id"] for row in prepared] == ["0", "1", "2", "3"]
    assert all("Return exactly one JSON integer" in row["prompt"] for row in prepared)
    assert all("Return exactly one line" not in row["prompt"] for row in prepared)
    assert prepared[0]["prompt"].endswith("Prompt 0\n")
    assert prepared[0]["metadata"]["base_rendered_prompt_sha256"] == (
        original[0]["metadata"]["rendered_prompt_sha256"]
    )
    assert prepared[0]["metadata"]["rendered_prompt_sha256"] != (
        original[0]["metadata"]["rendered_prompt_sha256"]
    )


def test_compare_paired_scores_accepts_close_scores() -> None:
    inputs = prepare_paired_prompts(input_rows(), total_rows=4)
    left = score_rows(inputs, [0.1, 0.8, 0.2, 0.9])
    right = score_rows(inputs, [0.101, 0.801, 0.201, 0.901])
    reference = reference_rows(inputs, [0.1, 0.8, 0.2, 0.9])
    summary = compare_paired_scores(
        inputs, reference, left, right, acceptance=acceptance()
    )
    assert summary["accepted"] is True
    assert summary["auroc"]["unstructured"] == 1.0
    assert summary["score_agreement"]["threshold_flips"] == 0


def test_compare_paired_scores_rejects_missing_pair() -> None:
    inputs = prepare_paired_prompts(input_rows(), total_rows=4)
    rows = score_rows(inputs, [0.1, 0.8, 0.2, 0.9])
    with pytest.raises(ValueError, match="every input ID"):
        compare_paired_scores(
            inputs,
            reference_rows(inputs, [0.1, 0.8, 0.2, 0.9]),
            rows[:-1],
            rows,
            acceptance=acceptance(),
        )
