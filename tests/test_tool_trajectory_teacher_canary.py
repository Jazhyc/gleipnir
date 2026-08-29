from __future__ import annotations

import hashlib

import pandas as pd
import pytest

from experiments.tool_trajectory_monitoring.teacher_canary import (
    select_balanced_candidates,
    stratified_bootstrap_auroc,
    summarize_scored_rows,
)


def candidates() -> list[dict]:
    return [
        {
            "id": f"{source}-{label}-{index}",
            "source_dataset": source,
            "ground_truth": label,
        }
        for source in ("a", "b")
        for label in (0, 1)
        for index in range(5)
    ]


def test_balanced_selection_is_deterministic_and_interleaved() -> None:
    first = select_balanced_candidates(
        candidates(), total_rows=12, seed=7, sources=("a", "b")
    )
    second = select_balanced_candidates(
        candidates(), total_rows=12, seed=7, sources=("a", "b")
    )
    assert [row["id"] for row in first] == [row["id"] for row in second]
    assert [
        (row["source_dataset"], row["ground_truth"]) for row in first[:4]
    ] == [("a", 0), ("a", 1), ("b", 0), ("b", 1)]
    with pytest.raises(ValueError, match="multiple"):
        select_balanced_candidates(
            candidates(), total_rows=10, seed=7, sources=("a", "b")
        )


def test_stratified_bootstrap_recovers_perfect_auc() -> None:
    frame = pd.DataFrame(
        {
            "source": ["a", "a", "b", "b"],
            "label": [0, 1, 0, 1],
            "score": [0.1, 0.9, 0.2, 0.8],
        }
    )
    interval = stratified_bootstrap_auroc(frame, resamples=50, seed=1)
    assert interval["low"] == interval["median"] == interval["high"] == 1.0


def test_summary_checks_prompt_identity_and_reports_auc() -> None:
    inputs = []
    scores = []
    for index, (source, label, score) in enumerate(
        (("a", 0, 0.1), ("a", 1, 0.9), ("b", 0, 0.2), ("b", 1, 0.8))
    ):
        record_id = str(index)
        prompt = f"prompt-{index}"
        prompt_sha256 = hashlib.sha256(prompt.encode()).hexdigest()
        inputs.append(
            {
                "id": record_id,
                "prompt": prompt,
                "metadata": {
                    "source_dataset": source,
                    "ground_truth": label,
                    "rendered_prompt_sha256": prompt_sha256,
                    "prompt_template_sha256": "template",
                    "cache_prefix_chars": 3,
                    "cache_prefix_sha256": "prefix",
                },
            }
        )
        scores.append(
            {
                "id": record_id,
                "prompt_sha256": prompt_sha256,
                "score": score,
                "text": f"Prediction:{label}",
                "provider": "Makora",
                "model": "moonshotai/kimi-k3",
                "usage": {"prompt_tokens": 100, "completion_tokens": 3, "cost": 0.01},
                "cache_usage": {"cached_tokens": 80, "cache_write_tokens": 20},
                "request_settings_sha256": "settings",
            }
        )

    summary = summarize_scored_rows(
        inputs,
        scores,
        bootstrap_resamples=50,
        seed=1,
    )
    assert summary["overall"]["auroc"] == 1.0
    assert summary["generated_label_accuracy"] == 1.0
    assert summary["generated_label_auroc"] == 1.0
    assert summary["mean_source_auroc"] == 1.0
    assert summary["usage"] == {
        "prompt_tokens": 400,
        "completion_tokens": 12,
        "cached_tokens": 320,
        "cache_write_tokens": 80,
        "cache_read_fraction": 0.8,
        "reported_cost_usd": 0.04,
    }
    assert summary["request_setting_groups"] == [
        {
            "request_settings_sha256": "settings",
            "rows": 4,
            "max_tokens": None,
            "prompt_tokens": 400,
            "cached_tokens": 320,
            "reported_cost_usd": 0.04,
        }
    ]
