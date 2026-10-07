import json

import pyarrow.parquet as pq
import pytest

from experiments.tool_trajectory_monitoring.export_teacher_cache_hf import (
    export_public_cache,
    load_cache_rows,
)


def cache_row(
    record_id: str,
    provider: str = "Makora",
    text: str = "Prediction:1",
) -> dict:
    return {
        "id": record_id,
        "metadata": {
            "ground_truth": 1,
            "source_dataset": "stride",
            "source_repo": "source/repo",
            "source_revision": "revision",
            "source_file": "train.parquet",
            "source_row_index": 4,
            "prompt_id": 3,
            "original_prompt_id": 2,
            "sample_index": 1,
            "raw_source": "ctrl",
            "trajectory_sha256": "a" * 64,
            "rendered_prompt_sha256": "b" * 64,
            "prompt_template_sha256": "c" * 64,
        },
        "label_logprobs": {"0": -2.0, "1": -0.2},
        "target_probs": {"negative": 0.2, "positive": 0.8},
        "score": 0.8,
        "text": text,
        "top_logprobs": {"0": -2.0, "1": -0.2},
        "model": "moonshotai/kimi-k3",
        "provider": provider,
        "request_settings_sha256": "d" * 64,
        "request_settings": {"max_tokens": 64},
        "usage": {"prompt_tokens": 100, "completion_tokens": 10},
        "cache_usage": {"cached_tokens": 0},
        "response_id": "private-response-id",
    }


def test_export_public_cache_removes_labels_prompts_and_response_ids(tmp_path) -> None:
    scores = tmp_path / "scores.jsonl"
    scores.write_text(
        json.dumps(cache_row("two", text="Brief refusal.\nPrediction: 1"))
        + "\n"
        + json.dumps(cache_row("one"))
        + "\n"
    )
    summary = tmp_path / "summary.json"
    summary.write_text("{}\n")
    card = tmp_path / "card.md"
    card.write_text("# Card\n")

    manifest = export_public_cache([scores], summary, card, tmp_path / "export")
    rows = pq.read_table(tmp_path / "export" / "train.parquet").to_pylist()

    assert [row["id"] for row in rows] == ["one", "two"]
    assert "ground_truth" not in rows[0]
    assert "response_id" not in rows[0]
    assert "prompt" not in rows[0]
    assert rows[0]["logprob_1"] == pytest.approx(-0.2)
    assert rows[1]["generated_label"] == 1
    assert manifest["contains_prompts_or_trajectories"] is False
    assert manifest["contains_ground_truth_labels"] is False


def test_load_cache_rows_rejects_cross_shard_duplicates(tmp_path) -> None:
    left = tmp_path / "left.jsonl"
    right = tmp_path / "right.jsonl"
    left.write_text(json.dumps(cache_row("same")) + "\n")
    right.write_text(json.dumps(cache_row("same", "Morph")) + "\n")

    with pytest.raises(ValueError, match="duplicate id"):
        load_cache_rows([left, right])
