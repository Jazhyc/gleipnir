from __future__ import annotations

import pytest

from experiments.tool_trajectory_monitoring.prepare_teacher_training_cache import (
    add_token_lengths,
    extract_trajectory,
    length_summary,
    validate_counts,
)


class LengthTokenizer:
    def __call__(
        self,
        texts: list[str],
        *,
        add_special_tokens: bool,
        return_length: bool,
    ) -> dict[str, list[int]]:
        assert add_special_tokens is False
        assert return_length is True
        return {"length": [len(text.split()) for text in texts]}


def test_extract_trajectory_requires_frozen_three_message_shape() -> None:
    messages = [
        {"role": "system", "content": "monitor"},
        {"role": "user", "content": "[USER]\nDo work\n[ASSISTANT]\nbash({})"},
        {"role": "assistant", "content": "rationale"},
    ]
    assert extract_trajectory(messages) == messages[1]["content"]
    with pytest.raises(ValueError, match="roles"):
        extract_trajectory(messages[:2])


def test_count_validation_preserves_source_label_contract() -> None:
    rows = [
        {"source_dataset": "a", "ground_truth": 0},
        {"source_dataset": "a", "ground_truth": 1},
    ]
    validate_counts(
        rows,
        source_key="source_dataset",
        expected={"a": {0: 1, 1: 1}},
    )
    with pytest.raises(ValueError, match="counts drifted"):
        validate_counts(
            rows[:1],
            source_key="source_dataset",
            expected={"a": {0: 1, 1: 1}},
        )


def test_token_lengths_and_summary_are_exact() -> None:
    rows = [{"text": "one two"}, {"text": "three four five"}]
    add_token_lengths(
        rows,
        LengthTokenizer(),
        text_key="text",
        output_key="tokens",
        batch_size=1,
    )
    assert [row["tokens"] for row in rows] == [2, 3]
    summary = length_summary([2, 3])
    assert summary["total"] == 5
    assert summary["mean"] == 2.5
    assert summary["max"] == 3
    with pytest.raises(ValueError, match="empty"):
        length_summary([])
