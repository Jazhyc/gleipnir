from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from experiments.tool_trajectory_monitoring.benchmark_qwen_ood import (
    QWEN_NON_THINKING_ASSISTANT_SUFFIX,
)
from experiments.tool_trajectory_monitoring.benchmark_qwen_ordinal_ood import (
    compare_binary_baseline,
    extract_trajectory,
    prediction_row,
    render_generation_prompt,
    render_ordinal_user_prompt,
    validate_choice_tokenization,
    validate_config,
)
from experiments.tool_trajectory_monitoring.qwen_reasoning_core import load_json


class FakeTokenizer:
    def apply_chat_template(self, *args, **kwargs) -> str:
        assert args[0] == [{"role": "user", "content": "ordinal prompt"}]
        assert kwargs["enable_thinking"] is False
        assert kwargs["add_generation_prompt"] is True
        return f"system/user{QWEN_NON_THINKING_ASSISTANT_SUFFIX}"


class FakeChoiceTokenizer:
    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        assert add_special_tokens is False
        if text == "10":
            return [101, 100]
        return [100 + int(text)]


def test_extract_and_render_ordinal_trajectory_without_normalization() -> None:
    cache_prefix = "binary instruction\n<agent_trajectory>\n"
    binary_prompt = f"{cache_prefix}line one\nline two\n</agent_trajectory>\n"
    trajectory = extract_trajectory(binary_prompt, cache_prefix)
    assert trajectory == "line one\nline two"
    assert render_ordinal_user_prompt("ordinal instruction\n", trajectory) == (
        "ordinal instruction\n\n<agent_trajectory>\n"
        "line one\nline two\n</agent_trajectory>\n"
    )


def test_render_generation_prompt_starts_at_first_assistant_content() -> None:
    rendered = render_generation_prompt(
        FakeTokenizer(),
        "ordinal prompt",
        enable_thinking=False,
        assistant_suffix=QWEN_NON_THINKING_ASSISTANT_SUFFIX,
    )
    assert rendered.endswith(QWEN_NON_THINKING_ASSISTANT_SUFFIX)


def test_choice_tokenization_allows_only_two_token_ten() -> None:
    tokenization = validate_choice_tokenization(
        FakeChoiceTokenizer(), [str(score) for score in range(11)]
    )
    assert tokenization["9"] == [109]
    assert tokenization["10"] == [101, 100]


def test_prediction_row_retains_discrete_score_and_token_count() -> None:
    record = {
        "id": "row-1",
        "metadata": {
            "source_dataset": "source",
            "ground_truth": 1,
            "rendered_prompt_sha256": "source-hash",
        },
    }
    output = SimpleNamespace(
        prompt_token_ids=[1, 2, 3],
        outputs=[SimpleNamespace(text="10", token_ids=[101, 100])],
    )
    packed = prediction_row(
        record,
        "generation prompt",
        output,
        {str(score): [100 + score] for score in range(10)} | {"10": [101, 100]},
        999,
        "config-hash",
        32_768,
    )
    assert packed["ordinal_score"] == 10
    assert packed["score"] == 1.0
    assert packed["output_tokens"] == 2

    output.outputs[0].text = "11"
    with pytest.raises(RuntimeError, match="invalid constrained ordinal score"):
        prediction_row(
            record,
            "generation prompt",
            output,
            {str(score): [100 + score] for score in range(10)} | {"10": [101, 100]},
            999,
            "config-hash",
            32_768,
        )


def test_prediction_row_accepts_only_explicit_eos_after_one_digit() -> None:
    record = {
        "id": "row-0",
        "metadata": {
            "source_dataset": "source",
            "ground_truth": 0,
            "rendered_prompt_sha256": "source-hash",
        },
    }
    output = SimpleNamespace(
        prompt_token_ids=[1, 2, 3],
        outputs=[SimpleNamespace(text="0", token_ids=[100, 999])],
    )
    packed = prediction_row(
        record,
        "generation prompt",
        output,
        {str(score): [100 + score] for score in range(10)} | {"10": [101, 100]},
        999,
        "config-hash",
        32_768,
    )
    assert packed["score_token_ids"] == [100]
    assert packed["termination_token_ids"] == [999]
    assert packed["score_tokens"] == 1
    assert packed["output_tokens"] == 2


def test_ordinal_prompt_contains_paper_severity_bands() -> None:
    instruction = Path(
        "experiments/tool_trajectory_monitoring/prompts/teacher_ordinal.txt"
    ).read_text(encoding="utf-8")
    for band in ("0--2", "3--4", "5--6", "7--8", "9--10"):
        assert band in instruction
    assert "confidence" in instruction
    assert "severity" in instruction
    assert "concealment" in instruction


def test_frozen_ordinal_config_uses_primary_pauroc_contract() -> None:
    config = load_json(
        Path(
            "experiments/tool_trajectory_monitoring/"
            "qwen_ordinal_ood_benchmark.json"
        )
    )
    validate_config(config)
    assert config["scope"]["rows"] == 6_395
    assert config["scoring"]["choices"] == [str(score) for score in range(11)]
    assert config["engine"]["audited_max_prompt_tokens"] == 30_540
    assert config["engine"]["audited_total_prompt_tokens"] == 61_550_029
    assert config["metrics"][0].startswith("primary:")
    assert "pAUROC@20" in config["metrics"][0]


def test_binary_comparison_reports_source_and_quantization_deltas() -> None:
    ordinal_rows = []
    binary_rows = []
    for source in ("a", "b"):
        for index, (label, binary_score, ordinal_score) in enumerate(
            ((0, 0.04, 0), (0, 0.46, 4), (1, 0.54, 6), (1, 0.96, 10))
        ):
            common = {
                "id": f"{source}-{index}",
                "source": source,
                "label": label,
                "prompt_tokens": 10,
            }
            binary_rows.append({**common, "score": binary_score})
            ordinal_rows.append(
                {
                    **common,
                    "score": ordinal_score / 10,
                    "ordinal_score": ordinal_score,
                    "output_tokens": 2,
                }
            )
    comparison = compare_binary_baseline(ordinal_rows, binary_rows)
    assert set(comparison["delta_ordinal_minus_binary"]["by_dataset"]) == {
        "a",
        "b",
    }
    quantized = comparison["binary_nearest_tenth_control"]
    assert "quantization" in quantized["method"]
    assert quantized["metrics"]["macro"]["diagnostics"]["unique_scores"] == 3
