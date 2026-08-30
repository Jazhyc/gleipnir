from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from experiments.tool_trajectory_monitoring.benchmark_gpt_oss_ood import (
    balanced_canary_rows,
    binary_logprobs,
    normalized_score,
    render_margin_prompt,
    validate_config,
)
from experiments.tool_trajectory_monitoring.qwen_reasoning_core import load_json


class FakeTokenizer:
    def apply_chat_template(self, *args, **kwargs) -> str:
        assert kwargs["reasoning_effort"] == "low"
        assert kwargs["add_generation_prompt"] is True
        return "system/user<|start|>assistant"


def test_render_margin_prompt_forces_harmony_final_boundary() -> None:
    rendered = render_margin_prompt(
        FakeTokenizer(),
        "full teacher prompt",
        reasoning_effort="low",
        assistant_boundary="<|channel|>final<|message|>",
        decision_prefix="Prediction:",
    )
    assert rendered.endswith(
        "<|start|>assistant<|channel|>final<|message|>Prediction:"
    )


def test_balanced_canary_has_every_source_label_stratum() -> None:
    sources = ["a", "b"]
    rows = [
        {
            "id": f"{source}-{label}-{index}",
            "metadata": {"source_dataset": source, "ground_truth": label},
        }
        for source in sources
        for label in (0, 1)
        for index in range(2)
    ]
    selected = balanced_canary_rows(
        rows, sources, rows_per_source_label=1
    )
    assert [row["id"] for row in selected] == [
        "a-0-0",
        "a-1-0",
        "b-0-0",
        "b-1-0",
    ]
    with pytest.raises(ValueError, match="too few"):
        balanced_canary_rows(rows, sources, rows_per_source_label=3)


def test_binary_logprob_normalization_retains_both_values() -> None:
    output = SimpleNamespace(
        outputs=[
            SimpleNamespace(
                logprobs=[
                    {
                        15: SimpleNamespace(logprob=-2.0),
                        16: SimpleNamespace(logprob=-1.0),
                    }
                ]
            )
        ]
    )
    logprob_0, logprob_1 = binary_logprobs(output, [15, 16])
    assert (logprob_0, logprob_1) == (-2.0, -1.0)
    assert normalized_score(logprob_0, logprob_1) == pytest.approx(0.7310585786)


def test_frozen_gpt_oss_config_uses_full_teacher_ood_contract() -> None:
    config = load_json(
        Path(
            "experiments/tool_trajectory_monitoring/gpt_oss_ood_benchmark.json"
        )
    )
    validate_config(config)
    assert config["scope"]["rows"] == 6_395
    assert config["prompt"]["role"] == "teacher"
    assert config["engine"]["audited_max_prompt_tokens"] == 27_473
    assert config["engine"]["max_model_len"] == 32_768
