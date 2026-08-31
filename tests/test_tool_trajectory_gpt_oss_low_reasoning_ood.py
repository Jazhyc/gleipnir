from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from experiments.tool_trajectory_monitoring.benchmark_gpt_oss_low_reasoning_ood import (
    audit_source_prompts,
    build_margin_token_ids,
    harmony_token_contract,
    pack_generation,
    render_reasoning_prompt,
    validate_config,
)
from experiments.tool_trajectory_monitoring.qwen_reasoning_core import load_json


class FakeTokenizer:
    chat_template = 'date={{ strftime_now("%Y-%m-%d") }}'
    values = {
        "<|channel|>analysis<|message|>": [5, 6, 7],
        "<|end|>": [8],
        "<|start|>assistant<|channel|>final<|message|>Prediction:": [9, 10],
    }

    def apply_chat_template(self, *args, **kwargs) -> str:
        assert kwargs["reasoning_effort"] == "low"
        assert kwargs["add_generation_prompt"] is True
        assert '"2026-08-30"' in kwargs["chat_template"]
        return f"{args[0][0]['content']}<|start|>assistant"

    def encode(self, value: str, *, add_special_tokens: bool) -> list[int]:
        assert add_special_tokens is False
        return list(self.values.get(value, range(len(value))))


def test_render_reasoning_prompt_leaves_native_assistant_boundary() -> None:
    rendered = render_reasoning_prompt(
        FakeTokenizer(),
        "full teacher prompt",
        reasoning_effort="low",
        current_date="2026-08-30",
    )
    assert rendered.endswith("<|start|>assistant")


def test_harmony_contract_and_margin_tokens_preserve_reasoning() -> None:
    prompt = {
        "analysis_prefix": "<|channel|>analysis<|message|>",
        "analysis_stop": "<|end|>",
        "final_boundary": (
            "<|start|>assistant<|channel|>final<|message|>Prediction:"
        ),
    }
    contract = harmony_token_contract(FakeTokenizer(), prompt)
    assert contract == {
        "analysis_prefix_ids": [5, 6, 7],
        "analysis_stop_id": 8,
        "final_boundary_ids": [9, 10],
    }
    assert build_margin_token_ids([1, 2], [42], contract) == [
        1,
        2,
        5,
        6,
        7,
        42,
        8,
        9,
        10,
    ]


def test_pack_generation_requires_native_analysis_and_end_stop() -> None:
    record = {
        "id": "row-1",
        "metadata": {"source_dataset": "source", "ground_truth": 1},
    }
    output = SimpleNamespace(
        outputs=[
            SimpleNamespace(
                token_ids=[42],
                text="analysis",
                finish_reason="stop",
                stop_reason=8,
            )
        ]
    )
    row = pack_generation(
        record,
        "prompt",
        [1, 2],
        output,
        {
            "analysis_prefix_ids": [5, 6, 7],
            "analysis_stop_id": 8,
        },
        "config-hash",
    )
    assert row["valid"] is True
    assert row["analysis_content_tokens"] == 1
    assert row["generation_token_ids"] == [42]

    output.outputs[0].finish_reason = "length"
    output.outputs[0].stop_reason = None
    invalid = pack_generation(
        record,
        "prompt",
        [1, 2],
        output,
        {
            "analysis_prefix_ids": [5, 6, 7],
            "analysis_stop_id": 8,
        },
        "config-hash",
    )
    assert invalid["valid"] is False
    assert invalid["format_errors"] == ["did_not_stop_at_harmony_end"]


def test_frozen_low_reasoning_config_covers_full_ood_contract() -> None:
    config = load_json(
        Path(
            "experiments/tool_trajectory_monitoring/"
            "gpt_oss_low_reasoning_ood_benchmark.json"
        )
    )
    validate_config(config)
    assert config["scope"]["rows"] == 6_395
    assert config["prompt"]["reasoning_effort_header"] == "low"
    assert config["prompt"]["current_date"] == "2026-08-30"
    assert config["generation"]["temperature"] == 1.0
    assert config["engine"]["max_model_len"] == 65_536


def test_source_prompt_audit_fails_closed_on_tokenizer_drift() -> None:
    records = [{"prompt": "abc"}, {"prompt": "abcdef"}]
    config = {
        "prompt": {
            "reasoning_effort_header": "low",
            "current_date": "2026-08-30",
        },
        "engine": {"audited_max_source_prompt_tokens": 24},
    }
    assert audit_source_prompts(FakeTokenizer(), records, config) == {
        "rows": 2,
        "minimum": 21,
        "maximum": 24,
    }
    config["engine"]["audited_max_source_prompt_tokens"] = 25
    try:
        audit_source_prompts(FakeTokenizer(), records, config)
    except ValueError as error:
        assert "differs from frozen audit" in str(error)
    else:
        raise AssertionError("tokenizer drift should fail the frozen audit")
