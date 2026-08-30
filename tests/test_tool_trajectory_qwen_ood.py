from __future__ import annotations

from pathlib import Path

from experiments.tool_trajectory_monitoring.benchmark_qwen_ood import (
    QWEN_NON_THINKING_ASSISTANT_SUFFIX,
    render_margin_prompt,
    validate_config,
)
from experiments.tool_trajectory_monitoring.qwen_reasoning_core import load_json


class FakeTokenizer:
    def apply_chat_template(self, *args, **kwargs) -> str:
        assert args[0] == [{"role": "user", "content": "full teacher prompt"}]
        assert kwargs["enable_thinking"] is False
        assert kwargs["add_generation_prompt"] is True
        return f"system/user{QWEN_NON_THINKING_ASSISTANT_SUFFIX}"


def test_render_margin_prompt_uses_qwen_non_thinking_boundary() -> None:
    rendered = render_margin_prompt(
        FakeTokenizer(),
        "full teacher prompt",
        enable_thinking=False,
        assistant_suffix=QWEN_NON_THINKING_ASSISTANT_SUFFIX,
        decision_prefix="Prediction:",
    )
    assert rendered.endswith(
        f"{QWEN_NON_THINKING_ASSISTANT_SUFFIX}Prediction:"
    )


def test_frozen_qwen_config_uses_full_teacher_ood_contract() -> None:
    config = load_json(
        Path("experiments/tool_trajectory_monitoring/qwen_ood_benchmark.json")
    )
    validate_config(config)
    assert config["scope"]["rows"] == 6_395
    assert config["prompt"]["role"] == "teacher"
    assert config["prompt"]["enable_thinking"] is False
    assert config["engine"]["audited_max_prompt_tokens"] == 30_382
    assert config["engine"]["audited_total_prompt_tokens"] == 60_539_619
    assert config["engine"]["max_model_len"] == 32_768
