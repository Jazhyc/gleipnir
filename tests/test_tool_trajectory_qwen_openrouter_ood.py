from __future__ import annotations

from pathlib import Path

import pytest

from experiments.tool_trajectory_monitoring.benchmark_qwen_openrouter_ood import (
    completion_payload,
    parse_completion_response,
    request_settings,
    validate_config,
)
from experiments.tool_trajectory_monitoring.qwen_reasoning_core import load_json


def frozen_config() -> dict:
    return load_json(
        Path(
            "experiments/tool_trajectory_monitoring/"
            "qwen27b_compact_openrouter_ood_benchmark.json"
        )
    )


def test_frozen_openrouter_config_is_one_token_compact_ablation() -> None:
    config = frozen_config()
    validate_config(config)
    assert config["prompt"]["role"] == "student"
    assert config["request"]["max_tokens"] == 8
    assert config["request"]["allow_fallbacks"] is False
    assert config["request"]["provider_only"] == "Alibaba"
    assert config["scope"]["rows"] == 6_395


def test_chat_payload_keeps_rubric_as_the_only_message() -> None:
    config = frozen_config()
    payload = completion_payload("compact rubric and trajectory", config)
    assert payload["messages"] == [
        {"role": "user", "content": "compact rubric and trajectory"}
    ]
    assert payload["max_tokens"] == 8
    assert payload["logprobs"] is True
    assert payload["top_logprobs"] == 5
    assert payload["provider"]["only"] == ["Alibaba"]
    assert payload["provider"]["allow_fallbacks"] is False
    assert "endpoint" not in payload
    assert set(request_settings(config)) - {"endpoint"} <= payload.keys()


def test_parse_completion_requires_one_binary_token_and_both_logprobs() -> None:
    response = {
        "choices": [
            {
                "message": {"content": "Prediction:1"},
                "logprobs": {
                    "content": [
                        {"token": "Prediction", "top_logprobs": []},
                        {"token": ":", "top_logprobs": []},
                        {
                            "token": "1",
                            "top_logprobs": [
                                {"token": "0", "logprob": -2.0},
                                {"token": "1", "logprob": -0.2},
                            ],
                        }
                    ],
                },
            }
        ]
    }
    assert parse_completion_response(response, [15, 16]) == (
        "1",
        -2.0,
        -0.2,
        "Prediction:1",
        {"0": -2.0, "1": -0.2},
        2,
    )
    response["choices"][0]["message"]["content"] = "Analysis. Prediction:1"
    with pytest.raises(RuntimeError, match="only a terminal binary prediction"):
        parse_completion_response(response, [15, 16])


def test_parse_completion_rejects_missing_counterfactual_logprob() -> None:
    response = {
        "choices": [
            {
                "message": {"content": "Prediction:1"},
                "logprobs": {
                    "content": [
                        {"token": "Prediction", "top_logprobs": []},
                        {"token": ":", "top_logprobs": []},
                        {
                            "token": "1",
                            "top_logprobs": [
                                {"token": "1", "logprob": -0.2}
                            ],
                        }
                    ],
                },
            }
        ]
    }
    with pytest.raises(RuntimeError, match="omitted literal logprobs"):
        parse_completion_response(response, [15, 16])
