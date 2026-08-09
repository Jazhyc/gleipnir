from __future__ import annotations

from dataclasses import replace

import pytest

from gleipnir.openrouter import (
    OpenRouterConfig,
    OpenRouterError,
    PromptRecord,
    binary_score_from_top_logprobs,
    extract_terminal_binary_top_logprobs,
    request_payload,
    score_prompt,
)


def response_payload() -> dict:
    return {
        "id": "response-1",
        "model": "qwen/test",
        "provider": "provider",
        "usage": {"prompt_tokens": 5, "completion_tokens": 3},
        "choices": [
            {
                "finish_reason": "stop",
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
                        },
                    ]
                },
            }
        ],
    }


class FakeResponse:
    status_code = 200
    text = ""
    headers = {}

    def json(self) -> dict:
        return response_payload()


class FakeSession:
    def __init__(self) -> None:
        self.calls = []

    def post(self, endpoint, *, headers, json, timeout):
        self.calls.append((endpoint, headers, json, timeout))
        return FakeResponse()


def test_binary_score_normalizes_literal_tokens() -> None:
    score, logprobs = binary_score_from_top_logprobs({"0": -2.0, "1": -0.2})
    assert score > 0.8
    assert logprobs == {"0": -2.0, "1": -0.2}
    with pytest.raises(OpenRouterError, match="omitted"):
        binary_score_from_top_logprobs({"1": -0.2})


def test_extracts_terminal_label_row() -> None:
    top, text, position = extract_terminal_binary_top_logprobs(response_payload())
    assert top == {"0": -2.0, "1": -0.2}
    assert text == "Prediction:1"
    assert position == 2


def test_request_disables_provider_data_collection() -> None:
    record = PromptRecord("row-1", "Judge this.\nPrediction:<0 or 1>", {})
    config = OpenRouterConfig(model="qwen/test", provider_only="Provider")
    payload = request_payload(record, config)
    assert payload["temperature"] == 0
    assert payload["provider"]["data_collection"] == "deny"
    assert payload["provider"]["only"] == ["Provider"]


def test_score_prompt_keeps_api_key_out_of_artifact() -> None:
    record = PromptRecord("row-1", "Judge this.\nPrediction:<0 or 1>", {"split": "dev"})
    config = replace(OpenRouterConfig(model="qwen/test"), max_retries=0)
    session = FakeSession()
    result = score_prompt(record, config, "secret-key", session=session)
    assert result["id"] == "row-1"
    assert result["score"] > 0.8
    assert "secret-key" not in repr(result)
    assert session.calls[0][1]["Authorization"] == "Bearer secret-key"
