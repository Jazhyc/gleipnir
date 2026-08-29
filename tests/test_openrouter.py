from __future__ import annotations

from dataclasses import replace

import pytest

from gleipnir.openrouter import (
    OpenRouterConfig,
    OpenRouterError,
    PromptRecord,
    binary_scalar_response_format,
    binary_score_from_top_logprobs,
    extract_scalar_binary_top_logprobs,
    extract_terminal_binary_top_logprobs,
    request_payload,
    request_settings_sha256,
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


def test_extracts_scalar_label_row() -> None:
    payload = response_payload()
    payload["choices"][0]["message"]["content"] = "1"
    payload["choices"][0]["finish_reason"] = "length"
    payload["choices"][0]["logprobs"]["content"] = [
        payload["choices"][0]["logprobs"]["content"][-1]
    ]
    top, text, position = extract_scalar_binary_top_logprobs(payload)
    assert top == {"0": -2.0, "1": -0.2}
    assert text == "1"
    assert position == 0


def test_scalar_structured_request_and_price_cap_are_fingerprinted() -> None:
    config = OpenRouterConfig(
        model="moonshotai/kimi-k3",
        max_tokens=1,
        binary_output_mode="scalar",
        structured_output=True,
        provider_max_prompt_price=2.55,
        provider_max_completion_price=12.75,
    )
    payload = request_payload(PromptRecord("row-1", "Prompt\nPrediction:", {}), config)
    assert payload["response_format"] == binary_scalar_response_format()
    assert payload["provider"]["max_price"] == {
        "prompt": 2.55,
        "completion": 12.75,
    }
    assert request_settings_sha256(config) != request_settings_sha256(
        replace(config, structured_output=False)
    )


def test_structured_output_requires_scalar_mode_and_complete_price_cap() -> None:
    with pytest.raises(ValueError, match="requires binary_output_mode"):
        request_payload(
            PromptRecord("row-1", "Prompt", {}),
            OpenRouterConfig(model="qwen/test", structured_output=True),
        )
    with pytest.raises(ValueError, match="must both be positive"):
        request_payload(
            PromptRecord("row-1", "Prompt", {}),
            OpenRouterConfig(
                model="qwen/test",
                provider_max_prompt_price=2.55,
            ),
        )


def test_request_disables_provider_data_collection() -> None:
    record = PromptRecord("row-1", "Judge this.\nPrediction:<0 or 1>", {})
    config = OpenRouterConfig(
        model="qwen/test",
        provider_only="makora",
        session_id="teacher-campaign",
        cache_prefix="Judge this.\n",
    )
    payload = request_payload(record, config)
    assert payload["temperature"] == 0
    assert payload["provider"]["data_collection"] == "deny"
    assert payload["provider"]["enforce_distillable_text"] is True
    assert payload["provider"]["only"] == ["makora"]
    assert payload["session_id"] == "teacher-campaign"
    assert payload["messages"][0]["content"] == [
        {
            "type": "text",
            "text": "Judge this.\n",
            "cache_control": {"type": "ephemeral"},
        },
        {"type": "text", "text": "Prediction:<0 or 1>"},
    ]


def test_provider_order_overrides_sort_and_is_fingerprinted() -> None:
    config = OpenRouterConfig(
        model="moonshotai/kimi-k3",
        provider_order=("makora", "fireworks"),
        session_id="cache-session",
    )
    payload = request_payload(PromptRecord("row-1", "Prompt", {}), config)
    assert payload["provider"]["order"] == ["makora", "fireworks"]
    assert "sort" not in payload["provider"]
    assert request_settings_sha256(config) != request_settings_sha256(
        replace(config, session_id="other-session")
    )


def test_cache_prefix_must_match_prompt() -> None:
    config = OpenRouterConfig(model="qwen/test", cache_prefix="Other")
    with pytest.raises(ValueError, match="does not start"):
        request_payload(PromptRecord("row-1", "Prompt", {}), config)


def test_score_prompt_keeps_api_key_out_of_artifact() -> None:
    record = PromptRecord("row-1", "Judge this.\nPrediction:<0 or 1>", {"split": "dev"})
    config = replace(OpenRouterConfig(model="qwen/test"), max_retries=0)
    session = FakeSession()
    result = score_prompt(record, config, "secret-key", session=session)
    assert result["id"] == "row-1"
    assert result["score"] > 0.8
    assert "secret-key" not in repr(result)
    assert session.calls[0][1]["Authorization"] == "Bearer secret-key"
    assert session.calls[0][1]["X-OpenRouter-Metadata"] == "enabled"
    assert result["request_settings_sha256"] == request_settings_sha256(config)
    assert result["cache_usage"] == {
        "cached_tokens": 0,
        "cache_write_tokens": 0,
        "cache_discount": None,
    }
