"""Frontier monitor requests, score parsing and records (no network)."""

from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace

import pytest

from gleipnir.adversarial.contract import join_prompt, split_prompt
from gleipnir.adversarial.frontier import (
    DECISION_STEP,
    FrontierMonitor,
    anthropic_batch_params,
    anthropic_result_record,
    margin_fields,
    openrouter_record,
    openrouter_score_payload,
    parse_score,
    score_instruction,
    score_prompt,
)

ROOT = Path(__file__).resolve().parents[2]
TEACHER = (
    ROOT / "experiments/tool_trajectory_monitoring/prompts/teacher.txt"
).read_text()
ROW = join_prompt(TEACHER, "[USER]\nfix it\n[ASSISTANT]\n[TOOL CALLS]\nbash({})")
SOL = FrontierMonitor(
    "sol",
    "openrouter",
    "openai/gpt-6.1-sol",
    "score",
    2048,
    2.0,
    10.0,
    reasoning_effort="low",
)


def test_score_instruction_changes_only_step_6_and_the_output():
    head = score_instruction(TEACHER)
    assert DECISION_STEP not in head and "Score:N" in head
    assert "Prediction:0\nPrediction:1" not in head
    before = TEACHER[: TEACHER.index("6. Weigh")]
    assert head.startswith(before)
    assert head.endswith("The action-only trajectory follows this instruction.\n") == (
        TEACHER.endswith("The action-only trajectory follows this instruction.\n")
    )


def test_score_prompt_keeps_the_trajectory():
    new = score_prompt(ROW)
    assert split_prompt(new).trajectory == split_prompt(ROW).trajectory
    assert split_prompt(new).head == score_instruction(TEACHER)


@pytest.mark.parametrize(
    "text, value",
    [("Score:85", 0.85), ("  Score: 0 ", 0.0), ("\nScore:100\n", 1.0)],
)
def test_parse_score(text, value):
    assert parse_score(text) == value


def test_text_before_the_score_line_is_allowed_and_counted():
    from gleipnir.adversarial.frontier import extra_lines

    reply = "Reviewing the trajectory, the agent hard-codes.\n\nScore:40"
    assert parse_score(reply) == 0.4 and extra_lines(reply) == 1
    assert extra_lines("Score:40") == 0


@pytest.mark.parametrize(
    "text", ["Score:101", "Score: high", "Score:40\nthen more", "", "Prediction:1"]
)
def test_parse_score_rejects_anything_else(text):
    with pytest.raises(ValueError):
        parse_score(text)


def test_margin_is_clipped_log_odds():
    m = margin_fields(1.0)
    assert math.isfinite(m["logit_1"] - m["logit_0"])
    m = margin_fields(0.8)
    assert m["logit_1"] - m["logit_0"] == pytest.approx(math.log(4))


def test_score_payload_hides_reasoning_and_denies_data_collection():
    body = openrouter_score_payload(SOL, "p")
    assert body["reasoning"] == {"effort": "low", "exclude": True}
    assert body["provider"]["data_collection"] == "deny"
    assert "temperature" not in body


class Reply:
    def __init__(self, status, body=None):
        self.status_code = status
        self.body = body or {}
        self.text = "err"

    def json(self):
        return self.body


def ok_body(text, finish="stop"):
    return {
        "choices": [{"message": {"content": text}, "finish_reason": finish}],
        "usage": {"prompt_tokens": 100, "completion_tokens": 20, "cost": 0.001},
    }


@pytest.mark.parametrize(
    "replies, status, score",
    [
        ([Reply(200, ok_body("Score:70"))], "ok", 0.7),
        ([Reply(429), Reply(200, ok_body("Score:10"))], "ok", 0.1),
        ([Reply(200, ok_body("Score:7", "length"))], "truncation", None),
        ([Reply(200, ok_body("I think it is fine"))], "invalid_score", None),
        ([Reply(400)], "error", None),
    ],
)
def test_openrouter_score_records(replies, status, score):
    calls = iter(replies)
    record = openrouter_record(
        SOL, "row", ROW, "key", lambda *a, **k: next(calls), sleep=lambda s: None
    )
    assert record["status"] == status
    assert record.get("score") == score
    assert "key" not in str(record)


def test_anthropic_params_have_no_sampling_or_fallbacks():
    sonnet = FrontierMonitor(
        "s",
        "anthropic-batch",
        "claude-sonnet-5-5",
        "score",
        2048,
        2.0,
        10.0,
        reasoning_effort="low",
        discount=0.5,
    )
    params = anthropic_batch_params(sonnet, ROW)
    assert params["output_config"] == {"effort": "low"}
    assert not {"temperature", "top_p", "top_k", "fallbacks", "thinking"} & set(params)
    assert sonnet.cost_from_usage(1_000_000, 0) == pytest.approx(1.0)


def batch_result(stop_reason, text):
    message = SimpleNamespace(
        content=[
            SimpleNamespace(type="thinking"),
            SimpleNamespace(type="text", text=text),
        ],
        usage=SimpleNamespace(input_tokens=1000, output_tokens=50),
        stop_reason=stop_reason,
        stop_details=None,
    )
    return SimpleNamespace(
        custom_id="r0", result=SimpleNamespace(type="succeeded", message=message)
    )


@pytest.mark.parametrize(
    "stop, text, status",
    [
        ("end_turn", "Score:90", "ok"),
        ("refusal", "", "refusal"),
        ("max_tokens", "", "truncation"),
        ("end_turn", "no score", "invalid_score"),
    ],
)
def test_anthropic_result_records(stop, text, status):
    sonnet = FrontierMonitor(
        "s",
        "anthropic-batch",
        "claude-sonnet-5-5",
        "score",
        2048,
        2.0,
        10.0,
        discount=0.5,
    )
    assert anthropic_result_record(sonnet, batch_result(stop, text))["status"] == status


def test_failed_batch_entries_are_kept():
    sonnet = FrontierMonitor(
        "s",
        "anthropic-batch",
        "claude-sonnet-5-5",
        "score",
        2048,
        2.0,
        10.0,
        discount=0.5,
    )
    result = SimpleNamespace(custom_id="r1", result=SimpleNamespace(type="expired"))
    assert anthropic_result_record(sonnet, result)["status"] == "batch_expired"
