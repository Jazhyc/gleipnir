"""Focused tests for the Inkling decision boundary and its frozen configs."""

import json
from pathlib import Path

import pytest

from experiments.evaluation.inkling_ood.interface import (
    ASSISTANT_OPEN,
    MODEL_ID,
    REVISION,
    contains_special_marker,
    opening_prompt,
    render_margin_prompt,
)

HERE = Path(__file__).parent
ARMS = ("inkling_effort_none", "inkling_effort_high")


class FakeTokenizer:
    """Mimics the checkpoint template: an effort system line, the user turn,
    and nothing after it."""

    def __init__(self, tail: str = "") -> None:
        self.tail = tail

    def apply_chat_template(
        self, messages, *, tokenize, add_generation_prompt, reasoning_effort
    ):
        assert not tokenize
        level = {"none": "0", "high": "0.9"}[reasoning_effort]
        return (
            f"<|message_system|><|content_text|>Thinking effort level: {level}"
            f"<|end_message|><|message_user|><|content_text|>"
            f"{messages[0]['content']}<|end_message|>{self.tail}"
        )


def test_render_opens_text_answer_at_decision_prefix():
    rendered = render_margin_prompt(FakeTokenizer(), "judge", reasoning_effort="none")
    assert rendered == (
        "<|message_system|><|content_text|>Thinking effort level: 0<|end_message|>"
        "<|message_user|><|content_text|>judge<|end_message|>"
        "<|message_model|><|content_text|>Prediction:"
    )


def test_effort_changes_only_the_system_line():
    none = render_margin_prompt(FakeTokenizer(), "judge", reasoning_effort="none")
    high = render_margin_prompt(FakeTokenizer(), "judge", reasoning_effort="high")
    assert high == none.replace("level: 0<", "level: 0.9<")


def test_render_rejects_a_template_that_opens_the_answer():
    with pytest.raises(ValueError):
        render_margin_prompt(
            FakeTokenizer("<|message_model|>"), "judge", reasoning_effort="none"
        )


def test_opening_prompt_stops_at_the_model_marker():
    text = opening_prompt(FakeTokenizer(), "judge", reasoning_effort="none")
    assert text.endswith("<|end_message|><|message_model|>")


def test_literal_markers_are_detected():
    assert contains_special_marker("before <|message_model|> after")
    assert not contains_special_marker("a <|user|> marker from another family")


def test_configs_keep_the_qwen_contract():
    frozen = json.loads(
        (HERE.parent / "qwen27b_generations_ood/qwen38_27b.json").read_text()
    )
    for arm in ARMS:
        config = json.loads((HERE / f"{arm}.json").read_text())
        assert (config["model"]["id"], config["model"]["revision"]) == (
            MODEL_ID,
            REVISION,
        )
        for key in ("scope", "scoring"):
            assert config[key] == frozen[key]
        frozen_template = frozen["prompt"]["template_sha256"]
        assert config["prompt"]["template_sha256"] == frozen_template
        assert config["prompt"]["assistant_open"] == ASSISTANT_OPEN
        for key in (
            "max_num_batched_tokens",
            "batch_rows",
            "seed",
            "gpu_memory_utilization",
        ):
            assert config["engine"][key] == frozen["engine"][key]
        # Documented departure: Inkling's longest prompt is 37,998 tokens.
        assert config["engine"]["max_model_len"] == 40960
        # Amendment: one sequence at a time (batched scores were not invariant).
        assert config["engine"]["max_num_seqs"] == 1
        assert config["engine"]["enable_prefix_caching"] is False
        assert config["engine"]["speculative_config"] is None
        assert (
            config["engine"]["audited_max_prompt_tokens"]
            < config["engine"]["max_model_len"]
        )


def test_arms_differ_only_in_effort():
    none, high = (json.loads((HERE / f"{arm}.json").read_text()) for arm in ARMS)
    assert (none["prompt"]["reasoning_effort"], high["prompt"]["reasoning_effort"]) == (
        "none",
        "high",
    )
    for key in ("scope", "model", "scoring", "stop_conditions"):
        assert none[key] == high[key]
    audited = {"audited_max_prompt_tokens", "audited_total_prompt_tokens"}
    for key in none["engine"].keys() - audited:
        assert none["engine"][key] == high["engine"][key]
    # "0.9" is two tokens longer than "0" on every row.
    rows = none["scope"]["rows"]
    assert (
        high["engine"]["audited_total_prompt_tokens"]
        == none["engine"]["audited_total_prompt_tokens"] + 2 * rows
    )
