"""Focused tests for the Flint decision boundary and its frozen config."""

import json
from pathlib import Path

import pytest

from experiments.flint_ood.interface import (
    MODEL_ID,
    REVISION,
    render_margin_prompt,
    template_changed_user_text,
)

HERE = Path(__file__).parent


class FakeTokenizer:
    def __init__(self, suffix: str = "<|assistant|>") -> None:
        self.suffix = suffix

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        assert not tokenize and add_generation_prompt
        text = messages[0]["content"].replace("<|user|>", "&lt;|user|&gt;")
        return f"<|user|>{text}{self.suffix}"


def test_render_opens_assistant_turn_at_decision_prefix():
    rendered = render_margin_prompt(FakeTokenizer(), "judge this")
    assert rendered == "<|user|>judge this<|assistant|>Prediction:"


def test_render_rejects_a_changed_boundary():
    with pytest.raises(ValueError):
        render_margin_prompt(FakeTokenizer("<|assistant|><think>"), "judge this")


def test_sanitizer_rewrite_is_detected():
    tokenizer = FakeTokenizer()
    plain = render_margin_prompt(tokenizer, "no markers")
    marked = render_margin_prompt(tokenizer, "has <|user|> inside")
    assert not template_changed_user_text(plain, "no markers")
    assert template_changed_user_text(marked, "has <|user|> inside")


def test_configs_keep_the_qwen_contract():
    flint = json.loads((HERE / "flint.json").read_text())
    frozen = json.loads(
        (HERE.parent / "qwen27b_generations_ood/qwen38_27b.json").read_text()
    )
    assert (flint["model"]["id"], flint["model"]["revision"]) == (MODEL_ID, REVISION)
    for key in ("scope", "scoring"):
        assert flint[key] == frozen[key]
    assert flint["prompt"]["template_sha256"] == frozen["prompt"]["template_sha256"]
    for key in (
        "max_num_seqs",
        "max_num_batched_tokens",
        "batch_rows",
        "seed",
        "max_model_len",
        "enable_prefix_caching",
        "gpu_memory_utilization",
    ):
        assert flint["engine"][key] == frozen["engine"][key]
    assert (
        flint["engine"]["audited_max_prompt_tokens"] < flint["engine"]["max_model_len"]
    )
