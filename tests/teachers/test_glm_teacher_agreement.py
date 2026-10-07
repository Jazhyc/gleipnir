import hashlib

import pytest

from experiments.teacher_agreement.glm import validate
from gleipnir.openrouter_cli import parse_args


def test_distillation_filter_default_and_explicit_evaluation_opt_out():
    argv = ["--input", "i", "--output", "o", "--model", "m"]
    assert parse_args(argv).enforce_distillable_text
    assert not parse_args(
        argv + ["--no-enforce-distillable-text"]
    ).enforce_distillable_text


def test_matched_teacher_response_validation():
    prompts = {"a": {"prompt": "example", "metadata": {"label": 0}}}
    row = {
        "id": "a",
        "model": "z-ai/glm-5.3-flash",
        "provider": "Wafer",
        "prompt_sha256": hashlib.sha256(b"example").hexdigest(),
        "metadata": {"label": 0},
        "score": 0.5,
        "top_logprobs": {"0": -1.0, "1": -1.0},
    }
    validate([row], prompts)
    for changed in (
        {"provider": "different"},
        {"score": 0.1},
        {"metadata": {"label": 1}},
    ):
        with pytest.raises(ValueError):
            validate([{**row, **changed}], prompts)
    with pytest.raises(ValueError):
        validate([row, row], prompts)


def test_minimax_requires_zero_reasoning_telemetry():
    prompts = {"a": {"prompt": "example", "metadata": {}}}
    row = {
        "id": "a",
        "model": "minimax/minimax-m3",
        "provider": "CoreWeave",
        "prompt_sha256": hashlib.sha256(b"example").hexdigest(),
        "metadata": {},
        "score": 0.5,
        "top_logprobs": {"0": -1.0, "1": -1.0},
        "request_settings": {"reasoning": {"effort": "none"}},
        "usage": {"completion_tokens_details": {"reasoning_tokens": 0}},
    }
    validate([row], prompts, "CoreWeave", "minimax/minimax-m3", True)
    row["usage"]["completion_tokens_details"]["reasoning_tokens"] = 1
    with pytest.raises(ValueError, match="non-thinking"):
        validate([row], prompts, "CoreWeave", "minimax/minimax-m3", True)
