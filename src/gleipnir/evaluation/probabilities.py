"""Lightweight score conversion for the frozen one-token decision interface."""

from __future__ import annotations

import math
from typing import Any


def logprob_value(value: Any) -> float:
    """Read numeric, mapping, or vLLM-object logprob representations."""
    if isinstance(value, (int, float)):
        return float(value)
    return float(value.logprob if hasattr(value, "logprob") else value["logprob"])


def normalized_binary_probability(logprob_zero: float, logprob_one: float) -> float:
    """Normalize the two logits with the historical margin clamp of ±80."""
    difference = max(-80.0, min(80.0, logprob_one - logprob_zero))
    return 1.0 / (1.0 + math.exp(-difference))


def score_from_output(output: Any, token_ids: list[int]) -> float:
    """Require both explicit first-token logprobs and return the class-one score."""
    if not output.outputs or not output.outputs[0].logprobs:
        raise RuntimeError("vLLM returned no first-token logprobs")
    values = output.outputs[0].logprobs[0] or {}
    expanded = {int(key): logprob_value(value) for key, value in values.items()}
    missing = [token_id for token_id in token_ids if token_id not in expanded]
    if missing:
        raise RuntimeError(f"vLLM omitted requested token logprobs: {missing}")
    return normalized_binary_probability(expanded[token_ids[0]], expanded[token_ids[1]])
