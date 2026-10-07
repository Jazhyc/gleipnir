"""Token-ID ablations must preserve the frozen text and token count contract."""

import pytest

from experiments.b200_inference_benchmark.tokenization import token_rows


def test_token_payload_preserves_identity_and_does_not_mutate_workload():
    original = {
        "id": "a",
        "prompt": "hello",
        "prompt_sha256": "source",
        "prompt_tokens": 2,
    }
    observed = token_rows([original], {"a": [15, 16]})
    assert observed == [{**original, "prompt": [15, 16]}]
    assert original["prompt"] == "hello"


@pytest.mark.parametrize("tokens", [[15], [True, 16], [-1, 16], [15.0, 16]])
def test_token_payload_rejects_count_and_integer_drift(tokens):
    with pytest.raises(ValueError, match="token payload drift"):
        token_rows([{"id": "a", "prompt_tokens": 2}], {"a": tokens})


def test_missing_prompt_ids_cannot_fall_back_to_text():
    with pytest.raises(KeyError):
        token_rows([{"id": "a", "prompt_tokens": 2}], {})
