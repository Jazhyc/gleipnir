"""Guard extension, exact input geometry and history-coverage contracts."""

from types import SimpleNamespace

import pytest

from experiments.b200_long_context.envelope import enable, validate
from experiments.b200_long_context.run import extended_command, synthetic


class CharacterTokenizer:
    def encode(self, text, add_special_tokens=False):
        return list(text)

    def decode(self, ids):
        return "".join(ids)


def test_exact_synthetic_lengths_and_identity():
    a = synthetic(CharacterTokenizer(), 8192)
    assert len(a["prompt"]) == a["prompt_tokens"] == 8192
    assert a == synthetic(CharacterTokenizer(), 8192)
    assert "label" not in a


def test_envelope_rejects_already_changed_parent():
    cleared = []
    module = SimpleNamespace(
        MAX_BATCH=128,
        MAX_LENGTH=32768,
        forward_plan=SimpleNamespace(cache_clear=lambda: cleared.append(True)),
    )
    enable(module)
    assert module.MAX_LENGTH == 262144 and cleared == [True]
    with pytest.raises(ValueError, match="original MXFP8"):
        enable(module)


def test_extended_launch_keeps_chunk_budget_and_scheduler():
    command = [
        "python",
        "--max-model-len",
        "32768",
        "--max-num-batched-tokens",
        "32768",
        "--max-num-seqs",
        "128",
        "--runner",
        "pooling",
        "--worker-cls",
        "old",
        "--enable-chunked-prefill",
        "--no-enable-prefix-caching",
    ]
    result = extended_command(command)
    assert result[result.index("--max-model-len") + 1] == "262144"
    assert result[result.index("--max-num-batched-tokens") + 1] == "32768"
    with pytest.raises(ValueError, match="recipe changed"):
        extended_command(command + ["--scheduler-cls", "alternative"])


def test_native_receipt_requires_all_long_histories(tmp_path):
    receipt = {
        "passed": True,
        "context_limit": 262144,
        "short_bitwise_cases": 2,
        "sources": {},
        "checks": [
            {
                "query_tokens": 32768,
                "history_tokens": 65536,
                "finite": True,
                "quantized_relative_l2": 0.001,
            }
        ],
    }
    with pytest.raises(ValueError, match="chunk/history coverage"):
        validate(receipt, tmp_path)
