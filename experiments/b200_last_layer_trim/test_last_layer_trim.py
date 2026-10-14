"""CPU checks for the trim variant command and its kernel-equivalence receipt."""

import hashlib
import json

import pytest

from experiments.b200_context_million.test_context_million import (
    PHASES,
    selected_command,
)
from experiments.b200_last_layer_trim.run import VARIANT
from experiments.b200_last_layer_trim.worker import ROOT, validate


def test_trim_command_changes_only_worker_and_bound_sources():
    from experiments.b200_context_million.run import phase_command

    baseline = phase_command(selected_command(), PHASES["yarn"], 32768)
    trimmed = phase_command(selected_command(), PHASES["yarn"], 32768, VARIANT)
    changed = {
        baseline[i - 1]
        for i, (a, b) in enumerate(zip(baseline, trimmed, strict=True))
        if a != b
    }
    assert changed == {"--worker-cls", "--additional-config"}
    sources = json.loads(trimmed[trimmed.index("--additional-config") + 1])[
        "gleipnir_frost_fp4"
    ]
    assert "src/gleipnir/serving/last_query_attention.py" in sources
    assert VARIANT.result_group == "b200_last_layer_trim"


def receipt(**overrides):
    path = "src/gleipnir/serving/last_query_attention.py"
    value = {
        "passed": True,
        "bitwise_cases": 6,
        "mismatched_cases": [],
        "sources": {path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest()},
    }
    return {**value, **overrides}


def test_validation_requires_complete_bitwise_coverage():
    validate(receipt())
    with pytest.raises(ValueError, match="failed"):
        validate(receipt(mismatched_cases=["batch_128"]))
    with pytest.raises(ValueError, match="coverage"):
        validate(receipt(bitwise_cases=5))
    with pytest.raises(ValueError, match="source changed"):
        validate(receipt(sources={"src/gleipnir/serving/last_query_attention.py": "0"}))
