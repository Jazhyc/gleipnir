"""CPU checks for the trim variant command and its kernel-equivalence receipt."""

import hashlib
import json

import pytest

from experiments.b200_context_million.test_context_million import (
    PHASES,
    selected_command,
)
from experiments.b200_last_layer_trim.run import FINISHED, LAST, TAIL
from experiments.b200_last_layer_trim.worker import ROOT, validate


def test_trim_command_changes_only_worker_and_bound_sources():
    from experiments.b200_context_million.run import phase_command

    baseline = phase_command(selected_command(), PHASES["yarn"], 32768)
    trimmed = phase_command(selected_command(), PHASES["yarn"], 32768, LAST)
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
    assert LAST.result_group == "b200_last_layer_trim"


def test_tail_variant_swaps_server_and_binds_tail_sources():
    from experiments.b200_context_million.run import phase_command

    command = phase_command(selected_command(), PHASES["native"], 32768, TAIL)
    assert command[1:3] == ["-m", "experiments.b200_last_layer_trim.server"]
    assert command[command.index("--worker-cls") + 1].endswith("TailTrimWorker")
    sources = json.loads(command[command.index("--additional-config") + 1])[
        "gleipnir_frost_fp4"
    ]
    assert "src/gleipnir/serving/last_query_tail.py" in sources
    finished = phase_command(selected_command(), PHASES["native"], 32768, FINISHED)
    assert finished[2] == "experiments.b200_attention_precision.server"


def receipt(**overrides):
    from experiments.b200_last_layer_trim.native import CASES, SUBSETS

    path = "src/gleipnir/serving/last_query_attention.py"
    names = [name for name, *_ in CASES] + [f"{name}_subset" for name in SUBSETS]
    value = {
        "passed": True,
        "bitwise_cases": len(names),
        "mismatched_cases": [],
        "cases": [{"name": name} for name in names],
        "sources": {path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest()},
    }
    return {**value, **overrides}


def test_validation_requires_complete_bitwise_coverage():
    validate(receipt())
    with pytest.raises(ValueError, match="failed"):
        validate(receipt(mismatched_cases=["batch_128"]))
    with pytest.raises(ValueError, match="coverage"):
        validate(receipt(bitwise_cases=7))
    with pytest.raises(ValueError, match="source changed"):
        validate(receipt(sources={"src/gleipnir/serving/last_query_attention.py": "0"}))
