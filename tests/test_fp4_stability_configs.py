"""Stability screens change one frozen mechanism without changing score contracts."""

import json
from pathlib import Path

import pytest


@pytest.mark.parametrize(
    "source,target,flag,value",
    [
        (
            "gptq_nvfp4_down_fp8_gate",
            "gptq_down_fp8_triton",
            "GLEIPNIR_NVFP4_PACKER",
            "triton",
        ),
        ("baseline", "bf16_no_mp", "VLLM_ENABLE_V1_MULTIPROCESSING", "0"),
        ("fp8_mlp", "fp8_mlp_no_mp", "VLLM_ENABLE_V1_MULTIPROCESSING", "0"),
        ("baseline", "bf16_batch_invariant", "VLLM_BATCH_INVARIANT", "1"),
    ],
)
def test_frozen_mechanism_preserves_scoring_and_artifact_identity(
    source, target, flag, value
):
    root = Path("experiments/fp4_inference/configs")
    original = json.loads((root / f"{source}.json").read_text())
    candidate = json.loads((root / f"{target}.json").read_text())
    assert candidate.pop("output") != original.pop("output")
    assert candidate.pop("diagnostic_parity_override")
    original.pop("diagnostic_parity_override", None)
    expected_environment = dict(original["environment"], **{flag: value})
    assert candidate["environment"] == expected_environment
    candidate["environment"] = original["environment"]
    assert candidate == original
