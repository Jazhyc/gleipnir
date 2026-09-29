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


@pytest.mark.parametrize(
    "name,budget,sequences,repeats",
    [
        ("fp8_mlp_b8192_confirm", 8192, 2, 3),
        ("fp8_mlp_b8192_s8", 8192, 8, 1),
        ("fp8_mlp_b16384_s2", 16384, 2, 1),
        ("fp8_mlp_b16384_s8", 16384, 8, 1),
    ],
)
def test_mlp_fp8_scheduler_screen_preserves_precision_and_scoring(
    name, budget, sequences, repeats
):
    root = Path("experiments/fp4_inference/configs")
    original = json.loads((root / "fp8_mlp_b8192_s2.json").read_text())
    candidate = json.loads((root / f"{name}.json").read_text())
    assert candidate.pop("output") != original.pop("output")
    assert candidate["engine"].pop("max_num_batched_tokens") == budget
    assert candidate["engine"].pop("max_num_seqs") == sequences
    original["engine"].pop("max_num_batched_tokens")
    original["engine"].pop("max_num_seqs")
    assert candidate.pop("repeats") == repeats
    original.pop("repeats")
    assert candidate == original


def test_triton_hybrid_confirmation_changes_only_output_and_repeats():
    root = Path("experiments/fp4_inference/configs")
    original = json.loads((root / "gptq_down_fp8_triton.json").read_text())
    candidate = json.loads((root / "gptq_down_fp8_triton_confirm.json").read_text())
    assert candidate.pop("output") != original.pop("output")
    assert candidate.pop("repeats") == 3
    assert original.pop("repeats") == 1
    assert candidate == original
