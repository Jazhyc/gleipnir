"""Check the boundary contract and causal isolation, including negative controls."""

import pytest
import torch

from experiments.monitoring_sequence_packing import audit_cpu
from experiments.monitoring_sequence_packing.audit_cpu import run_audit
from gleipnir.packed_sequences import PackedSequenceLayout


@pytest.mark.parametrize("lengths", [(), (0,), (-1, 2), (True,), (1.5,), (2**31,)])
def test_layout_rejects_invalid_lengths(lengths):
    with pytest.raises(ValueError):
        PackedSequenceLayout(lengths)


def test_boundaries_share_one_source_and_preserve_decisions():
    layout = PackedSequenceLayout((3, 1, 2))
    kwargs = layout.kernel_kwargs()
    assert kwargs["position_ids"].tolist() == [[0, 1, 2, 0, 0, 1]]
    assert kwargs["seq_idx"].tolist() == [[0, 0, 0, 1, 2, 2]]
    assert kwargs["seq_idx"].dtype == torch.int32
    assert kwargs["cu_seq_lens_q"].tolist() == [0, 3, 4, 6]
    assert kwargs["cu_seq_lens_k"].dtype == torch.int32
    assert kwargs["max_length_q"] == kwargs["max_length_k"] == 3
    assert kwargs["use_cache"] is False
    assert "attention_mask" not in kwargs
    assert layout.decision_positions().tolist() == [2, 3, 5]


def test_boolean_mask_blocks_other_examples_and_future_tokens():
    mask = PackedSequenceLayout((2, 2)).dense_causal_mask()[0, 0]
    assert mask.tolist() == [
        [True, False, False, False],
        [True, True, False, False],
        [False, False, True, False],
        [False, False, True, True],
    ]


def test_dense_oracle_refuses_long_context_allocation():
    with pytest.raises(ValueError, match="token limit"):
        PackedSequenceLayout((1025, 1024)).dense_causal_mask()


@pytest.mark.parametrize("lengths", [[7, 5], [1, 3], [63, 65]])
def test_real_tiny_qwen_detects_missing_resets_and_mask_misuse(lengths):
    result = run_audit(
        dict(
            seed=0,
            lengths=lengths,
            forward_tolerance=1e-5,
            cross_gradient_tolerance=1e-8,
        )
    )
    assert result["passed"], result["cases"]


def test_audit_rejects_nonfinite_negative_controls(monkeypatch):
    monkeypatch.setattr(
        audit_cpu,
        "measure_case",
        lambda *args, **kwargs: {
            "packed_vs_independent_max": float("nan"),
            "prefix_perturbation_max": 0.0,
            "cross_example_input_gradient_max": 0.0,
            "within_example_input_gradient_max": 1.0,
        },
    )
    result = run_audit(
        dict(
            seed=0,
            lengths=[7, 5],
            forward_tolerance=1e-5,
            cross_gradient_tolerance=1e-8,
        )
    )
    assert not result["passed"]
    assert not any(case["finite"] for case in result["cases"].values())
