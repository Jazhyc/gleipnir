"""Focused router and experimental training-contract checks."""

from copy import deepcopy

import pytest
import torch

from gleipnir.nvidia_mxfp8_varlen_attention import packed_mxfp8_interface
from gleipnir.packed_training import validate_packed_training_config


def test_router_dispatches_whole_row_and_preserves_boundaries():
    calls = []

    def kernel(q, k, v, cuts, maximum, *, scale):
        calls.append((q, k, v, cuts, maximum, scale))
        return q

    def original(*args, **kwargs):
        raise AssertionError("unexpected unpacked route")

    router = packed_mxfp8_interface(original, kernel)
    q = torch.randn(1, 16, 7, 256)
    k = torch.randn(1, 4, 7, 256)
    cuts = torch.tensor([0, 1, 4, 7], dtype=torch.int32)
    output, auxiliary = router(
        None,
        q,
        k,
        k,
        None,
        cu_seq_lens_q=cuts,
        cu_seq_lens_k=cuts,
        max_length_q=3,
        max_length_k=3,
        scaling=0.0625,
    )
    assert len(calls) == 1
    assert calls[0][0].shape == (7, 16, 256)
    assert calls[0][3] is cuts
    assert calls[0][4:] == (3, 0.0625)
    assert torch.equal(output, q.transpose(1, 2))
    assert auxiliary is None


def test_router_preserves_unpacked_attention():
    sentinel = object()
    router = packed_mxfp8_interface(lambda *a, **k: sentinel)
    assert router(None, None, None, None, None) is sentinel


@pytest.mark.parametrize("change", ["different_cuts", "mask", "dropout", "maximum"])
def test_router_rejects_unsupported_semantics(change):
    cuts = torch.tensor([0, 2, 4], dtype=torch.int32)
    kwargs = dict(
        cu_seq_lens_q=cuts, cu_seq_lens_k=cuts, max_length_q=2, max_length_k=2
    )
    mask = None
    if change == "different_cuts":
        kwargs["cu_seq_lens_k"] = cuts.clone()
    elif change == "mask":
        mask = torch.ones(1)
    elif change == "dropout":
        kwargs["dropout"] = 0.1
    else:
        kwargs["max_length_k"] = 3
    q = torch.empty(1, 16, 4, 256)
    with pytest.raises(ValueError):
        packed_mxfp8_interface(lambda *a, **k: None)(None, q, q, q, mask, **kwargs)


def student():
    from tests.test_packed_training import student_config

    value = deepcopy(student_config())
    value["training"].update(
        packed_attention_backend="nvidia_mxfp8_varlen",
        packed_attention_version="1.31.0",
    )
    return value


def test_varlen_recipe_requires_fresh_startup_validation():
    value = student()
    assert validate_packed_training_config(value)
    value["training"]["startup_validation_reference"] = "old.json"
    with pytest.raises(ValueError, match="fresh"):
        validate_packed_training_config(value)


def test_varlen_timing_authority_stays_bounded():
    value = student()
    value["training"].update(
        packing_timing_authority="Explicit bounded timing run", max_steps=20
    )
    assert validate_packed_training_config(value)
    value["training"]["max_steps"] = 21
    with pytest.raises(ValueError, match="at most 20"):
        validate_packed_training_config(value)


def native_receipt() -> dict:
    """A timing-eligible receipt keeps its failed strict numerical result."""
    return {
        "status": "execution_complete",
        "quantizer_checks": [
            {
                "heads": h,
                "columnwise": col,
                "payload_bitexact": True,
                "canonical_scale_bitexact": True,
                "packed_scale_bitexact": True,
            }
            for h in (16, 4)
            for col in (False, True)
        ],
        "native_dense_comparison": {
            "finite": True,
            "forward_relative_l2": 0,
            "gradient_relative_l2": [0, 0, 0],
        },
        "isolation": {"passed": True},
        "graph_replay": {"passed": True},
        "poisoned_dead_storage": {
            "passed": True,
            "forward_relative_l2": 0,
            "gradient_relative_l2": [0, 0, 0],
        },
        "fp32_comparison": {"strict_passed": False},
    }


def test_native_timing_acceptance_retains_failed_strict_parity():
    from experiments.b200_nvidia_mxfp8_varlen.training_screen import accept_native

    result = accept_native(native_receipt())
    assert result["execution_correct"]
    assert result["strict_parity_passed"] is False


@pytest.mark.parametrize(
    "failure", ["layout", "incomplete", "nonfinite", "isolation", "replay", "poison"]
)
def test_native_execution_failure_blocks_training(failure):
    from experiments.b200_nvidia_mxfp8_varlen.training_screen import accept_native

    receipt = native_receipt()
    if failure == "layout":
        receipt["quantizer_checks"][0]["packed_scale_bitexact"] = False
    elif failure == "incomplete":
        receipt["quantizer_checks"].pop()
    elif failure == "nonfinite":
        receipt["native_dense_comparison"]["finite"] = False
    else:
        key = {
            "isolation": "isolation",
            "replay": "graph_replay",
            "poison": "poisoned_dead_storage",
        }[failure]
        receipt[key]["passed"] = False
    with pytest.raises(ValueError):
        accept_native(receipt)
