"""Focused explicit-precision routing and bounded training contracts."""

from copy import deepcopy
from unittest.mock import patch

import pytest
import torch

from experiments.b200_mxfp8_fused.square_reference import square_reference
from experiments.b200_mxfp8_fused.training_screen import accept_native
from gleipnir.nvidia_mxfp8_fused_attention import fused_interface
from gleipnir.packed_training import validate_packed_training_config
from tests.helpers.training import student_config


@pytest.mark.parametrize("square", [False, True])
def test_router_keeps_precision_choice_and_one_packed_call(square):
    q = torch.randn(1, 16, 7, 256)
    k = torch.randn(1, 4, 7, 256)
    cuts = torch.tensor([0, 1, 4, 7], dtype=torch.int32)
    with patch(
        "gleipnir.nvidia_mxfp8_fused_attention.packed_attention",
        side_effect=lambda *a, **kw: a[0],
    ) as function:
        output, auxiliary = fused_interface(lambda *a, **kw: None, square=square)(
            None,
            q,
            k,
            k,
            None,
            cu_seq_lens_q=cuts,
            cu_seq_lens_k=cuts,
            max_length_q=3,
            max_length_k=3,
        )
    assert function.call_count == 1
    assert function.call_args.kwargs["square"] is square
    assert function.call_args.args[3] is cuts
    assert torch.equal(output, q.transpose(1, 2))
    assert auxiliary is None


@pytest.mark.parametrize("backend", ["nvidia_mxfp8_fused", "nvidia_mxfp8_square"])
def test_new_backends_require_fresh_validation_and_bounded_authority(backend):
    student = deepcopy(student_config())
    student["training"].update(
        packed_attention_backend=backend,
        packed_attention_version="1.31.0",
        packing_timing_authority="User timing screen",
        max_steps=20,
    )
    assert validate_packed_training_config(student)
    student["training"]["max_steps"] = 21
    with pytest.raises(ValueError, match="at most 20"):
        validate_packed_training_config(student)
    student["training"].update(max_steps=20, startup_validation_reference="old.json")
    with pytest.raises(ValueError, match="fresh"):
        validate_packed_training_config(student)


def native_receipt(square):
    key = "seven_reference_layouts_bitexact" if square else "seven_layouts_bitexact"
    return {
        "status": "execution_complete",
        "square": square,
        "finite": True,
        "quantizer_checks": [
            {"heads": h, key: [True] * 7, "shared_payload": square} for h in (16, 4)
        ],
        "old_native_comparison": {
            "forward_relative_l2": 0,
            "gradient_relative_l2": [0, 0, 0],
        },
        "isolation": {"passed": True},
        "graph_replay": {"passed": True},
        "poisoned_dead_storage": {
            "forward_relative_l2": 0,
            "gradient_relative_l2": [0, 0, 0],
        },
        "fp32_comparison": {"strict_passed": False},
    }


@pytest.mark.parametrize("square", [False, True])
def test_native_acceptance_keeps_strict_failure(square):
    acceptance = accept_native(native_receipt(square), square=square)
    assert acceptance["execution_correct"] is True
    assert acceptance["strict_parity_passed"] is False


@pytest.mark.parametrize(
    "failure", ["mode", "layout", "nonfinite", "isolation", "replay", "poison"]
)
def test_native_failures_block_training(failure):
    receipt = native_receipt(True)
    if failure == "mode":
        receipt["square"] = False
    elif failure == "layout":
        receipt["quantizer_checks"][0]["seven_reference_layouts_bitexact"][3] = False
    elif failure == "nonfinite":
        receipt["finite"] = False
    elif failure in {"isolation", "replay"}:
        receipt["graph_replay" if failure == "replay" else failure]["passed"] = False
    else:
        receipt["poisoned_dead_storage"]["gradient_relative_l2"][0] = 0.01
    with pytest.raises(ValueError):
        accept_native(receipt, square=True)


def test_square_reference_resets_blocks_at_example_boundaries():
    x = torch.ones(34, 1, 256, dtype=torch.bfloat16)
    original = square_reference(x, (1, 33), 33)
    x[1:].mul_(1024)
    changed = square_reference(x, (1, 33), 33)
    assert torch.equal(
        original[0][0].view(torch.uint8), changed[0][0].view(torch.uint8)
    )
    assert torch.equal(original[3][:, :1], changed[3][:, :1])


def test_square_scale_is_shared_across_both_axes_and_stops_at_32():
    x = torch.ones(33, 1, 256, dtype=torch.bfloat16)
    _, _, _, before, _ = square_reference(x, (33,), 33)
    x[32].mul_(1024)
    _, _, _, after, _ = square_reference(x, (33,), 33)
    # Canonical row-zero, feature-block-zero is the first scale byte.
    assert before[0, 0, 0] == after[0, 0, 0]
    x[31].mul_(2048)
    _, _, _, within, _ = square_reference(x, (33,), 33)
    assert before[0, 0, 0] != within[0, 0, 0]
