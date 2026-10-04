"""Verify sequence isolation, native launch layouts and failed receipt semantics."""

import pytest
import torch
import torch.nn.functional as functional

from experiments.b200_nvidia_mxfp8.kernel_canary import difference
from experiments.b200_nvidia_mxfp8.training_screen import accept_native
from gleipnir.nvidia_mxfp8_attention import (
    _ExactSingletonAttention,
    _execute,
    segmented_mxfp8_interface,
    validate_inputs,
)
from gleipnir.packed_sequences import PackedSequenceLayout


def dense(q, k, v, *, scale=None):
    return functional.scaled_dot_product_attention(
        q.transpose(0, 1),
        k.transpose(0, 1),
        v.transpose(0, 1),
        is_causal=True,
        enable_gqa=True,
        scale=scale,
    ).transpose(0, 1)


def test_segmented_attention_keeps_examples_and_gradients_isolated():
    torch.manual_seed(0)
    q, k, v = [torch.randn(1, h, 8, 8, requires_grad=True) for h in [4, 2, 2]]
    calls = []

    def kernel(q, k, v, **kwargs):
        calls.append(q.shape[0])
        return dense(q, k, v, **kwargs)

    router = segmented_mxfp8_interface(None, kernel)
    kwargs = PackedSequenceLayout((3, 5)).kernel_kwargs()
    actual, _ = router(None, q, k, v, None, **kwargs)
    assert calls == [3, 5]
    actual[:, 3:].square().sum().backward()
    for tensor in (q, k, v):
        assert torch.count_nonzero(tensor.grad[:, :, :3]) == 0
        assert torch.count_nonzero(tensor.grad[:, :, 3:]) > 0
    perturbed = v.detach().clone()
    perturbed[:, :, :3] += 100
    after, _ = router(None, q, k, perturbed, None, **kwargs)
    torch.testing.assert_close(actual[:, 3:], after[:, 3:], atol=0, rtol=0)


@pytest.mark.parametrize("problem", ["mask", "boundaries", "dtype", "dropout"])
def test_segmented_invalid_inputs_fail_before_launch(problem):
    kwargs = PackedSequenceLayout((3, 5)).kernel_kwargs()
    mask = None
    q = torch.randn(1, 4, 8, 256)
    if problem == "mask":
        mask = torch.ones(1, 8)
    elif problem == "boundaries":
        kwargs["cu_seq_lens_k"] = torch.tensor([0, 4, 8], dtype=torch.int32)
    elif problem == "dtype":
        kwargs["cu_seq_lens_q"] = kwargs["cu_seq_lens_q"].long()
    else:
        kwargs["dropout"] = 0.1
    with pytest.raises(ValueError):
        segmented_mxfp8_interface(None, dense)(None, q, q, q, mask, **kwargs)


def test_dense_contract_rejects_wrong_dimensions_and_quantized_boundaries():
    q = torch.empty(3, 16, 256, dtype=torch.bfloat16)
    k = torch.empty(3, 4, 256, dtype=torch.bfloat16)
    validate_inputs(q, k, k)
    for wrong in [k[:, :, :128], k.float(), k[:2], k[:, :3]]:
        with pytest.raises(ValueError):
            validate_inputs(q, wrong, wrong)
    validate_inputs(q[:1], k[:1], k[:1])


def test_singleton_has_exact_values_zero_qk_gradients_and_gqa_v_reduction():
    torch.manual_seed(0)
    q, k, v = [
        torch.randn(1, h, 256, dtype=torch.bfloat16).requires_grad_()
        for h in [16, 4, 4]
    ]
    actual = _ExactSingletonAttention.apply(q, k, v)
    references = [x.detach().float().requires_grad_() for x in (q, k, v)]
    expected = dense(*references)
    gradient = torch.randn_like(actual)
    actual.backward(gradient)
    expected.backward(gradient.float())
    torch.testing.assert_close(actual.float(), expected, rtol=0, atol=0)
    for x, y in zip((q, k, v), references, strict=True):
        torch.testing.assert_close(x.grad, y.grad.to(x.dtype), rtol=0, atol=0)


def test_native_binding_preserves_bshd_strides_and_lse_shape(monkeypatch):
    class Graph:
        def get_workspace_size(self):
            return 0

        def execute(self, bindings, workspace):
            assert bindings["q"].shape == (1, 16, 3, 256)
            assert bindings["q"].stride() == (12288, 256, 4096, 1)
            assert bindings["stats"].shape == (1, 16, 3, 1)
            assert bindings["sf_q"].dtype == torch.uint8
            assert workspace.numel() == 1

    q = torch.empty(3, 16, 256)
    buffers = dict(
        q=q, stats=torch.empty(1, 16, 3, 1), sf_q=torch.empty(1024, dtype=torch.uint8)
    )
    _execute(Graph(), {k: k for k in buffers}, buffers)


def test_zero_reference_derivative_and_nonfinite_results_do_not_pass():
    zeros = torch.zeros(5)
    assert difference(zeros, zeros, 0.05)["passed"]
    assert not difference(torch.ones(5), zeros, 0.05)["passed"]
    assert not difference(torch.full((5,), float("nan")), zeros, 0.05)["passed"]
    assert not difference(torch.full((5,), float("inf")), torch.ones(5), 0.05)["passed"]


def test_learning_acceptance_preserves_strict_failure_and_rejects_bad_values():
    from copy import deepcopy

    config = {
        "native_learning_forward_limit": 0.05,
        "native_learning_gradient_limit": 0.1,
        "learning_acceptance_authority": "explicit user instruction",
    }
    receipt = {
        "passed": False,
        "error": (
            "AssertionError: native MXFP8 numerical gate failed; "
            "no model updates permitted"
        ),
        "cases": [
            {
                "length": length,
                "errors": {
                    "forward": {"finite": True, "relative_l2": 0.04},
                    "dq": {"finite": True, "relative_l2": 0.08},
                    "dk": {"finite": True, "relative_l2": 0.08},
                    "dv": {"finite": True, "relative_l2": 0.04},
                },
            }
            for length in [3, 31, 33, 127, 129, 257]
        ],
        "quantizer_layout": [{"passed": True} for _ in range(24)],
        "causal_quantization": {"across_block_future_perturbation_max_absolute": 0},
    }
    result = accept_native(receipt, config)
    assert result["strict_passed"] is False
    assert result["accepted_for_learning_comparison"] is True
    for problem in [
        "finite",
        "gradient",
        "missing_case",
        "isolation",
        "kernel_error",
        "missing_gradient",
        "nan_metric",
        "missing_layout",
    ]:
        changed = deepcopy(receipt)
        if problem == "finite":
            changed["cases"][0]["errors"]["dq"]["finite"] = False
        elif problem == "gradient":
            changed["cases"][0]["errors"]["dq"]["relative_l2"] = 0.15
        elif problem == "missing_case":
            changed["cases"].pop()
        elif problem == "missing_gradient":
            changed["cases"][0]["errors"].pop("dv")
        elif problem == "nan_metric":
            changed["cases"][0]["errors"]["dq"]["relative_l2"] = float("nan")
        elif problem == "missing_layout":
            changed["quantizer_layout"].pop()
        elif problem == "isolation":
            changed["causal_quantization"][
                "across_block_future_perturbation_max_absolute"
            ] = 1
        else:
            changed["error"] = "CUDA kernel failed"
        with pytest.raises(ValueError):
            accept_native(changed, config)


def test_mxfp8_runtime_is_scoped_and_requires_fresh_validation():
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    from gleipnir.packed_sequences import installed_segmented_sdpa
    from gleipnir.packed_training import validate_packed_training_config
    from tests.test_packed_training import student_config

    cfg = student_config()
    cfg["training"].update(
        packed_attention_backend="nvidia_mxfp8",
        packed_attention_version="1.31.0",
        packing_learning_gradient_tolerance=0.1,
    )
    assert validate_packed_training_config(cfg)
    original = ALL_ATTENTION_FUNCTIONS["sdpa"]
    with installed_segmented_sdpa("nvidia_mxfp8", "1.31.0"):
        assert ALL_ATTENTION_FUNCTIONS["sdpa"] is not original
    assert ALL_ATTENTION_FUNCTIONS["sdpa"] is original
    cfg["training"]["startup_validation_reference"] = "old.json"
    with pytest.raises(ValueError):
        validate_packed_training_config(cfg)
