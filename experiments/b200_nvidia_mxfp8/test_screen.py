"""Verify sequence isolation, native launch layouts and failed receipt semantics."""

import pytest
import torch
import torch.nn.functional as functional

from experiments.b200_nvidia_mxfp8.kernel_canary import difference
from gleipnir.nvidia_mxfp8_attention import (
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
