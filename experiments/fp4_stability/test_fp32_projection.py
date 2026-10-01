"""Verify the projection boundary under outer BF16 autocast."""

from types import SimpleNamespace

import pytest
import torch

from gleipnir.fp32_projection import install_fp32_lm_head


def test_fp32_projection_preserves_values_gradients_weights_under_outer_autocast():
    head = torch.nn.Linear(4, 3).requires_grad_(False)
    model = SimpleNamespace(lm_head=head)
    x = torch.randn(2, 4, requires_grad=True)
    reference = head(x)
    reference.sum().backward()
    gradient = x.grad.clone()
    x.grad = None
    weight = head.weight.clone()
    receipt = install_fp32_lm_head(model)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        actual = head(x)
    actual.sum().backward()
    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, reference)
    torch.testing.assert_close(x.grad, gradient)
    torch.testing.assert_close(head.weight, weight)
    assert receipt["original_output_dtypes"] == ["torch.bfloat16"]
    assert install_fp32_lm_head(model) is receipt


def test_fp32_projection_rejects_trainable_head():
    with pytest.raises(ValueError, match="frozen ordinary"):
        install_fp32_lm_head(SimpleNamespace(lm_head=torch.nn.Linear(4, 3)))
