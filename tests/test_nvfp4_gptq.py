"""Hessian feedback prerequisites are checked independently of GPU packing."""

import pytest
import torch

from gleipnir.nvfp4_gptq import hessian_factor, quantize_gptq


def test_inverse_hessian_factor_and_dead_channels():
    x = torch.tensor([[1.0, 2.0, 0], [2.0, -1.0, 0], [-1.0, 1.0, 0]])
    factor, dead = hessian_factor(x, 0.01)
    h = x.t() @ x / len(x)
    diagonal = h.diagonal()
    diagonal[dead] = 1
    diagonal.add_(0.01 * diagonal.mean())
    torch.testing.assert_close(factor.t() @ factor, torch.linalg.inv(h))
    assert dead.tolist() == [False, False, True]


def test_hessian_rejects_nonfinite_and_invalid_damping():
    for x, damping in [(torch.tensor([[float("nan")]]), 0.01), (torch.ones(2, 2), 0)]:
        with pytest.raises(ValueError, match="Invalid"):
            hessian_factor(x, damping)


def test_gptq_cannot_silently_fall_back_to_cpu():
    with pytest.raises(ValueError, match="CUDA"):
        quantize_gptq(
            torch.zeros(32, 64, dtype=torch.bfloat16), torch.ones(128, 64), block=64
        )
