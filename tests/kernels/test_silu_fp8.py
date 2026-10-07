"""Guard the fused activation contract without requiring CUDA execution."""

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode

from gleipnir.silu_fp8 import silu_fp8, validate_input


@pytest.mark.parametrize("shape", [(4,), (2, 3), (2, 0)])
def test_reject_invalid_shapes(shape):
    with pytest.raises(ValueError, match="gate/up"):
        validate_input(torch.empty(shape, dtype=torch.bfloat16))


def test_reject_wrong_dtype_and_cpu():
    with pytest.raises(ValueError, match="BF16"):
        validate_input(torch.empty((2, 4)))
    with pytest.raises(ValueError, match="CUDA"):
        validate_input(torch.empty((2, 4), dtype=torch.bfloat16))


def test_fake_registration_preserves_shapes_and_scale_dtype():
    with FakeTensorMode():
        x = torch.empty((128, 18432), device="cuda", dtype=torch.bfloat16)
        q, scale = silu_fp8(x)
        assert q.shape == (128, 9216)
        assert q.dtype == torch.float8_e4m3fn
        assert scale.shape == (128, 1)
        assert scale.dtype == torch.float32
