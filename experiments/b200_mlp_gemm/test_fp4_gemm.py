"""Independent block-scale layout and shape rejection checks."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from gleipnir.cudnn_fp4_epilogue import Nvfp4ScaledGemm
from gleipnir.cudnn_fp4_gemm import (
    PackedNvfp4,
    decode_operand,
    pack_operand,
    validate_weight_shape,
)
from gleipnir.cudnn_fp4_mlp import _plan_key
from gleipnir.nvfp4_reference import decode_nvfp4


@pytest.mark.parametrize("shape", [(0, 64), (17, 64), (16, 48), (16,), (16, 64, 1)])
def test_incomplete_weight_tiles_rejected(shape):
    with pytest.raises(ValueError):
        validate_weight_shape(shape)


def test_actual_geometry_accepts_transposed_weights():
    for shape in [(18432, 2560), (2560, 18432), (2560, 9216), (9216, 2560)]:
        validate_weight_shape(shape)


def test_cpu_packing_rejected_before_cuda_launch():
    with pytest.raises(ValueError, match="CUDA BF16"):
        pack_operand(torch.zeros(16, 64, dtype=torch.bfloat16), weight=True)


@pytest.mark.parametrize("rows,k", [(17, 64), (193, 256)])
@pytest.mark.parametrize("row_scales", [False, True])
def test_swizzled_decoder_against_independent_numpy(rows, k, row_scales):
    rng = np.random.default_rng(17)
    codes = rng.integers(0, 256, (rows, k // 2), dtype=np.uint8)
    scales = (rng.integers(1, 16, (rows, k // 16)) / 16).astype(np.float32)
    row_pad = (rows + 127) // 128 * 128
    logical = np.zeros((row_pad, k // 16), dtype=np.float32)
    logical[:rows] = scales
    # Physical atom grid [row_tiles, k_tiles, row_mod32, row_div32, k_mod4].
    swizzled = logical.reshape(row_pad // 128, 4, 32, k // 64, 4)
    swizzled = swizzled.transpose(0, 3, 2, 1, 4).copy().reshape(-1)
    inverse = (
        np.exp2((np.arange(rows) % 9 - 4).astype(np.float32))
        if row_scales
        else np.array([3.25], dtype=np.float32)
    )
    packed = PackedNvfp4(
        torch.from_numpy(codes), torch.from_numpy(swizzled), torch.from_numpy(inverse)
    )
    expected = np.concatenate(
        [
            decode_nvfp4(
                codes[r : r + 1],
                scales[r : r + 1],
                1 / float(inverse[r] if row_scales else inverse[0]),
            )
            for r in range(rows)
        ]
    )
    np.testing.assert_allclose(
        decode_operand(packed).numpy(), expected, rtol=1e-7, atol=0
    )


def test_training_input_cannot_silently_lose_autograd():
    x = torch.ones(16, 64, dtype=torch.bfloat16, requires_grad=True)
    with pytest.raises(ValueError, match="explicit autograd integration"):
        pack_operand(x)


def runtime_operands(rows, k=64, n=256, *, inverse_rows=None):
    def operand(m, width, inverses):
        return PackedNvfp4(
            torch.empty(m, width // 2, dtype=torch.uint8),
            torch.empty(0),
            torch.ones(inverses),
        )

    return operand(rows, k, rows if inverse_rows is None else inverse_rows), operand(
        n, k, 1
    )


@pytest.mark.parametrize("rows", [1, 127, 128, 129, 5047, 16322, 29337])
def test_runtime_row_plan_accepts_tail_and_long_shapes_without_replanning(rows):
    plan = Nvfp4ScaledGemm.__new__(Nvfp4ScaledGemm)
    plan.runtime_m = True
    plan.plan = SimpleNamespace(m=16322, k=64, n=256)
    assert plan._operand_rows(*runtime_operands(rows)) == rows


@pytest.mark.parametrize("change", ["empty", "k", "n", "inverse_rows", "fixed_m"])
def test_runtime_row_plan_keeps_width_and_scale_guards(change):
    plan = Nvfp4ScaledGemm.__new__(Nvfp4ScaledGemm)
    plan.runtime_m = change != "fixed_m"
    plan.plan = SimpleNamespace(m=16322, k=64, n=256)
    kwargs = {
        "empty": {"rows": 0},
        "k": {"k": 128},
        "n": {"n": 512},
        "inverse_rows": {"inverse_rows": 1},
        "fixed_m": {},
    }[change]
    with pytest.raises(ValueError):
        plan._operand_rows(*runtime_operands(**({"rows": 129} | kwargs)))


def test_runtime_plan_identity_shares_rows_but_separates_device_and_geometry():
    geometries = [(2560, 18432), (9216, 2560), (18432, 2560), (2560, 9216)]
    keys = {
        _plan_key(device, m, k, n, fused_descale=True, runtime_m=True)
        for device in (0, 1)
        for m in (129, 5047, 16322, 29337)
        for k, n in geometries
    }
    assert len(keys) == 8
    assert _plan_key(0, 129, 2560, 9216, fused_descale=True) not in keys
    assert _plan_key(0, 129, 2560, 9216) not in keys
    with pytest.raises(ValueError):
        _plan_key(0, 129, 2560, 9216, runtime_m=True)
