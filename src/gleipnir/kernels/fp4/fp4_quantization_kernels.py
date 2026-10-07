"""Fuse per-token normalization into the pinned Four Over Six activation packer.

This experimental path retains the proven BF16 normalization boundary and the
upstream MSE selector. It does not implement TransformerEngine's FP16 selector.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from fouroversix.kernels.triton.quantize import nvfp4_fouroversix_quantization_kernel
from fouroversix.quantize.quantized_tensor import QuantizedTensor
from fouroversix.utils import DataType, ScaleRule
from triton.tools.tensor_descriptor import TensorDescriptor


@triton.jit
def _normalize_and_pack_rows(
    X,
    VALUES,
    BLOCK_SCALES,
    ROW_SCALES,
    AMAX,
    M: tl.constexpr,
    N: tl.constexpr,
    PADDED_N: tl.constexpr,
    BLOCK: tl.constexpr,
    MAJOR: tl.constexpr,
):
    row = tl.program_id(0)
    columns = tl.arange(0, BLOCK)
    values = tl.load(X + row * N + columns, (row < M) & (columns < N), other=0)
    values = values.to(tl.float32)
    scale = tl.max(tl.abs(values), axis=0)
    scale = tl.where(scale == 0, 1.0, scale)
    normalized = tl.div_rn(values, scale).to(tl.bfloat16).to(tl.float32)
    packed, block_scales = nvfp4_fouroversix_quantization_kernel(
        normalized[None, :],
        AMAX,
        BLOCK_SIZE_M=1,
        BLOCK_SIZE_N=BLOCK,
        ROUND_STYLE="nearest",
        SCALE_TYPE="nv",
        SCALE_GROUP_SIZE=16,
        SCALE_RULE="mse",
        BLOCK_SCALE_2D=False,
        RBITS=-1,
        MAJOR_COMPUTE_CAPABILITY=MAJOR,
    )
    packed_columns = tl.arange(0, BLOCK // 2)
    tl.store(
        VALUES + row * (PADDED_N // 2) + packed_columns,
        packed.reshape(BLOCK // 2),
        packed_columns < PADDED_N // 2,
    )
    groups = tl.arange(0, BLOCK // 16)
    # CUTLASS's Blackwell layout: each 128-row by 64-column tile occupies
    # 512 scale bytes, arranged as [row % 32, row // 32, column group].
    offsets = (
        ((row // 128) * (PADDED_N // 64) + groups // 4) * 512
        + (row % 32) * 16
        + ((row % 128) // 32) * 4
        + groups % 4
    )
    tl.store(
        BLOCK_SCALES + offsets,
        block_scales.reshape(BLOCK // 16),
        groups < PADDED_N // 16,
    )
    tl.store(ROW_SCALES + row, scale, row < M)


@triton.jit
def _row_maxima(X, S, M: tl.constexpr, N: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    columns = tl.arange(0, BLOCK)
    values = tl.load(X + row * N + columns, (row < M) & (columns < N), other=0)
    scale = tl.max(tl.abs(values.to(tl.float32)), axis=0)
    tl.store(S + row, tl.where(scale == 0, 1.0, scale))


@triton.jit
def _pack_normalized_tiles(X_DESC, V_DESC, SF_DESC, S, AMAX, MAJOR: tl.constexpr):
    row_block = tl.program_id(0)
    column_block = tl.program_id(1)
    output_scales = tl.zeros((8, 16, 4), tl.uint8)
    tile_indices = tl.arange(0, 8)[:, None, None]
    # Retain the pinned packer's 16x64 TMA tile and reduction layout.
    for tile in range(8):
        row_start = row_block * 128 + tile * 16
        values = X_DESC.load([row_start, column_block * 64]).to(tl.float32)
        row_scales = tl.load(S + row_start + tl.arange(0, 16))
        normalized = tl.div_rn(values, row_scales[:, None])
        normalized = normalized.to(tl.bfloat16).to(tl.float32)
        packed, block_scales = nvfp4_fouroversix_quantization_kernel(
            normalized,
            AMAX,
            BLOCK_SIZE_M=16,
            BLOCK_SIZE_N=64,
            ROUND_STYLE="nearest",
            SCALE_TYPE="nv",
            SCALE_GROUP_SIZE=16,
            SCALE_RULE="mse",
            BLOCK_SCALE_2D=False,
            RBITS=-1,
            MAJOR_COMPUTE_CAPABILITY=MAJOR,
        )
        V_DESC.store([row_start, column_block * 32], packed)
        output_scales = tl.where(
            tile_indices == tile, block_scales[None, :, :], output_scales
        )
    swizzled = output_scales.reshape(4, 32, 4).permute(1, 0, 2).ravel()
    SF_DESC.store([(row_block * tl.num_programs(1) + column_block) * 512], swizzled)


def quantize_activation_rows(
    inputs: torch.Tensor, fixed_amax: torch.Tensor, *, implementation: str = "tiled"
) -> tuple[QuantizedTensor, torch.Tensor]:
    """Pack normalized BF16 rows with exact FP32 maxima and fixed amax one.

    The caller supplies the existing immutable one-valued quantizer amax;
    validating its contents on every call would introduce a CUDA host wait.
    This candidate is only for the isolated, pinned SM100/SM103 environment.
    """
    if inputs.ndim != 2 or inputs.dtype != torch.bfloat16 or not inputs.is_cuda:
        raise ValueError("fused activation packing requires a CUDA BF16 matrix")
    if not inputs.is_contiguous() or not 0 < inputs.shape[1] <= 16384:
        raise ValueError("fused activation packing requires contiguous supported rows")
    if (
        fixed_amax.shape != (1,)
        or fixed_amax.dtype != torch.float32
        or fixed_amax.device != inputs.device
        or not fixed_amax.is_contiguous()
    ):
        raise ValueError("fused activation packing requires a matching FP32 scalar")
    major, minor = torch.cuda.get_device_capability(inputs.device)
    if (major, minor) not in {(10, 0), (10, 3)}:
        raise ValueError(
            "fused activation packing requires validated Blackwell hardware"
        )
    rows, columns = inputs.shape
    if rows == 0:
        raise ValueError("fused activation packing requires at least one row")
    if implementation not in {"row", "tiled"}:
        raise ValueError("unknown activation packing implementation")
    if implementation == "tiled" and columns % 8:
        raise ValueError("TMA activation packing requires a 16-byte aligned row stride")
    padded_rows = triton.cdiv(rows, 128) * 128
    padded_columns = triton.cdiv(columns, 64) * 64
    values = torch.empty(
        (padded_rows, padded_columns // 2), device=inputs.device, dtype=torch.uint8
    )
    block_scales = torch.empty(
        padded_rows * padded_columns // 16, device=inputs.device, dtype=torch.uint8
    )
    row_scales = torch.empty((rows, 1), device=inputs.device, dtype=torch.float32)
    if implementation == "row":
        _normalize_and_pack_rows[(padded_rows,)](
            inputs,
            values,
            block_scales,
            row_scales,
            fixed_amax,
            rows,
            columns,
            padded_columns,
            triton.next_power_of_2(padded_columns),
            major,
            num_warps=16 if columns > 4096 else 8,
            enable_fp_fusion=False,
        )
    else:
        padded_scales = torch.empty(
            (padded_rows, 1), device=inputs.device, dtype=torch.float32
        )
        row_scales = padded_scales[:rows]
        _row_maxima[(padded_rows,)](
            inputs,
            padded_scales,
            rows,
            columns,
            triton.next_power_of_2(columns),
            num_warps=8 if columns > 4096 else 4,
            enable_fp_fusion=False,
        )
        _pack_normalized_tiles[(padded_rows // 128, padded_columns // 64)](
            TensorDescriptor.from_tensor(inputs, [16, 64]),
            TensorDescriptor.from_tensor(values, [16, 32]),
            TensorDescriptor.from_tensor(block_scales, [512]),
            padded_scales,
            fixed_amax,
            major,
            num_warps=4,
        )
    return (
        QuantizedTensor(
            values,
            block_scales.view(torch.float8_e4m3fn),
            fixed_amax,
            DataType.nvfp4,
            inputs.shape,
            ScaleRule.mse,
            padded_shape=(padded_rows, padded_columns),
        ),
        row_scales,
    )
