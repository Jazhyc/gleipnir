"""Blog-inspired FP16 candidate products and FP32 MSE for NVFP4 activations.

This isolated kernel reuses pinned FourOverSix candidate construction and its
Blackwell packing layout. It is not a claim of TransformerEngine bit parity.
The changed error comparison scales the FP32 target once instead of decoding
both candidates with FP32 multiply/divide chains. Frozen weight packing stays
on the original strict selector.
"""

from __future__ import annotations

from typing import Any

import torch
import triton
import triton.language as tl
from fouroversix.kernels.triton.fp4 import convert_to_e2m1x2_and_quantized_fp16
from fouroversix.kernels.triton.fp8 import convert_e4m3_to_high_precision
from fouroversix.kernels.triton.quantize import (
    nvfp4_fouroversix_quantization_kernel,
    prepare_inputs_for_block_scaling,
)
from fouroversix.quantize.quantized_tensor import QuantizedTensor
from fouroversix.utils import DataType, ScaleRule
from triton.tools.tensor_descriptor import TensorDescriptor

FP16_SELECTOR_TIE_RELATIVE_BAND = 1e-4


@triton.jit
def _candidate(X, AMAX, EXPANSION: tl.constexpr, MAJOR: tl.constexpr):
    scaled, scales, _ = prepare_inputs_for_block_scaling(
        X, AMAX, 16, 64, False, 6, 256, "nv", 16, EXPANSION, MAJOR
    )
    lo, hi = scaled.reshape(16, 32, 2).split()
    packed, quantized = convert_to_e2m1x2_and_quantized_fp16(
        lo, hi, 16, 64, "nearest", 16, -1, MAJOR
    )
    scale16 = convert_e4m3_to_high_precision(scales, tl.float16, MAJOR)
    qlo, qhi = quantized.reshape(16, 32, 2).split()
    qbits = qlo.to(tl.uint16, bitcast=True).to(tl.uint32) | (
        qhi.to(tl.uint16, bitcast=True).to(tl.uint32) << 16
    )
    sf = scale16[:, :, None].broadcast_to(16, 4, 8).reshape(16, 32)
    sf_bits = sf.to(tl.uint16, bitcast=True).to(tl.uint32)
    sf_bits = sf_bits | (sf_bits << 16)
    products = tl.inline_asm_elementwise(
        "mul.rn.f16x2 $0, $1, $2;",
        constraints="=r,r,r",
        args=[qbits, sf_bits],
        dtype=tl.uint32,
        is_pure=True,
        pack=1,
    )
    prod_lo = (products & 65535).to(tl.uint16).to(tl.float16, bitcast=True)
    prod_hi = (products >> 16).to(tl.uint16).to(tl.float16, bitcast=True)
    decoded = tl.join(prod_lo, prod_hi).reshape(16, 4, 16).to(tl.float32)
    return packed, scales, decoded


@triton.jit
def _pack_fast_tiles(
    X_DESC,
    V_DESC,
    SF_DESC,
    AMAX,
    MAJOR: tl.constexpr,
    TIE_BAND: tl.constexpr,
):
    row_block, column_block = tl.program_id(0), tl.program_id(1)
    output_scales = tl.zeros((8, 16, 4), tl.uint8)
    tile_indices = tl.arange(0, 8)[:, None, None]
    for tile in range(8):
        row_start = row_block * 128 + tile * 16
        x = X_DESC.load([row_start, column_block * 64]).to(tl.float32)
        p6, s6, d6 = _candidate(x, AMAX, None, MAJOR)
        p4, s4, d4 = _candidate(x, AMAX, 1.5, MAJOR)
        target = x.reshape(16, 4, 16) * tl.div_rn(1536.0, tl.load(AMAX))
        diff6, diff4 = d6 - target, d4 - target
        err6 = tl.sum(diff6 * diff6, axis=2)
        err4 = tl.sum(diff4 * diff4, axis=2)
        pick4 = err4 < err6
        packed = tl.where(
            pick4[:, :, None], p4.reshape(16, 4, 8), p6.reshape(16, 4, 8)
        ).reshape(16, 32)
        block_scales = tl.where(pick4, s4, s6)
        error_sum = err4 + err6
        ambiguous = (error_sum > 0) & (tl.abs(err4 - err6) <= TIE_BAND * error_sum)
        # Exact strict comparisons on ambiguous tiles avoid changing near-tied
        # choices. All-zero groups already choose six on both paths.
        if tl.sum(tl.sum(ambiguous.to(tl.int32), axis=1), axis=0) > 0:
            packed, block_scales = nvfp4_fouroversix_quantization_kernel(
                x,
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


def quantize_normalized_fp16_selector(
    inputs: torch.Tensor, config: Any
) -> QuantizedTensor:
    """Pack already-normalized contiguous BF16 rows with fixed global amax one."""
    if (
        inputs.ndim != 2
        or not inputs.is_cuda
        or inputs.dtype != torch.bfloat16
        or not inputs.is_contiguous()
        or inputs.shape[0] == 0
        or inputs.shape[1] == 0
        or inputs.shape[1] % 8
    ):
        raise ValueError("FP16 selector requires a nonempty aligned CUDA BF16 matrix")
    major, minor = torch.cuda.get_device_capability(inputs.device)
    if (major, minor) not in {(10, 0), (10, 3)}:
        raise ValueError("FP16 selector requires validated Blackwell hardware")
    amax = config.kwargs["x_amax"]
    if (
        amax.shape != (1,)
        or amax.dtype != torch.float32
        or amax.device != inputs.device
        or not amax.is_contiguous()
    ):
        raise ValueError("FP16 selector requires a matching immutable FP32 amax one")
    m, n = inputs.shape
    pm, pn = triton.cdiv(m, 128) * 128, triton.cdiv(n, 64) * 64
    values = torch.empty((pm, pn // 2), device=inputs.device, dtype=torch.uint8)
    scales = torch.empty(pm * pn // 16, device=inputs.device, dtype=torch.uint8)
    _pack_fast_tiles[(pm // 128, pn // 64)](
        TensorDescriptor.from_tensor(inputs, [16, 64]),
        TensorDescriptor.from_tensor(values, [16, 32]),
        TensorDescriptor.from_tensor(scales, [512]),
        amax,
        major,
        FP16_SELECTOR_TIE_RELATIVE_BAND,
        num_warps=4,
        enable_fp_fusion=True,
    )
    return QuantizedTensor(
        values,
        scales.view(torch.float8_e4m3fn),
        amax,
        DataType.nvfp4,
        inputs.shape,
        ScaleRule.mse,
        padded_shape=(pm, pn),
    )
