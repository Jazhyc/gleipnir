"""Decode a packed operand directly to BF16 without full-size index tensors."""

import torch
import triton
import triton.language as tl


@triton.jit
def _decode(
    Q,
    SF,
    INV,
    OUT,
    ROWS,
    K: tl.constexpr,
    ROW_INVERSE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    row, column = i // K, i % K
    payload = tl.load(Q + row * (K // 2) + column // 2, row < ROWS, other=0)
    code = (payload >> ((column % 2) * 4)) & 15
    low = code & 7
    magnitude = tl.where(
        low < 4,
        low.to(tl.float32) * 0.5,
        tl.where(low == 4, 2.0, tl.where(low == 5, 3.0, tl.where(low == 6, 4.0, 6.0))),
    )
    value = tl.where((code & 8) != 0, -magnitude, magnitude)
    group = column // 16
    offset = (
        ((row // 128 * (K // 64) + group // 4) * 32 + row % 32) * 4 + row % 128 // 32
    ) * 4 + group % 4
    scale = tl.load(SF + offset, row < ROWS, other=0.0).to(tl.float32)
    inverse = tl.load(INV + tl.where(ROW_INVERSE, row, 0), row < ROWS, other=1)
    decoded = tl.div_rn(value * scale, inverse)
    tl.store(OUT + i, decoded, row < ROWS)


def decode_bf16(operand):
    """Keep the independent PyTorch decoder as oracle; avoid its int64 gather."""
    rows, half = operand.codes.shape
    output = torch.empty(
        (rows, half * 2), device=operand.codes.device, dtype=torch.bfloat16
    )
    _decode[(triton.cdiv(output.numel(), 1024),)](
        operand.codes.view(torch.uint8),
        operand.scales,
        operand.inverse,
        output,
        rows,
        half * 2,
        operand.inverse.numel() != 1,
        1024,
        enable_fp_fusion=False,
    )
    return output
