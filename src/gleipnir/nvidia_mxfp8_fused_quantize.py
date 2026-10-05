"""Single-pass packed MXFP8 producer with native consumer scale layouts.

Dual scaling preserves the existing row/column arithmetic. Square scaling uses
one 32x32 block maximum and shares a transpose-invariant payload, following the
Meta low-precision FA4 report. All tile origins are sequence-local.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _boundaries(CU, VALID, total, maximum, batch):
    i = tl.arange(0, 32)
    a = tl.load(CU + i, i < batch, other=0)
    b = tl.load(CU + i + 1, i < batch, other=0)
    each = (b > a) & (b - a <= maximum)
    valid = tl.min(tl.where(i < batch, each, True).to(tl.int32), 0)
    valid &= (tl.load(CU) == 0) & (tl.load(CU + batch) == total)
    tl.store(VALID, valid)


def check_boundaries(cumulative: torch.Tensor, total: int, maximum: int) -> None:
    """Keep device checks while replacing many eager pointwise launches."""
    valid = torch.empty((), dtype=torch.bool, device=cumulative.device)
    _boundaries[(1,)](
        cumulative, valid, total, maximum, cumulative.numel() - 1, num_warps=1
    )
    torch._assert_async(valid, "invalid packed MXFP8 device boundaries")


@triton.jit
def _scale(amax):
    # Match NVIDIA's pinned producer: rounded-up power of two for amax / 448.
    ratio = amax * (1.0 / 448.0)
    packed = tl.inline_asm_elementwise(
        "{ .reg .b16 sf; cvt.rp.satfinite.ue8m0x2.f32 sf, $1, $1; "
        "cvt.u32.u16 $0, sf; }",
        constraints="=r,f",
        args=[ratio],
        dtype=tl.uint32,
        is_pure=True,
        pack=1,
    )
    byte = (packed & 255).to(tl.int32)
    inverse = ((254 - byte) << 23).to(tl.float32, bitcast=True)
    return byte.to(tl.uint8), inverse


@triton.jit
def _prepare(
    X,
    ROW,
    COL,
    SFA,
    SFB,
    SFC,
    PACKED_ROW,
    PACKED_COL,
    CU,
    MAXIMUM,
    CAPACITY,
    HEADS: tl.constexpr,
    BATCH,
    SQUARE: tl.constexpr,
):
    tile, head, batch = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    start, end = tl.load(CU + batch), tl.load(CU + batch + 1)
    length = end - start
    rows = tile * 128 + tl.arange(0, 128)
    cols = tl.arange(0, 256)
    value = tl.load(
        X + ((start + rows[:, None]) * HEADS + head) * 256 + cols[None, :],
        rows[:, None] < length,
        other=0,
    ).to(tl.float32)
    if SQUARE:
        block = tl.reshape(value, (4, 32, 8, 32))
        maximum = tl.max(tl.max(tl.abs(block), 3), 1)
        sf, inverse = _scale(maximum)
        scaled = block * inverse[:, None, :, None]
        row_value = tl.reshape(scaled, (128, 256))
        col_value = row_value
        row_sf = tl.reshape(tl.broadcast_to(sf[:, None, :], (4, 32, 8)), (128, 8))
        col_sf = tl.reshape(
            tl.broadcast_to(tl.trans(sf)[:, None, :], (8, 32, 4)), (256, 4)
        )
    else:
        row_block = tl.reshape(value, (128, 8, 32))
        row_sf, row_inverse = _scale(tl.max(tl.abs(row_block), 2))
        row_value = tl.reshape(row_block * row_inverse[:, :, None], (128, 256))
        col_block = tl.reshape(value, (4, 32, 256))
        col_sf_t, col_inverse = _scale(tl.max(tl.abs(col_block), 1))
        col_value = tl.reshape(col_block * col_inverse[:, None, :], (128, 256))
        col_sf = tl.trans(col_sf_t)
    tl.store(
        ROW + ((start + rows[:, None]) * HEADS + head) * 256 + cols[None, :],
        row_value.to(tl.float8e4nv),
        rows[:, None] < length,
    )
    if not SQUARE:
        tl.store(
            COL + ((start + rows[:, None]) * HEADS + head) * 256 + cols[None, :],
            col_value.to(tl.float8e4nv),
            rows[:, None] < length,
        )

    # Compact forward SF atoms; each sequence receives independent 128 padding.
    prefix = tl.full((), 0, tl.int32)
    for prior in range(batch):
        n = tl.load(CU + prior + 1) - tl.load(CU + prior)
        prefix += tl.cdiv(n, 128)
    i = tl.arange(0, 1024)
    r = (i // 4 % 4) * 32 + i // 16 % 32
    c = i // 512 * 4 + i % 4
    packed_row = tl.gather(tl.reshape(row_sf, (1024,)), r * 8 + c, 0)
    d = i // 512 * 128 + (i // 4 % 4) * 32 + i // 16 % 32
    g = i % 4
    packed_col = tl.gather(tl.reshape(col_sf, (1024,)), d * 4 + g, 0)
    packed = (head * CAPACITY + prefix + tile) * 1024 + i
    tl.store(PACKED_ROW + packed, packed_row, tile * 128 < length)
    tl.store(PACKED_COL + packed, packed_col, tile * 128 < length)

    # Native row SFA: two slots, two Q-half positions and two D scale atoms.
    plane = batch * HEADS + head
    rm = tl.cdiv(MAXIMUM, 128)
    j = tl.arange(0, 4096)
    k0, m1, m0 = j % 4, j // 4 % 4, j // 16 % 32
    ct, odd, slot = j // 512 % 2, j // 1024 % 2, j // 2048
    sm1 = tl.where((slot == 0) & (odd == 1) & (m1 < 2), m1 + 2, m1)
    sm1 = tl.where((slot == 1) & (odd == 1) & (m1 == 1), 3, sm1)
    source = (slot == 0) | (m1 == 1)
    value_sf = tl.gather(
        tl.reshape(row_sf, (1024,)), (sm1 * 32 + m0) * 8 + ct * 4 + k0, 0
    )
    value_sf = tl.where(tile * 128 + sm1 * 32 + m0 < length, value_sf, 127)
    value_sf = tl.where(source, value_sf, 0)
    address = (
        (((plane * 2 + slot) * (rm * 2) + tile * 2 + odd) * 2 + ct) * 512
        + m0 * 16
        + m1 * 4
        + k0
    )
    tl.store(SFA + address, value_sf)

    # Native row SFB: same canonical atoms with the second slot shifted.
    j = tl.arange(0, 2048)
    k0, m1, m0 = j % 4, j // 4 % 4, j // 16 % 32
    ct, slot = j // 512 % 2, j // 1024
    sm1 = tl.where(slot == 1, m1 + 2, m1)
    source = (slot == 0) | (m1 < 2)
    safe_row = tl.minimum(sm1, 3) * 32 + m0
    value_sf = tl.gather(tl.reshape(row_sf, (1024,)), safe_row * 8 + ct * 4 + k0, 0)
    value_sf = tl.where(tile * 128 + safe_row < length, value_sf, 127)
    value_sf = tl.where(source, value_sf, 0)
    address = (((plane * 2 + slot) * rm + tile) * 2 + ct) * 512 + m0 * 16 + m1 * 4 + k0
    tl.store(SFB + address, value_sf)

    # Native column SFB: each sequence tile owns one token-scale atom.
    rk = tl.cdiv(tl.cdiv(MAXIMUM, 32), 4)
    j = tl.arange(0, 2048)
    k0, m1, m0 = j % 4, j // 4 % 4, j // 16 % 32
    rt, slot = j // 512 % 2, j // 1024
    sm1 = tl.where(slot == 1, m1 + 2, m1)
    source = (slot == 0) | (m1 < 2)
    dim = rt * 128 + tl.minimum(sm1, 3) * 32 + m0
    value_sf = tl.gather(tl.reshape(col_sf, (1024,)), dim * 4 + k0, 0)
    valid = (rt * 128 + sm1 * 32 + m0 < 256) & (tile * 4 + k0 < tl.cdiv(length, 32))
    value_sf = tl.where(valid, value_sf, 127)
    value_sf = tl.where(source, value_sf, 0)
    address = (((plane * 2 + slot) * 2 + rt) * rk + tile) * 512 + m0 * 16 + m1 * 4 + k0
    tl.store(SFC + address, value_sf)


def prepare(
    source: torch.Tensor,
    cumulative: torch.Tensor,
    maximum: int,
    *,
    square: bool = False,
) -> tuple[torch.Tensor, ...]:
    """Return shared/dual payloads and initialized native scale layouts."""
    source = source.contiguous()
    total, heads, dim = source.shape
    if dim != 256 or source.dtype != torch.bfloat16:
        raise ValueError("fused MXFP8 preparation requires BF16 D256 inputs")
    batch = cumulative.numel() - 1
    tiles = (maximum + 127) // 128
    capacity = (total + 127) // 128 + batch
    row = torch.empty_like(source, dtype=torch.float8_e4m3fn)
    col = row if square else torch.empty_like(row)
    sizes = (
        batch * heads * tiles * 4096,
        batch * heads * tiles * 2048,
        batch * heads * tiles * 2048,
        heads * capacity * 1024,
        heads * capacity * 1024,
    )
    scales = tuple(
        torch.empty(n, device=source.device, dtype=torch.uint8) for n in sizes
    )
    _prepare[(tiles, heads, batch)](
        source,
        row,
        col,
        *scales,
        cumulative,
        maximum,
        capacity,
        heads,
        batch,
        square,
        num_warps=8,
        enable_fp_fusion=False,
    )
    return row, col, *scales
