"""Runtime-shaped equivalent of NVIDIA's documented canonical SF repack.

Eleven launches cover the whole packed physical row, independent of example
count. This is explicit conversion work, included in attention/update timing.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _singleton_forward(v, out, cu, HQ: tl.constexpr, HK: tl.constexpr):
    b, head = tl.program_id(0), tl.program_id(1)
    start, end = tl.load(cu + b), tl.load(cu + b + 1)
    if end - start == 1:
        d = tl.arange(0, 256)
        value = tl.load(v + (start * HK + head // (HQ // HK)) * 256 + d)
        tl.store(out + (start * HQ + head) * 256 + d, value)


@triton.jit
def _singleton_backward(grad, dq, dk, dv, cu, HQ: tl.constexpr, HK: tl.constexpr):
    b, head = tl.program_id(0), tl.program_id(1)
    start, end = tl.load(cu + b), tl.load(cu + b + 1)
    if end - start == 1:
        d = tl.arange(0, 256)
        group = tl.arange(0, HQ // HK)
        offsets = (start * HQ + head * (HQ // HK) + group[:, None]) * 256 + d
        value = tl.load(grad + offsets).to(tl.float32)
        tl.store(dq + offsets, tl.full((HQ // HK, 256), 0, tl.float32))
        tl.store(dk + (start * HK + head) * 256 + d, 0)
        tl.store(dv + (start * HK + head) * 256 + d, tl.sum(value, axis=0))


def singleton_forward(v, output, cumulative) -> None:
    """Preserve exact one-token identity without dispatching examples on CPU."""
    _singleton_forward[(cumulative.numel() - 1, output.shape[1])](
        v, output, cumulative, output.shape[1], v.shape[1]
    )


def singleton_backward(grad, dq, dk, dv, cumulative) -> None:
    """Overwrite the exact zero Q/K and summed V gradients for singleton rows."""
    _singleton_backward[(cumulative.numel() - 1, dk.shape[1])](
        grad, dq, dk, dv, cumulative, dq.shape[1], dk.shape[1]
    )


@triton.jit
def _repack(
    src,
    dst,
    rows,
    groups,
    planes,
    size,
    cumulative,
    HEADS: tl.constexpr,
    SFA: tl.constexpr,
    COLUMN: tl.constexpr,
    BLOCK: tl.constexpr,
):
    idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    rm, rk = tl.cdiv(rows, 128), tl.cdiv(groups, 4)
    pm = rm * (2 if SFA else 1)
    k0, m1, m0 = idx % 4, idx // 4 % 4, idx // 16 % 32
    ct, rtp, lp = idx // 512 % rk, idx // (512 * rk) % pm, idx // (512 * rk * pm)
    plane, slot = lp // 2, lp % 2
    if SFA:
        rt = rtp // 2
        odd = rtp % 2
        sm1 = tl.where((slot == 0) & (odd == 1) & (m1 < 2), m1 + 2, m1)
        sm1 = tl.where((slot == 1) & (odd == 1) & (m1 == 1), 3, sm1)
        source = (slot == 0) | (m1 == 1)
    else:
        rt = rtp
        sm1 = tl.where(slot == 1, m1 + 2, m1)
        source = (slot == 0) | (m1 < 2)
    sequence = plane // HEADS
    length = tl.load(cumulative + sequence + 1, idx < size, other=0) - tl.load(
        cumulative + sequence, idx < size, other=0
    )
    live_rows = 256 if COLUMN else length
    live_groups = tl.cdiv(length, 32) if COLUMN else 8
    valid = (rt * 128 + sm1 * 32 + m0 < live_rows) & (ct * 4 + k0 < live_groups)
    if COLUMN:
        atom = (rt * planes + plane) * rk + ct
    else:
        atom = (plane * rm + rt) * rk + ct
    value = tl.load(
        src + atom * 512 + m0 * 16 + sm1 * 4 + k0,
        (idx < size) & source & valid,
        other=127,
    )
    value = tl.where(source, value, 0)
    tl.store(dst + idx, value, idx < size)


def repack(
    source: torch.Tensor,
    maximum: int,
    batch: int,
    heads: int,
    cumulative: torch.Tensor,
    *,
    columnwise: bool,
    sfa: bool,
) -> torch.Tensor:
    """Allocate and convert one whole-row operand to the 2-CTA slot layout."""
    rows, groups = (256, (maximum + 31) // 32) if columnwise else (maximum, 8)
    planes = batch * heads
    size = (
        2
        * planes
        * ((rows + 127) // 128)
        * (2 if sfa else 1)
        * ((groups + 3) // 4)
        * 512
    )
    result = torch.empty(size, device=source.device, dtype=torch.uint8)
    _repack[(triton.cdiv(size, 256),)](
        source,
        result,
        rows,
        groups,
        planes,
        size,
        cumulative,
        heads,
        sfa,
        columnwise,
        256,
    )
    return result


def backward_scales(
    sfq,
    sfqt,
    sfk,
    sfkt,
    sfv,
    sfdo,
    sfdot,
    maximum: int,
    batch: int,
    hq: int,
    hk: int,
    cumulative: torch.Tensor,
) -> tuple:
    """Retain the pinned native backward's eleven operand forms."""
    contract = (
        (sfq, hq, False, True),
        (sfk, hk, False, False),
        (sfkt, hk, True, False),
        (sfv, hk, False, False),
        (sfdo, hq, False, True),
        (sfq, hq, False, False),
        (sfqt, hq, True, False),
        (sfk, hk, False, True),
        (sfv, hk, False, True),
        (sfdo, hq, False, False),
        (sfdot, hq, True, False),
    )
    return tuple(
        repack(src, maximum, batch, heads, cumulative, columnwise=col, sfa=sfa)
        for src, heads, col, sfa in contract
    )
