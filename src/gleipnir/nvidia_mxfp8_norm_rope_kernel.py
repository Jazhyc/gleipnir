"""Inverse partial RoPE and head RMSNorm gradient in a single Triton kernel."""

import triton
import triton.language as tl


@triton.jit
def norm_rope_backward(
    X,
    W,
    COS,
    SIN,
    G,
    OUT,
    TOTAL,
    HEADS,
    X_ROW,
    X_HEAD,
    G_ROW,
    G_HEAD,
    EPS,
    ROTARY: tl.constexpr,
):
    rows = tl.program_id(0) * 16 + tl.arange(0, 16)
    head = tl.program_id(1)
    cols = tl.arange(0, 256)
    x = tl.load(
        X + rows[:, None] * X_ROW + head * X_HEAD + cols[None, :],
        rows[:, None] < TOTAL,
        other=0,
    ).to(tl.float32)
    g = tl.load(
        G + rows[:, None] * G_ROW + head * G_HEAD + cols[None, :],
        rows[:, None] < TOTAL,
        other=0,
    ).to(tl.float32)
    partner = tl.where(cols < ROTARY // 2, cols + ROTARY // 2, cols - ROTARY // 2)
    partner = tl.where(cols < ROTARY, partner, cols)
    pg = tl.gather(g, tl.broadcast_to(partner[None, :], (16, 256)), 1)
    c = tl.load(
        COS + rows[:, None] * ROTARY + cols[None, :],
        (rows[:, None] < TOTAL) & (cols[None, :] < ROTARY),
        other=0,
    )
    s = tl.load(
        SIN + rows[:, None] * ROTARY + partner[None, :],
        (rows[:, None] < TOTAL) & (cols[None, :] < ROTARY),
        other=0,
    )
    direct = (g * c.to(tl.float32)).to(tl.bfloat16).to(tl.float32)
    rotated = (pg * s.to(tl.float32)).to(tl.bfloat16).to(tl.float32)
    rotated = tl.where(cols[None, :] < ROTARY // 2, rotated, -rotated)
    grad = (direct + rotated).to(tl.bfloat16).to(tl.float32)
    grad = tl.where(cols[None, :] < ROTARY, grad, g)
    scale = 1.0 + tl.load(W + cols).to(tl.float32)
    rstd = tl.rsqrt(tl.sum(x * x, 1) / 256.0 + EPS)
    norm = x * rstd[:, None]
    weighted = grad * scale[None, :]
    correction = tl.sum(weighted * norm, 1) / 256.0
    dx = rstd[:, None] * (weighted - norm * correction[:, None])
    tl.store(
        OUT + (rows[:, None] * HEADS + head) * 256 + cols[None, :],
        dx.to(tl.bfloat16),
        rows[:, None] < TOTAL,
    )
