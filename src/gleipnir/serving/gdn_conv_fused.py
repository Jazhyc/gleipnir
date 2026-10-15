"""Fused GDN causal conv1d + post-conv prep for prefill, matching vLLM bitwise.

vLLM 0.31 runs `causal_conv1d_fn` (writes the conv'd activation, BF16) and then
`fused_post_conv_prep` (reads it back; splits q/k/v, L2-normalizes q/k and
computes the g/beta gates). This kernel computes the width-4 convolution for one
head's channels inside the post-conv program and applies the post-conv math to
that tile, so the activation never round-trips through memory.

Arithmetic follows upstream exactly: an FP32 accumulator starting at the bias
(or zero), BF16 x*w products added oldest tap first, SiLU as acc/(1+exp(-acc)),
rounding to the activation dtype, then post-conv's FP32 L2 norm and gate
formulas on the same [BLOCK_T, 128] tile shape and warp count. Previous tokens
come from the same sequence, else from the conv state when the sequence has an
initial state, else zero. A second small kernel then writes each sequence's
conv state exactly as upstream does (after the main kernel has read it).
"""

from __future__ import annotations

import torch
from vllm.triton_utils import tl, triton

BLOCK_T = 16


@triton.jit
def _conv_tile(
    x_ptr,
    w_ptr,
    bias_ptr,
    state_ptr,
    cache_ptr,
    has_init_ptr,
    seq_ptr,
    pos_ptr,
    offs_t,
    mask_t,
    chans,
    stride_x_tok,
    stride_w_dim,
    stride_w_width,
    stride_s_seq,
    stride_s_dim,
    stride_s_tok,
    HAS_BIAS: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    """Upstream's width-4 conv + SiLU for [BLOCK_T, BLOCK_C] (token, channel)."""
    seq = tl.load(seq_ptr + offs_t, mask=mask_t, other=0)
    pos = tl.load(pos_ptr + offs_t, mask=mask_t, other=0)
    slot = tl.load(cache_ptr + seq, mask=mask_t, other=0).to(tl.int64)
    has_init = tl.load(has_init_ptr + seq, mask=mask_t, other=0).to(tl.int1)
    w0 = tl.load(w_ptr + chans * stride_w_dim + 0 * stride_w_width)
    w1 = tl.load(w_ptr + chans * stride_w_dim + 1 * stride_w_width)
    w2 = tl.load(w_ptr + chans * stride_w_dim + 2 * stride_w_width)
    w3 = tl.load(w_ptr + chans * stride_w_dim + 3 * stride_w_width)
    if HAS_BIAS:
        acc = tl.load(bias_ptr + chans).to(tl.float32)[None, :] + tl.zeros(
            (offs_t.shape[0], BLOCK_C), dtype=tl.float32
        )
    else:
        acc = tl.zeros((offs_t.shape[0], BLOCK_C), dtype=tl.float32)
    tok = offs_t.to(tl.int64)
    x_cols = chans[None, :]
    state_base = state_ptr + slot[:, None] * stride_s_seq + x_cols * stride_s_dim
    # Taps j = 0..3 read x[t-3+j]; upstream accumulates them in this order.
    for j in tl.static_range(4):
        back = 3 - j
        in_seq = pos >= back
        from_state = (pos < back) & has_init
        x_val = tl.load(
            x_ptr + (tok - back)[:, None] * stride_x_tok + x_cols,
            mask=(mask_t & in_seq)[:, None],
            other=0.0,
        )
        # Upstream's state column for "back" tokens before the start.
        s_col = 3 - (back - pos)
        s_val = tl.load(
            state_base + s_col[:, None] * stride_s_tok,
            mask=(mask_t & from_state)[:, None],
            other=0.0,
        )
        x_val = tl.where(in_seq[:, None], x_val, s_val)
        if j == 0:
            w = w0
        elif j == 1:
            w = w1
        elif j == 2:
            w = w2
        else:
            w = w3
        acc += x_val * w[None, :]
    acc = acc / (1 + tl.exp(-acc))
    return acc.to(x_ptr.dtype.element_ty)


@triton.jit
def _fused_conv_post_kernel(
    x_ptr,
    w_ptr,
    bias_ptr,
    state_ptr,
    cache_ptr,
    has_init_ptr,
    seq_ptr,
    pos_ptr,
    a_ptr,
    b_ptr,
    A_log_ptr,
    dt_bias_ptr,
    q_ptr,
    k_ptr,
    v_ptr,
    g_ptr,
    beta_ptr,
    stride_x_tok,
    stride_w_dim,
    stride_w_width,
    stride_s_seq,
    stride_s_dim,
    stride_s_tok,
    stride_a_tok,
    stride_b_tok,
    stride_q_tok,
    stride_k_tok,
    stride_v_tok,
    L,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    L2NORM_EPS: tl.constexpr,
    SOFTPLUS_THRESHOLD: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
):
    i_tb = tl.program_id(0)
    i_head = tl.program_id(1)
    HK: tl.constexpr = H * K
    offs_t = i_tb * BLOCK_T + tl.arange(0, BLOCK_T)
    mask_t = offs_t < L
    if i_head < H:
        i_h = i_head
        offs_k = tl.arange(0, BK)
        mask_2d = mask_t[:, None] & (offs_k < K)[None, :]
        q_conv = _conv_tile(
            x_ptr,
            w_ptr,
            bias_ptr,
            state_ptr,
            cache_ptr,
            has_init_ptr,
            seq_ptr,
            pos_ptr,
            offs_t,
            mask_t,
            i_h * K + offs_k,
            stride_x_tok,
            stride_w_dim,
            stride_w_width,
            stride_s_seq,
            stride_s_dim,
            stride_s_tok,
            HAS_BIAS,
            BK,
        )
        k_conv = _conv_tile(
            x_ptr,
            w_ptr,
            bias_ptr,
            state_ptr,
            cache_ptr,
            has_init_ptr,
            seq_ptr,
            pos_ptr,
            offs_t,
            mask_t,
            HK + i_h * K + offs_k,
            stride_x_tok,
            stride_w_dim,
            stride_w_width,
            stride_s_seq,
            stride_s_dim,
            stride_s_tok,
            HAS_BIAS,
            BK,
        )
        q_f32 = tl.where(mask_2d, q_conv, 0).to(tl.float32)
        k_f32 = tl.where(mask_2d, k_conv, 0).to(tl.float32)
        q_sq_sum = tl.sum(q_f32 * q_f32, axis=1)
        q_inv = 1.0 / tl.sqrt(q_sq_sum + L2NORM_EPS)
        q_f32 = q_f32 * q_inv[:, None]
        k_sq_sum = tl.sum(k_f32 * k_f32, axis=1)
        k_inv = 1.0 / tl.sqrt(k_sq_sum + L2NORM_EPS)
        k_f32 = k_f32 * k_inv[:, None]
        q_out = offs_t[:, None] * stride_q_tok + i_h * K + offs_k[None, :]
        tl.store(q_ptr + q_out, q_f32.to(q_ptr.dtype.element_ty), mask=mask_2d)
        k_out = offs_t[:, None] * stride_k_tok + i_h * K + offs_k[None, :]
        tl.store(k_ptr + k_out, k_f32.to(k_ptr.dtype.element_ty), mask=mask_2d)
    else:
        i_hv = i_head - H
        offs_v = tl.arange(0, BV)
        mask_2d = mask_t[:, None] & (offs_v < V)[None, :]
        V_OFFSET: tl.constexpr = 2 * H * K
        v_vals = _conv_tile(
            x_ptr,
            w_ptr,
            bias_ptr,
            state_ptr,
            cache_ptr,
            has_init_ptr,
            seq_ptr,
            pos_ptr,
            offs_t,
            mask_t,
            V_OFFSET + i_hv * V + offs_v,
            stride_x_tok,
            stride_w_dim,
            stride_w_width,
            stride_s_seq,
            stride_s_dim,
            stride_s_tok,
            HAS_BIAS,
            BV,
        )
        v_out = offs_t[:, None] * stride_v_tok + i_hv * V + offs_v[None, :]
        tl.store(v_ptr + v_out, v_vals, mask=mask_2d)
        A_log_val = tl.load(A_log_ptr + i_hv).to(tl.float32)
        dt_bias_val = tl.load(dt_bias_ptr + i_hv).to(tl.float32)
        a_vals = tl.load(a_ptr + offs_t * stride_a_tok + i_hv, mask=mask_t, other=0).to(
            tl.float32
        )
        b_vals = tl.load(b_ptr + offs_t * stride_b_tok + i_hv, mask=mask_t, other=0).to(
            tl.float32
        )
        x = a_vals + dt_bias_val
        sp = tl.where(x > 0, x + tl.log(1.0 + tl.exp(-x)), tl.log(1.0 + tl.exp(x)))
        sp = tl.where(x <= SOFTPLUS_THRESHOLD, sp, x)
        g_vals = -tl.exp(A_log_val) * sp
        beta_vals = tl.sigmoid(b_vals)
        gb = offs_t * HV + i_hv
        tl.store(g_ptr + gb, g_vals, mask=mask_t)
        tl.store(beta_ptr + gb, beta_vals, mask=mask_t)


@triton.jit
def _conv_state_update_kernel(
    x_ptr,
    state_ptr,
    cache_ptr,
    has_init_ptr,
    qsl_ptr,
    stride_x_tok,
    stride_s_seq,
    stride_s_dim,
    stride_s_tok,
    dim,
    BLOCK_N: tl.constexpr,
):
    """Upstream's per-sequence conv-state write (width 4, no prefix caching)."""
    seq = tl.program_id(0)
    feats = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    fmask = feats < dim
    start = tl.load(qsl_ptr + seq).to(tl.int64)
    seqlen = tl.load(qsl_ptr + seq + 1).to(tl.int64) - start
    slot = tl.load(cache_ptr + seq).to(tl.int64)
    # Upstream skips empty (padding) sequences and vLLM's reserved null block 0.
    if (seqlen <= 0) | (slot <= 0):
        return
    has_init = tl.load(has_init_ptr + seq).to(tl.int1)
    cols = tl.arange(0, 4)
    cmask = cols < 3
    base = state_ptr + slot * stride_s_seq + feats[None, :] * stride_s_dim
    target = base + cols[:, None] * stride_s_tok
    smask = cmask[:, None] & fmask[None, :]
    if seqlen >= 3:
        src = seqlen - 3 + cols
        vals = tl.load(
            x_ptr + (start + src)[:, None] * stride_x_tok + feats[None, :],
            mask=smask,
            other=0.0,
        )
        tl.store(target, vals, smask)
    else:
        shift = 3 - seqlen
        from_x = cols >= shift
        x_vals = tl.load(
            x_ptr + (start + cols - shift)[:, None] * stride_x_tok + feats[None, :],
            mask=(from_x & cmask)[:, None] & fmask[None, :],
            other=0.0,
        )
        old = tl.load(
            base + (cols + seqlen)[:, None] * stride_s_tok,
            mask=((cols + seqlen) < 3)[:, None] & fmask[None, :] & has_init,
            other=0.0,
        )
        tl.debug_barrier()
        tl.store(target, tl.where(from_x[:, None], x_vals, old), smask)


def token_map(
    query_start_loc: torch.Tensor, tokens: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-token sequence index and position within the sequence (int32)."""
    lengths = (query_start_loc[1:] - query_start_loc[:-1]).to(torch.int64)
    seqs = torch.arange(lengths.numel(), device=lengths.device)
    seq = torch.repeat_interleave(seqs, lengths, output_size=tokens)
    pos = torch.arange(tokens, device=lengths.device) - query_start_loc[seq].to(
        torch.int64
    )
    return seq.to(torch.int32), pos.to(torch.int32)


def fused_conv_post(
    mixed_qkv: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    conv_state: torch.Tensor,
    cache_indices: torch.Tensor,
    has_initial_state: torch.Tensor,
    query_start_loc: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    num_k_heads: int,
    head_k_dim: int,
    head_v_dim: int,
    seq_pos: tuple[torch.Tensor, torch.Tensor] | None = None,
):
    """Return (q, k, v, g, beta) as upstream conv + post-conv; update conv state."""
    tokens, dim = mixed_qkv.shape
    H, K, V = num_k_heads, head_k_dim, head_v_dim
    HV = A_log.shape[0]
    if (
        dim != 2 * H * K + HV * V
        or weight.shape != (dim, 4)
        or mixed_qkv.stride(-1) != 1
        or conv_state.dtype != mixed_qkv.dtype
        or cache_indices.ndim != 1
    ):
        raise ValueError("fused GDN conv requires width-4, channel-last inputs")
    device, dtype = mixed_qkv.device, mixed_qkv.dtype
    q = torch.empty(tokens, H, K, dtype=dtype, device=device)
    k = torch.empty(tokens, H, K, dtype=dtype, device=device)
    v = torch.empty(tokens, HV, V, dtype=dtype, device=device)
    g = torch.empty(tokens, HV, dtype=torch.float32, device=device)
    beta = torch.empty(tokens, HV, dtype=torch.float32, device=device)
    if tokens == 0:
        return q, k, v, g, beta
    seq, pos = seq_pos if seq_pos is not None else token_map(query_start_loc, tokens)
    # conv_state is [cache, dim, state_len] (channel-major per cache line).
    s_seq, s_dim, s_tok = conv_state.stride()
    _fused_conv_post_kernel[(triton.cdiv(tokens, BLOCK_T), H + HV)](
        mixed_qkv,
        weight,
        bias,
        conv_state,
        cache_indices,
        has_initial_state,
        seq,
        pos,
        a,
        b,
        A_log,
        dt_bias,
        q,
        k,
        v,
        g,
        beta,
        mixed_qkv.stride(0),
        weight.stride(0),
        weight.stride(1),
        s_seq,
        s_dim,
        s_tok,
        a.stride(0),
        b.stride(0),
        q.stride(0),
        k.stride(0),
        v.stride(0),
        tokens,
        H=H,
        HV=HV,
        K=K,
        V=V,
        L2NORM_EPS=1e-6,
        SOFTPLUS_THRESHOLD=20.0,
        HAS_BIAS=bias is not None,
        BLOCK_T=BLOCK_T,
        BK=triton.next_power_of_2(K),
        BV=triton.next_power_of_2(V),
        num_warps=4,
        num_stages=2,
    )
    batch = query_start_loc.numel() - 1
    _conv_state_update_kernel[(batch, triton.cdiv(dim, 256))](
        mixed_qkv,
        conv_state,
        cache_indices,
        has_initial_state,
        query_start_loc,
        mixed_qkv.stride(0),
        s_seq,
        s_dim,
        s_tok,
        dim,
        BLOCK_N=256,
    )
    return q, k, v, g, beta
