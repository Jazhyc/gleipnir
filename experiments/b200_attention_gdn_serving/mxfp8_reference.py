"""Independent FP32 arithmetic oracle for MXFP8 Q/K/V and online FP8 P."""

import torch


def rows(x: torch.Tensor) -> torch.Tensor:
    groups = x.float().reshape(*x.shape[:-1], 8, 32)
    amax = groups.abs().amax(-1)
    scales = torch.exp2(torch.ceil(torch.log2(amax.clamp_min(1e-30) / 448)))
    payload = (groups / scales[..., None]).to(torch.float8_e4m3fn).float()
    return (payload * scales[..., None]).reshape(x.shape)


def columns(x: torch.Tensor) -> torch.Tensor:
    n, heads, dim = x.shape
    padded = torch.zeros(((n + 31) // 32 * 32, heads, dim), device=x.device)
    padded[:n].copy_(x.float())
    groups = padded.reshape(-1, 32, heads, dim)
    amax = groups.abs().amax(1)
    scales = torch.exp2(torch.ceil(torch.log2(amax.clamp_min(1e-30) / 448)))
    payload = (groups / scales[:, None]).to(torch.float8_e4m3fn).float()
    return (payload * scales[:, None]).reshape(-1, heads, dim)[:n]


def attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """D256: 128-key tiles, unit P scale and four-log2-unit rescale threshold."""
    query, key, value = rows(q), rows(k), columns(v)
    key = key.repeat_interleave(4, 1)
    value = value.repeat_interleave(4, 1)
    score = torch.einsum("qhd,khd->hqk", query, key) * (1.4426950408889634 / 16)
    qn, kn = q.shape[0], k.shape[0]
    mask = (
        torch.arange(kn, device=q.device)[None, :]
        > torch.arange(qn, device=q.device)[:, None] + kn - qn
    )
    score.masked_fill_(mask[None], float("-inf"))
    m = torch.full((16, qn), float("-inf"), device=q.device)
    denominator = torch.zeros_like(m)
    accumulator = torch.zeros((16, qn, 256), device=q.device)
    for start in range(0, kn, 128):
        tile = score[:, :, start : start + 128]
        current = tile.amax(-1)
        next_m = torch.where(current - m > 4.0, current, m)
        alpha = torch.where(torch.isfinite(next_m), torch.exp2(m - next_m), 1.0)
        p = torch.where(
            torch.isfinite(next_m[..., None]), torch.exp2(tile - next_m[..., None]), 0.0
        )
        fp8_p = p.to(torch.float8_e4m3fn).float()
        accumulator = accumulator * alpha[..., None] + torch.einsum(
            "hqk,khd->hqd", fp8_p, value[start : start + 128]
        )
        denominator = denominator * alpha + p.sum(-1)
        m = next_m
    return (accumulator / denominator[..., None]).permute(1, 0, 2)
