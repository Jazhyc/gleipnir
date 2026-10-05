"""Independent eager Qwen BF16 head norm and partial rotary producer oracle."""

import torch


def norm_rope(
    source: torch.Tensor,
    weight: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Preserve each BF16 rounding boundary of Transformers' Qwen producer."""
    x = source.float()
    norm = (
        x
        * torch.rsqrt(x.square().mean(-1, keepdim=True) + eps)
        * (1.0 + weight.float())
    ).to(source.dtype)
    rotary = cos.shape[-1]
    a = norm[..., :rotary]
    first, second = a.chunk(2, dim=-1)
    opposite = torch.cat((-second, first), dim=-1)
    rotated = a * cos[:, None, :] + opposite * sin[:, None, :]
    return torch.cat((rotated, norm[..., rotary:]), dim=-1)
