"""Sequence probes over frozen per-token activations.

The pooled probes follow Kramár et al. 2026 ("Building Production-Ready Probes
For Gemini", arXiv:2601.11516): a per-token ReLU MLP (two layers of width 100)
feeds ``heads`` learned query/value vectors, whose per-head readouts are summed
with a bias. Every forward takes padded ``[batch, tokens, hidden]`` activations
and a boolean ``[batch, tokens]`` validity mask and returns one logit per row.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

WIDTH = 100
HEADS = 10
ROLLING_CHUNK = 2048
# Finite stand-in for -inf inside softmax windows: an all-padding window then
# yields finite (later discarded) values instead of NaN gradients.
MASKED_LOGIT = -1e9


def token_mlp(hidden: int, width: int = WIDTH) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(hidden, width), nn.ReLU(), nn.Linear(width, width), nn.ReLU()
    )


def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weights = mask.to(values.dtype)
    return (values * weights).sum(-1) / weights.sum(-1).clamp_min(1)


class MeanProbe(nn.Module):
    """Mean over tokens of a per-token linear (``mlp=False``) or MLP score."""

    def __init__(self, hidden: int, *, mlp: bool):
        super().__init__()
        self.body = token_mlp(hidden) if mlp else nn.Identity()
        self.score = nn.Linear(WIDTH if mlp else hidden, 1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return masked_mean(self.score(self.body(x)).squeeze(-1), mask)


class HeadProbe(nn.Module):
    """Shared base: per-token MLP, then per-head attention logits and values."""

    def __init__(self, hidden: int, heads: int = HEADS, *, attention: bool = True):
        super().__init__()
        self.body = token_mlp(hidden)
        self.query = nn.Linear(WIDTH, heads, bias=False) if attention else None
        self.value = nn.Linear(WIDTH, heads, bias=False)
        self.bias = nn.Parameter(torch.zeros(()))

    def heads(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return attention logits (``None`` without a query) and values, both
        ``[batch, heads, tokens]``."""
        y = self.body(x)
        values = self.value(y).float().transpose(1, 2)
        if self.query is None:
            return None, values
        return self.query(y).float().transpose(1, 2), values


def attention_pool(
    logits: torch.Tensor, values: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """Softmax-weighted average over valid tokens: ``[B, C, T]`` -> ``[B, C]``."""
    logits = logits.masked_fill(~mask.unsqueeze(1), float("-inf"))
    return (torch.softmax(logits, -1) * values).sum(-1)


def max_pool(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Hard maximum over valid tokens: ``[B, C, T]`` -> ``[B, C]``."""
    return values.masked_fill(~mask.unsqueeze(1), float("-inf")).amax(-1)


def _window_means(
    logits: torch.Tensor, values: torch.Tensor, window: int
) -> torch.Tensor:
    """Attention-weighted mean inside every width-``window`` window, stride 1."""
    weights = torch.softmax(logits.unfold(-1, window, 1), -1)
    return (weights * values.unfold(-1, window, 1)).sum(-1)


def rolling_pool(
    logits: torch.Tensor, values: torch.Tensor, mask: torch.Tensor, window: int
) -> torch.Tensor:
    """Maximum over sliding windows of the attention-weighted mean.

    ``[B, C, T]`` -> ``[B, C]``. Rows shorter than ``window`` use one window
    covering the whole row. Windows are evaluated in checkpointed chunks to
    bound activation memory.
    """
    logits = logits.masked_fill(~mask.unsqueeze(1), MASKED_LOGIT)
    values = values.masked_fill(~mask.unsqueeze(1), 0.0)
    tokens = logits.shape[-1]
    if tokens < window:
        logits = F.pad(logits, (0, window - tokens), value=MASKED_LOGIT)
        values = F.pad(values, (0, window - tokens))
        tokens = window
    starts = tokens - window + 1
    last_start = (mask.sum(-1) - window).clamp_min(0)
    best = None
    for begin in range(0, starts, ROLLING_CHUNK):
        stop = min(begin + ROLLING_CHUNK, starts)
        span = slice(begin, stop + window - 1)
        if torch.is_grad_enabled():
            means = checkpoint(
                _window_means,
                logits[..., span],
                values[..., span],
                window,
                use_reentrant=False,
            )
        else:
            means = _window_means(logits[..., span], values[..., span], window)
        index = torch.arange(begin, stop, device=logits.device)
        valid = index.unsqueeze(0) <= last_start.unsqueeze(1)
        means = means.masked_fill(~valid.unsqueeze(1), float("-inf")).amax(-1)
        best = means if best is None else torch.maximum(best, means)
    return best


class AttentionProbe(HeadProbe):
    """Softmax-weighted average of values over all valid tokens, per head."""

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        logits, values = self.heads(x)
        return attention_pool(logits, values, mask).sum(-1) + self.bias


class MultiMaxProbe(HeadProbe):
    """Hard maximum of each head's value over valid tokens (no attention)."""

    def __init__(self, hidden: int, heads: int = HEADS):
        super().__init__(hidden, heads, attention=False)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        _, values = self.heads(x)
        return max_pool(values, mask).sum(-1) + self.bias


class RollingAttentionProbe(HeadProbe):
    """Maximum over sliding windows of the per-head attention-weighted mean."""

    def __init__(self, hidden: int, window: int, heads: int = HEADS):
        super().__init__(hidden, heads)
        self.window = window

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        logits, values = self.heads(x)
        return rolling_pool(logits, values, mask, self.window).sum(-1) + self.bias


class AlphaEvolveProbe(nn.Module):
    """The best AlphaEvolve-discovered probe (Kramár et al. 2026, Algorithm 1).

    ``H = MLP(LayerNorm(X))``, ``V = (W_proj H) * Softplus(W_gate H)``, bipolar
    pooling ``concat(max_t V, -min_t V)`` and a linear readout. Training adds
    ``l1 * sum |W|`` over all parameters and ``ortho * ||W_proj^T W_proj - I||^2``
    (orthonormal projection columns), see :meth:`regularization`. The paper does
    not state the channel count or the two weights; we use its 10-head default
    and record ``l1``/``ortho`` in the probe name.
    """

    def __init__(self, hidden: int, *, l1: float, ortho: float, channels: int = HEADS):
        super().__init__()
        self.norm = nn.LayerNorm(hidden)
        self.body = token_mlp(hidden)
        self.project = nn.Linear(WIDTH, channels, bias=False)
        self.gate = nn.Linear(WIDTH, channels, bias=False)
        self.out = nn.Linear(2 * channels, 1)
        self.l1, self.ortho = l1, ortho

    def gated(self, h: torch.Tensor) -> torch.Tensor:
        """``[..., WIDTH]`` -> ``[..., channels]`` gated projections."""
        return self.project(h) * F.softplus(self.gate(h))

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        values = self.gated(self.body(self.norm(x))).float().transpose(1, 2)
        return bipolar_readout(values, mask, self.out.weight[0], self.out.bias[0])

    def regularization(self) -> torch.Tensor:
        return alphaevolve_penalty(
            list(self.parameters()), self.project.weight, self.l1, self.ortho
        )


def bipolar_readout(
    values: torch.Tensor, mask: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor
) -> torch.Tensor:
    """``concat(max_t V, -min_t V) @ weight + bias`` for ``V`` in ``[B, C, T]``."""
    pooled = torch.cat([max_pool(values, mask), max_pool(-values, mask)], -1)
    return pooled @ weight + bias


def alphaevolve_penalty(
    parameters: list[torch.Tensor], project: torch.Tensor, l1: float, ortho: float
) -> torch.Tensor:
    """L1 over every parameter plus orthonormality of the projection columns.

    ``project`` is ``[channels, WIDTH]`` (PyTorch layout); the paper's
    ``W_proj^T W_proj`` over output columns is ``project @ project.T`` here.
    """
    l1_term = sum(p.abs().sum() for p in parameters)
    gram = project @ project.T
    eye = torch.eye(gram.shape[0], device=gram.device, dtype=gram.dtype)
    return l1 * l1_term + ortho * (gram - eye).pow(2).sum()


ALPHAEVOLVE_DEFAULTS = {"l1": 1e-6, "ortho": 1e-3}


def parse_alphaevolve(name: str) -> dict[str, float]:
    """``alphaevolve`` or ``alphaevolve_l1e-06_o0.001`` -> penalty weights."""
    if name == "alphaevolve":
        return dict(ALPHAEVOLVE_DEFAULTS)
    _, l1, ortho = name.split("_")
    if not (l1.startswith("l") and ortho.startswith("o")):
        raise ValueError(f"malformed AlphaEvolve probe name {name!r}")
    return {"l1": float(l1[1:]), "ortho": float(ortho[1:])}


def build(name: str, hidden: int) -> nn.Module:
    """Construct a probe from its sweep name, e.g. ``rolling_w64``."""
    if name == "mean_linear":
        return MeanProbe(hidden, mlp=False)
    if name == "mean_mlp":
        return MeanProbe(hidden, mlp=True)
    if name == "attention":
        return AttentionProbe(hidden)
    if name == "multimax":
        return MultiMaxProbe(hidden)
    if name.startswith("alphaevolve"):
        return AlphaEvolveProbe(hidden, **parse_alphaevolve(name))
    if name.startswith("rolling_w"):
        return RollingAttentionProbe(hidden, int(name.removeprefix("rolling_w")))
    raise ValueError(f"unknown probe architecture {name!r}")
