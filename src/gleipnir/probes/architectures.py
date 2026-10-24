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


class AttentionProbe(HeadProbe):
    """Softmax-weighted average of values over all valid tokens, per head."""

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        logits, values = self.heads(x)
        logits = logits.masked_fill(~mask.unsqueeze(1), float("-inf"))
        pooled = (torch.softmax(logits, -1) * values).sum(-1)
        return pooled.sum(-1) + self.bias


class MultiMaxProbe(HeadProbe):
    """Hard maximum of each head's value over valid tokens (no attention)."""

    def __init__(self, hidden: int, heads: int = HEADS):
        super().__init__(hidden, heads, attention=False)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        _, values = self.heads(x)
        values = values.masked_fill(~mask.unsqueeze(1), float("-inf"))
        return values.amax(-1).sum(-1) + self.bias


def _window_means(
    logits: torch.Tensor, values: torch.Tensor, window: int
) -> torch.Tensor:
    """Attention-weighted mean inside every width-``window`` window, stride 1."""
    weights = torch.softmax(logits.unfold(-1, window, 1), -1)
    return (weights * values.unfold(-1, window, 1)).sum(-1)


class RollingAttentionProbe(HeadProbe):
    """Maximum over sliding windows of the per-head attention-weighted mean.

    Sequences shorter than ``window`` use one window covering the whole row.
    Windows are evaluated in checkpointed chunks to bound activation memory.
    """

    def __init__(self, hidden: int, window: int, heads: int = HEADS):
        super().__init__(hidden, heads)
        self.window = window

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        logits, values = self.heads(x)
        logits = logits.masked_fill(~mask.unsqueeze(1), MASKED_LOGIT)
        values = values.masked_fill(~mask.unsqueeze(1), 0.0)
        tokens = logits.shape[-1]
        if tokens < self.window:
            logits = F.pad(logits, (0, self.window - tokens), value=MASKED_LOGIT)
            values = F.pad(values, (0, self.window - tokens))
            tokens = self.window
        starts = tokens - self.window + 1
        lengths = mask.sum(-1)
        last_start = (lengths - self.window).clamp_min(0)
        best = None
        for begin in range(0, starts, ROLLING_CHUNK):
            stop = min(begin + ROLLING_CHUNK, starts)
            span = slice(begin, stop + self.window - 1)
            if torch.is_grad_enabled():
                means = checkpoint(
                    _window_means,
                    logits[..., span],
                    values[..., span],
                    self.window,
                    use_reentrant=False,
                )
            else:
                means = _window_means(logits[..., span], values[..., span], self.window)
            index = torch.arange(begin, stop, device=x.device)
            valid = index.unsqueeze(0) <= last_start.unsqueeze(1)
            means = means.masked_fill(~valid.unsqueeze(1), float("-inf")).amax(-1)
            best = means if best is None else torch.maximum(best, means)
        return best.sum(-1) + self.bias


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
    if name.startswith("rolling_w"):
        return RollingAttentionProbe(hidden, int(name.removeprefix("rolling_w")))
    raise ValueError(f"unknown probe architecture {name!r}")
