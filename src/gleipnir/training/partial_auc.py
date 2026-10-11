"""Source-macro pairwise surrogate over a fractional high-scoring negative tail."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def negative_tail_weights(scores: torch.Tensor, beta: float) -> torch.Tensor:
    """Give the highest beta fraction unit mass; split boundary ties fairly."""
    if not 0 < beta <= 1 or scores.ndim != 1 or not scores.numel():
        raise ValueError("nonempty scores and beta in (0,1] required")
    if not torch.isfinite(scores).all():
        raise ValueError("nonfinite negative scores")
    mass = beta * scores.numel()
    cutoff = torch.sort(scores.detach(), descending=True).values[math.ceil(mass) - 1]
    above, tied = scores.detach() > cutoff, scores.detach() == cutoff
    fraction = (mass - above.sum()) / tied.sum()
    return above.to(scores.dtype) + tied.to(scores.dtype) * fraction


def partial_auc_loss(
    margins: torch.Tensor,
    labels: torch.Tensor,
    sources: torch.Tensor,
    beta: float = 0.2,
) -> torch.Tensor:
    """Average softplus ranking loss within sources containing both classes."""
    if (
        margins.ndim != 1
        or labels.shape != margins.shape
        or sources.shape != margins.shape
    ):
        raise ValueError("aligned one-dimensional margins, labels and sources required")
    if not torch.isfinite(margins).all() or not ((labels == 0) | (labels == 1)).all():
        raise ValueError("finite margins and binary labels required")
    losses = []
    for source in torch.unique(sources):
        positives = margins[(sources == source) & (labels == 1)]
        negatives = margins[(sources == source) & (labels == 0)]
        if positives.numel() and negatives.numel():
            weights = negative_tail_weights(negatives, beta)
            pair_losses = F.softplus(negatives[None, :] - positives[:, None])
            losses.append(
                (pair_losses * weights[None, :]).sum()
                / (positives.numel() * beta * negatives.numel())
            )
    return torch.stack(losses).mean() if losses else margins.sum() * 0


def replay_row_losses(
    margins: torch.Tensor,
    gradients: torch.Tensor,
    logical_size: int,
    loss_value: torch.Tensor,
) -> torch.Tensor:
    """Row means reproduce the logical loss gradient under size-weighted replay."""
    if margins.shape != gradients.shape or logical_size < margins.numel():
        raise ValueError("gradient replay shape/logical-size drift")
    return loss_value.detach() + logical_size * gradients.detach() * (
        margins - margins.detach()
    )
