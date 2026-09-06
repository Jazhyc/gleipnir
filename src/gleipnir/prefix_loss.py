"""Parent-normalized combination of full and intermediate soft losses."""

import math

import torch


def trajectory_prefix_loss(
    full_losses: torch.Tensor,
    prefix_losses: torch.Tensor,
    prefix_parent_indices: torch.Tensor,
    prefix_weight: float,
) -> torch.Tensor:
    """Average parents equally, regardless of their number of prefix targets.

    Inputs are unreduced per-example losses. Prefix indices address full_losses.
    With sampled prefixes, each parent's sample must be uniform to estimate its
    mean without bias. Parents without prefixes retain their unscaled full loss.
    """
    if not math.isfinite(prefix_weight) or prefix_weight < 0:
        raise ValueError("prefix_weight must be finite and nonnegative")
    if full_losses.ndim != 1 or full_losses.numel() == 0:
        raise ValueError("full losses must be a nonempty vector")
    if prefix_losses.ndim != 1 or prefix_parent_indices.shape != prefix_losses.shape:
        raise ValueError("prefix losses and parent indices must be matching vectors")
    if prefix_parent_indices.dtype != torch.long:
        raise ValueError("parent indices must have dtype int64")
    if not (full_losses.device == prefix_losses.device == prefix_parent_indices.device):
        raise ValueError("all loss inputs must share a device")
    if prefix_parent_indices.numel() and (
        (prefix_parent_indices < 0).any()
        or (prefix_parent_indices >= full_losses.numel()).any()
    ):
        raise ValueError("prefix parent index out of range")
    if prefix_weight == 0 or not prefix_losses.numel():
        return full_losses.mean()
    totals = torch.zeros_like(full_losses).scatter_add(
        0, prefix_parent_indices, prefix_losses.to(full_losses.dtype)
    )
    counts = torch.zeros_like(full_losses).scatter_add(
        0,
        prefix_parent_indices,
        torch.ones_like(prefix_losses, dtype=full_losses.dtype),
    )
    combined = (full_losses + prefix_weight * totals / counts.clamp_min(1)) / (
        1 + prefix_weight
    )
    return torch.where(counts > 0, combined, full_losses).mean()
