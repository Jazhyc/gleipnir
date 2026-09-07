"""Per-parent MIL eligibility and loss reduction for mixed training corpora."""

from collections.abc import Callable, Mapping
from typing import Any

import torch
import torch.nn.functional as F


def mil_row_enabled(record: Mapping[str, Any]) -> bool:
    """Preserve historical MIL behavior unless a row explicitly opts out."""
    enabled = record.get("mil_enabled", True)
    if not isinstance(enabled, bool):
        raise ValueError("mil_enabled must be a boolean")
    return enabled


def masked_mil_bce(
    margins: torch.Tensor,
    mask: torch.Tensor,
    targets: torch.Tensor,
    *,
    pool: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
) -> torch.Tensor:
    """Average eligible bag losses over *all* parents, with zero for opt-outs.

    Do not normalize by eligible count: that would upweight monitoring examples
    according to the other examples sharing their microbatch.
    """
    if (
        margins.ndim != 2
        or mask.shape != margins.shape
        or targets.shape != margins.shape[:1]
    ):
        raise ValueError("invalid mixed MIL batch shapes")
    eligible = mask.any(dim=1)
    if not bool(eligible.any()):
        return margins.sum() * 0.0
    logits = pool(margins[eligible], mask[eligible])
    return (
        F.binary_cross_entropy_with_logits(
            logits, targets[eligible].float(), reduction="sum"
        )
        / margins.shape[0]
    )
