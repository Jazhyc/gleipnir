"""Explicit auxiliary decision surfaces from one full-vocabulary readout."""

from typing import Any

import torch


def judge_readout(
    logits: torch.Tensor, hidden: torch.Tensor, embedding: torch.Tensor
) -> dict[str, Any]:
    """Return A/B logits and an independent FP32 selected-row head check."""
    if logits.ndim != 1 or logits.numel() < 34 or hidden.shape[0] != 1:
        raise ValueError("invalid A/B readout geometry")
    ids = [32, 33]
    answer = logits[ids].float()
    reference = torch.nn.functional.linear(hidden.float(), embedding[ids].float())[0]
    if not bool(torch.isfinite(answer).all() & torch.isfinite(reference).all()):
        raise ValueError("nonfinite A/B readout")
    return {
        "token_ids": ids,
        "logits": answer.cpu().tolist(),
        "fp32_head_logits": reference.cpu().tolist(),
        "probability_mass": float((answer.logsumexp(0) - logits.logsumexp(0)).exp()),
        "method": "exact_fused_final_norm_tied_full_embedding",
    }
