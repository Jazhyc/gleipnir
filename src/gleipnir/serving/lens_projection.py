"""Request-local residual clamping and FP32 span accumulation primitives."""

import torch


def apply_projection(
    target: torch.Tensor,
    residual_stream: torch.Tensor,
    direction: torch.Tensor,
    beta: float,
    decision_center: float,
    span_center: float,
    start: int,
    end: int,
    absolute_start: int,
    final_position: int,
) -> None:
    """Edit the hidden delta so the complete fused residual receives the correction."""
    if beta == 0:
        return
    x = residual_stream[start:end].float()
    centers = torch.full(
        (end - start,), span_center, device=x.device, dtype=torch.float32
    )
    final = final_position - absolute_start
    if 0 <= final < end - start:
        centers[final] = decision_center
    correction = beta * ((x @ direction) - centers)[:, None] * direction
    target[start:end] = (target[start:end].float() - correction).to(target.dtype)


def apply_projection_batch(
    target: torch.Tensor,
    residual_stream: torch.Tensor,
    direction: torch.Tensor,
    beta: float,
    decision_center: float,
    span_center: float,
    real_tokens: int,
    final_rows: list[int],
) -> None:
    """One matvec for a homogeneous batch, preserving padded output rows."""
    if beta == 0:
        return
    x = residual_stream[:real_tokens].float()
    centers = torch.full(
        (real_tokens,), span_center, device=x.device, dtype=torch.float32
    )
    if final_rows:
        centers[torch.tensor(final_rows, device=x.device)] = decision_center
    correction = beta * ((x @ direction) - centers)[:, None] * direction
    target[:real_tokens] = (target[:real_tokens].float() - correction).to(target.dtype)
