"""Pass vLLM's BF16 GDN destination through FlashInfer's public output API."""

from collections.abc import Callable
from typing import Any

import torch


def destination_view(
    destination: torch.Tensor, q: torch.Tensor, v: torch.Tensor
) -> torch.Tensor:
    """Return the exact active output view without allocating or copying."""
    if q.ndim != 4 or v.ndim != 4 or q.shape[0] != 1 or v.shape[0] != 1:
        raise ValueError("direct GDN output requires singleton-batch Q/V")
    tokens, heads, width = q.shape[1], max(q.shape[2], v.shape[2]), q.shape[3]
    if v.shape[1] != tokens or v.shape[3] != width:
        raise ValueError("direct GDN output Q/V dimensions disagree")
    if destination.dtype != q.dtype or destination.device != q.device:
        raise ValueError("direct GDN output dtype/device mismatch")
    elements = tokens * heads * width
    if not destination.is_contiguous() or destination.numel() < elements:
        raise ValueError("direct GDN output needs a contiguous sufficient buffer")
    if destination.data_ptr() % 16:
        raise ValueError("direct GDN output requires 16-byte alignment")
    return destination.view(-1)[:elements].view(tokens, heads, width)


def make_forward(
    kernel: Callable[..., Any],
    normalize: Callable[[torch.Tensor], torch.Tensor],
    observed: Callable[[int, bool], None] | None = None,
) -> Callable[..., tuple[torch.Tensor, torch.Tensor | None]]:
    """Preserve vLLM's normalization, BF16 operands, exp gates and FP32 state."""

    def forward(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        initial_state: torch.Tensor,
        output_final_state: bool,
        cu_seqlens: torch.Tensor | None = None,
        chunk_indices: torch.Tensor | None = None,
        chunk_offsets: torch.Tensor | None = None,
        use_qk_l2norm_in_kernel: bool = True,
        core_attn_out: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        output = None
        if core_attn_out is not None:
            output = destination_view(core_attn_out, q, v)
            # A shared storage could overwrite operands, gates or recurrent state.
            for operand in (q, k, v, g, beta, initial_state):
                if torch._C._overlaps(output, operand):
                    raise ValueError("direct GDN destination aliases an input")
        if use_qk_l2norm_in_kernel:
            q, k = normalize(q), normalize(k)
        q, k, v = [x.squeeze(0).contiguous() for x in (q, k, v)]
        g, beta = [x.squeeze(0).contiguous() for x in (g, beta)]
        result = kernel(
            q=q,
            k=k,
            v=v,
            g=torch.exp(g.to(torch.float32)),
            beta=beta.to(torch.float32),
            initial_state=initial_state.to(torch.float32),
            output_final_state=output_final_state,
            cu_seqlens=cu_seqlens,
            output=output,
        )
        value, state = result if output_final_state else (result, None)
        if output is not None and (
            value.data_ptr() != output.data_ptr() or value.shape != output.shape
        ):
            raise ValueError("FlashInfer did not return the supplied GDN destination")
        if observed is not None:
            observed(q.shape[0], output is not None)
        return value.unsqueeze(0), state

    return forward


def validate_native(receipt: dict) -> None:
    """Reject failed or incomplete arithmetic/layout/replay admission."""
    if (
        receipt.get("passed") is not True
        or receipt.get("intervention") != "flashinfer_gdn_direct_output"
        or not receipt.get("checks")
        or not all(check.get("passed") is True for check in receipt["checks"])
        or not {1, 17, 129, 641, 4096, 16384, 32768}.issubset(
            {sum(check["lengths"]) for check in receipt["checks"]}
        )
        or receipt.get("graph_replay_passed") is not True
        or receipt.get("isolation_passed") is not True
        or receipt.get("zero_passed") is not True
        or receipt.get("sources") is None
    ):
        raise ValueError("incomplete direct GDN output native admission")


def validate_runtime(receipt: dict, worker_pid: int, validation_path: str) -> None:
    """Require direct destination use from every GDN layer in this worker."""
    if (
        receipt.get("passed") is not True
        or receipt.get("worker_pid") != worker_pid
        or receipt.get("validation_path") != validation_path
        or {call["layer"] for call in receipt.get("calls", []) if call["direct"]}
        != set(range(24))
    ):
        raise ValueError("incomplete direct GDN output live dispatch")
