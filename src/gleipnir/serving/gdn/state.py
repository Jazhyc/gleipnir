"""BF16 FlashInfer state I/O with unchanged FP32 gates and MMA accumulation."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch


def make_forward(
    kernel: Callable[..., Any],
    normalize: Callable[[torch.Tensor], torch.Tensor],
    *,
    round_gates: bool = False,
    observed: Callable[[torch.Tensor, torch.Tensor | None], None] | None = None,
) -> Callable[..., tuple[torch.Tensor, torch.Tensor | None]]:
    """Pass BF16 state to the pinned SM100 path; gate rounding is diagnostic."""

    def forward(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        initial_state: torch.Tensor,
        output_final_state: bool,
        cu_seqlens: torch.Tensor | None = None,
        use_qk_l2norm_in_kernel: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if initial_state.dtype not in (torch.float32, torch.bfloat16):
            raise ValueError("GDN state must be FP32 or BF16")
        if use_qk_l2norm_in_kernel:
            q, k = normalize(q.contiguous()), normalize(k.contiguous())
        q, k, v = [x.squeeze(0).contiguous() for x in (q, k, v)]
        forget = torch.exp(g.squeeze(0).contiguous().float())
        update = beta.squeeze(0).contiguous().float()
        if round_gates:
            forget = forget.bfloat16().float()
            update = update.bfloat16().float()
        state = initial_state.to(torch.bfloat16).contiguous()
        final = torch.empty_like(state) if output_final_state else None
        result = kernel(
            q=q,
            k=k,
            v=v,
            g=forget,
            beta=update,
            initial_state=state,
            output_final_state=output_final_state,
            output_state=final,
            cu_seqlens=cu_seqlens,
        )
        value, final = result if output_final_state else (result, None)
        if final is not None and final.dtype != torch.bfloat16:
            raise ValueError("native GDN returned a non-BF16 state")
        if observed is not None:
            observed(initial_state, final)
        return value.unsqueeze(0), final

    return forward


def validate_native(receipt: dict) -> None:
    """Require long/ragged, continued-state, oracle and graph-replay admission."""
    checks = receipt.get("checks", [])
    if (
        receipt.get("passed") is not True
        or receipt.get("intervention") != "flashinfer_gdn_bf16_state"
        or receipt.get("relative_l2_limit") != 0.03
        or receipt.get("accumulation_dtype") != "float32"
        or receipt.get("gate_dtype") != "float32"
        or receipt.get("state_dtype") != "bfloat16"
        or not {1, 17, 129, 641, 4096, 16384, 32768}.issubset(
            {sum(c["lengths"]) for c in checks}
        )
        or not all(c.get("passed") is True for c in checks)
        or not all(
            receipt.get(key) is True
            for key in (
                "oracle_passed",
                "continuation_passed",
                "isolation_passed",
                "graph_replay_passed",
            )
        )
        or not receipt.get("sources")
    ):
        raise ValueError("incomplete BF16 GDN state native admission")
