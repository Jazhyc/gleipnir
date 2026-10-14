"""Final-layer attention for causal LAST pooling: only each request's last query.

With causal LAST pooling, the final layer's output is read only at each request's
final token. Its keys/values are still written for every token (vLLM's separate
KV-cache update is untouched), but attention runs for one query per prefill
request, plus unchanged decode-path rows. Other output rows are zero; their
downstream rows are never pooled. Row counts and per-row arithmetic are unchanged,
so the selected rows keep the original kernels and inputs.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from typing import Any

import torch


def selected_rows(
    num_decode_tokens: int, cum_seq_lens_q: torch.Tensor, num_prefills: int
) -> torch.Tensor:
    """Decode rows first, then each prefill request's last token, as int64 rows."""
    if cum_seq_lens_q.numel() != num_prefills + 1:
        raise ValueError("prefill query offsets do not match the request count")
    last = cum_seq_lens_q[1:].to(torch.int64) - 1 + num_decode_tokens
    if num_decode_tokens == 0:
        return last
    decode = torch.arange(num_decode_tokens, device=last.device, dtype=torch.int64)
    return torch.cat([decode, last])


def install(
    layer: torch.nn.Module,
    *,
    prefill_type: type,
    observe: Callable[[dict], None] | None = None,
) -> None:
    """Wrap one attention layer's backend forward; never fall back silently."""
    impl = layer.impl
    if getattr(impl, "_gleipnir_last_query", False):
        raise ValueError("last-query attention is already installed")
    original = impl.forward

    def forward(
        attention_layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor | None,
        value: torch.Tensor | None,
        kv_cache: torch.Tensor,
        attn_metadata: Any,
        output: torch.Tensor,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if attn_metadata is None or attn_metadata.num_prefills == 0:
            return original(
                attention_layer,
                query,
                key,
                value,
                kv_cache,
                attn_metadata,
                output=output,
                output_scale=output_scale,
                output_block_scale=output_block_scale,
            )
        if (
            attn_metadata.use_cascade
            or not isinstance(attn_metadata.prefill, prefill_type)
            or output_scale is not None
            or output_block_scale is not None
        ):
            raise ValueError("last-query attention requires the TRTLLM prefill path")
        decode_tokens = attn_metadata.num_decode_tokens
        prefills = attn_metadata.num_prefills
        prefill = attn_metadata.prefill
        rows = selected_rows(decode_tokens, prefill.cum_seq_lens_q, prefills)
        compact_query = query.index_select(0, rows)
        compact_output = torch.empty(
            (rows.numel(), *output.shape[1:]), dtype=output.dtype, device=output.device
        )
        # Bottom-right causal alignment: one query at each request's final
        # position attends to that request's whole cached history.
        metadata = dataclasses.replace(
            attn_metadata,
            num_actual_tokens=rows.numel(),
            num_prefill_tokens=prefills,
            prefill=dataclasses.replace(
                prefill,
                cum_seq_lens_q=torch.arange(
                    prefills + 1, dtype=torch.int32, device=query.device
                ),
                max_q_len=1,
            ),
        )
        original(
            attention_layer,
            compact_query,
            None,
            None,
            kv_cache,
            metadata,
            output=compact_output,
        )
        actual = output[: attn_metadata.num_actual_tokens]
        actual.zero_()
        actual.index_copy_(0, rows, compact_output)
        if observe is not None:
            observe(
                {
                    "tokens": attn_metadata.num_actual_tokens,
                    "decode_tokens": decode_tokens,
                    "prefills": prefills,
                    "max_query_tokens": prefill.max_q_len,
                    "max_history_tokens": prefill.max_seq_len,
                }
            )
        return output

    impl.forward = forward
    impl._gleipnir_last_query = True
