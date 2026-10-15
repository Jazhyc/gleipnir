"""Final-layer attention for causal LAST pooling: only each request's last query.

With causal LAST pooling, the final layer's output is read only at each request's
final token. Its keys/values are still written for every token (vLLM's separate
KV-cache update is untouched), but attention runs for one query per prefill
request, plus unchanged decode-path rows. Other output rows are zero; their
downstream rows are never pooled. Row counts and per-row arithmetic are unchanged,
so the selected rows keep the original kernels and inputs.

With `finished`, prefill requests whose prompt does not complete in this step are
skipped entirely: their final-layer output is never pooled, so neither their
attention nor the history packing it needs is computed.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from typing import Any

import numpy as np
import torch


@dataclasses.dataclass
class LastQueryState:
    """Rows attended in the current forward, shared with a trimmed layer tail."""

    rows: torch.Tensor | None = None
    context: object | None = None


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


def select_prefills(prefill: Any, indices: torch.Tensor, max_seq_len: int) -> Any:
    """One query per chosen prefill request, keeping page offsets consistent."""
    pages = (prefill.cum_seq_lens_kv[1:] - prefill.cum_seq_lens_kv[:-1]).index_select(
        0, indices
    )
    cumulative = torch.zeros(
        indices.numel() + 1, dtype=prefill.cum_seq_lens_kv.dtype, device=pages.device
    )
    torch.cumsum(pages, 0, out=cumulative[1:])
    return dataclasses.replace(
        prefill,
        block_tables=prefill.block_tables.index_select(0, indices),
        seq_lens=prefill.seq_lens.index_select(0, indices),
        cum_seq_lens_kv=cumulative,
        cum_seq_lens_q=torch.arange(
            indices.numel() + 1, dtype=torch.int32, device=pages.device
        ),
        max_q_len=1,
        max_seq_len=max_seq_len,
    )


def install(
    layer: torch.nn.Module,
    *,
    prefill_type: type,
    observe: Callable[[dict], None] | None = None,
    finished: Callable[[], tuple[np.ndarray, np.ndarray]] | None = None,
    state: LastQueryState | None = None,
    current_context: Callable[[], object] | None = None,
) -> None:
    """Wrap one attention layer's backend forward; never fall back silently.

    `finished` returns CPU (prompt lengths, sequence lengths after this step) in
    batch order, decode requests first, matching the attention metadata.
    """
    impl = layer.impl
    if getattr(impl, "_gleipnir_last_query", False):
        raise ValueError("last-query attention is already installed")
    original = impl.forward

    def publish(rows: torch.Tensor | None) -> None:
        if state is not None:
            state.rows = rows
            state.context = current_context() if current_context else None

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
            publish(
                None
                if attn_metadata is None
                else torch.arange(
                    attn_metadata.num_actual_tokens,
                    device=query.device,
                    dtype=torch.int64,
                )
            )
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
        kept = prefills
        if finished is not None:
            prompt, sequence = finished()
            decodes = attn_metadata.num_decodes
            # vLLM's max_seq_len spans the whole batch, decodes included.
            if prompt.shape[0] != decodes + prefills or (
                int(sequence.max()) != prefill.max_seq_len
            ):
                raise ValueError("runner request state does not match attention batch")
            done = np.flatnonzero(prompt[decodes:] == sequence[decodes:])
            kept = int(done.size)
            if kept < prefills:
                indices = torch.from_numpy(done).to(rows.device)
                rows = torch.cat([rows[:decode_tokens], rows[decode_tokens:][indices]])
                if kept:
                    prefill = select_prefills(
                        prefill, indices, int(sequence[decodes:][done].max())
                    )
        if kept == prefills:
            prefill = dataclasses.replace(
                prefill,
                cum_seq_lens_q=torch.arange(
                    prefills + 1, dtype=torch.int32, device=query.device
                ),
                max_q_len=1,
            )
        actual = output[: attn_metadata.num_actual_tokens]
        actual.zero_()
        if rows.numel():
            compact_query = query.index_select(0, rows)
            compact_output = torch.empty(
                (rows.numel(), *output.shape[1:]),
                dtype=output.dtype,
                device=output.device,
            )
            # Bottom-right causal alignment: one query at each request's final
            # position attends to that request's whole cached history.
            metadata = dataclasses.replace(
                attn_metadata,
                num_actual_tokens=rows.numel(),
                num_prefills=kept,
                num_prefill_tokens=kept,
                prefill=prefill if kept else None,
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
            actual.index_copy_(0, rows, compact_output)
        publish(rows)
        if observe is not None:
            observe(
                {
                    "tokens": attn_metadata.num_actual_tokens,
                    "decode_tokens": decode_tokens,
                    "prefills": prefills,
                    "attended_prefills": kept,
                    "max_query_tokens": attn_metadata.prefill.max_q_len,
                    "max_history_tokens": attn_metadata.prefill.max_seq_len,
                }
            )
        return output

    impl.forward = forward
    impl._gleipnir_last_query = True
