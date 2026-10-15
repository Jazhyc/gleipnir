"""Last-query attention keeps decode rows, picks final prefill rows, fails closed."""

from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from gleipnir.serving.last_query_attention import (
    LastQueryState,
    install,
    select_prefills,
    selected_rows,
)
from gleipnir.serving.last_query_tail import tail


@dataclass
class Prefill:
    cum_seq_lens_q: torch.Tensor
    max_q_len: int
    max_seq_len: int
    block_tables: torch.Tensor | None = None
    seq_lens: torch.Tensor | None = None
    cum_seq_lens_kv: torch.Tensor | None = None


@dataclass
class OtherPrefill:
    pass


@dataclass
class Metadata:
    num_actual_tokens: int
    num_decode_tokens: int
    num_prefills: int
    num_prefill_tokens: int
    use_cascade: bool
    prefill: object


def offsets(*values: int) -> torch.Tensor:
    return torch.tensor(values, dtype=torch.int32)


def test_selected_rows_follow_decode_then_prefill_layout():
    assert selected_rows(0, offsets(0, 5, 6, 10), 3).tolist() == [4, 5, 9]
    assert selected_rows(2, offsets(0, 3, 7), 2).tolist() == [0, 1, 4, 8]
    with pytest.raises(ValueError):
        selected_rows(0, offsets(0, 5), 2)


def recording_layer():
    calls = []

    def forward(layer, query, key, value, kv_cache, metadata, output, **kwargs):
        calls.append((query.clone(), metadata, kwargs))
        # Mark each row so placement is observable.
        output.copy_(query + 1)
        return output

    return SimpleNamespace(impl=SimpleNamespace(forward=forward)), calls


def test_wrapper_attends_only_final_rows_and_zeroes_the_rest():
    layer, calls = recording_layer()
    observed = []
    install(layer, prefill_type=Prefill, observe=observed.append)
    tokens, padded = 12, 14
    query = torch.arange(padded * 2, dtype=torch.float32).reshape(padded, 1, 2)
    output = torch.full((padded, 1, 2), 7.0)
    metadata = Metadata(
        num_actual_tokens=tokens,
        num_decode_tokens=2,
        num_prefills=2,
        num_prefill_tokens=10,
        use_cascade=False,
        prefill=Prefill(offsets(0, 4, 10), max_q_len=6, max_seq_len=900),
    )
    layer.impl.forward(layer, query, query, query, None, metadata, output=output)
    compact, passed, kwargs = calls[0]
    rows = [0, 1, 5, 11]
    assert torch.equal(compact, query[rows])
    assert passed.num_actual_tokens == 4 and passed.num_prefill_tokens == 2
    assert passed.prefill.cum_seq_lens_q.tolist() == [0, 1, 2]
    assert passed.prefill.max_q_len == 1 and passed.prefill.max_seq_len == 900
    assert metadata.prefill.max_q_len == 6  # caller metadata is unchanged
    assert torch.equal(output[rows], query[rows] + 1)
    others = [i for i in range(tokens) if i not in rows]
    assert not output[others].any()
    assert torch.equal(output[tokens:], torch.full((2, 1, 2), 7.0))
    assert observed[0]["prefills"] == 2 and observed[0]["tokens"] == tokens


def prefill_batch(queries, histories):
    pages = [(n + 15) // 16 for n in histories]
    return Prefill(
        cum_seq_lens_q=offsets(0, *np.cumsum(queries).tolist()),
        max_q_len=max(queries),
        max_seq_len=max(histories),
        block_tables=torch.arange(len(pages) * 4).reshape(len(pages), 4),
        seq_lens=torch.tensor(histories, dtype=torch.int32),
        cum_seq_lens_kv=offsets(0, *np.cumsum(pages).tolist()),
    )


def test_select_prefills_keeps_page_offsets_consistent():
    prefill = prefill_batch([3, 4, 5], [40, 17, 100])
    chosen = select_prefills(prefill, torch.tensor([0, 2]), 100)
    assert chosen.block_tables.tolist() == [[0, 1, 2, 3], [8, 9, 10, 11]]
    assert chosen.seq_lens.tolist() == [40, 100]
    assert chosen.cum_seq_lens_kv.tolist() == [0, 3, 10]
    assert chosen.cum_seq_lens_q.tolist() == [0, 1, 2]
    assert chosen.max_q_len == 1 and chosen.max_seq_len == 100


def test_finished_only_attends_completing_prompts():
    layer, calls = recording_layer()
    state = LastQueryState()
    context = object()
    install(
        layer,
        prefill_type=Prefill,
        finished=lambda: (np.array([1, 50, 60, 9]), np.array([1, 50, 40, 9])),
        state=state,
        current_context=lambda: context,
    )
    query = torch.arange(13 * 2, dtype=torch.float32).reshape(13, 1, 2)
    output = torch.full((13, 1, 2), 7.0)
    metadata = Metadata(
        num_actual_tokens=13,
        num_decode_tokens=1,
        num_prefills=3,
        num_prefill_tokens=12,
        use_cascade=False,
        prefill=prefill_batch([5, 4, 3], [50, 40, 9]),
    )
    metadata.num_decodes = 1
    layer.impl.forward(layer, query, query, query, None, metadata, output=output)
    compact, passed, _ = calls[0]
    # Decode row 0, then prefill requests 0 and 2 (request 1 is mid-prompt).
    assert torch.equal(compact, query[[0, 5, 12]])
    assert passed.num_prefills == 2 and passed.num_actual_tokens == 3
    assert passed.prefill.seq_lens.tolist() == [50, 9]
    assert passed.prefill.max_seq_len == 50
    assert state.rows.tolist() == [0, 5, 12] and state.context is context
    assert not output[[i for i in range(13) if i not in (0, 5, 12)]].any()


def test_unfinished_prefill_only_batch_skips_attention():
    layer, calls = recording_layer()
    install(
        layer,
        prefill_type=Prefill,
        finished=lambda: (np.array([100]), np.array([40])),
    )
    query = torch.ones((8, 1, 2))
    output = torch.full((8, 1, 2), 3.0)
    metadata = Metadata(8, 0, 1, 8, False, prefill_batch([8], [40]))
    metadata.num_decodes = 0
    layer.impl.forward(layer, query, query, query, None, metadata, output=output)
    assert not calls and not output.any()


def test_mismatched_runner_state_fails_closed():
    layer, _ = recording_layer()
    install(
        layer, prefill_type=Prefill, finished=lambda: (np.array([9]), np.array([7]))
    )
    query = torch.ones((8, 1, 2))
    metadata = Metadata(8, 0, 1, 8, False, prefill_batch([8], [40]))
    metadata.num_decodes = 0
    with pytest.raises(ValueError, match="runner request state"):
        layer.impl.forward(
            layer, query, query, query, None, metadata, output=torch.empty_like(query)
        )


def fake_decoder():
    weight = torch.tensor([[1.0, 2.0], [3.0, 4.0]])

    def norm(hidden, residual):
        total = hidden + residual
        residual.copy_(total)  # fused norms update residual in place
        return total * 0.5, residual

    return SimpleNamespace(
        self_attn=SimpleNamespace(o_proj=lambda x: (x @ weight, None)),
        post_attention_layernorm=norm,
        mlp=lambda x: x * 2 + 1,
    )


def test_tail_matches_full_rows_and_never_mutates_inputs():
    layer = fake_decoder()
    attn = torch.randn(6, 2)
    gate = torch.randn(6, 2)
    residual = torch.randn(6, 2)
    original = residual.clone()
    full_hidden, full_residual = tail(layer, attn, gate, residual, None)
    assert torch.equal(residual, original)
    rows = torch.tensor([1, 4])
    hidden, kept = tail(layer, attn, gate, residual, rows)
    assert torch.equal(residual, original)
    assert torch.allclose(hidden[rows], full_hidden[rows])
    assert torch.allclose(kept[rows], full_residual[rows])
    assert not hidden[[0, 2, 3, 5]].any()
    assert torch.equal(kept[[0, 2, 3, 5]], original[[0, 2, 3, 5]])
    empty, unchanged = tail(
        layer, attn, gate, residual, torch.tensor([], dtype=torch.int64)
    )
    assert not empty.any() and torch.equal(unchanged, original)


def test_decode_only_batches_use_the_original_path():
    layer, calls = recording_layer()
    install(layer, prefill_type=Prefill)
    query = torch.ones((3, 1, 2))
    output = torch.empty_like(query)
    metadata = Metadata(3, 3, 0, 0, False, None)
    layer.impl.forward(layer, query, query, query, None, metadata, output=output)
    assert calls[0][1] is metadata and torch.equal(output, query + 1)


@pytest.mark.parametrize(
    "change",
    [
        {"use_cascade": True},
        {"prefill": OtherPrefill()},
    ],
)
def test_unsupported_paths_fail_closed(change):
    layer, _ = recording_layer()
    install(layer, prefill_type=Prefill)
    metadata = Metadata(4, 0, 1, 4, False, Prefill(offsets(0, 4), 4, 4))
    for key, value in change.items():
        setattr(metadata, key, value)
    query = torch.ones((4, 1, 2))
    with pytest.raises(ValueError, match="TRTLLM"):
        layer.impl.forward(
            layer, query, query, query, None, metadata, output=torch.empty_like(query)
        )


def test_fused_output_quantization_is_rejected_and_double_install_refused():
    layer, _ = recording_layer()
    install(layer, prefill_type=Prefill)
    metadata = Metadata(4, 0, 1, 4, False, Prefill(offsets(0, 4), 4, 4))
    query = torch.ones((4, 1, 2))
    with pytest.raises(ValueError):
        layer.impl.forward(
            layer,
            query,
            query,
            query,
            None,
            metadata,
            output=torch.empty_like(query),
            output_scale=torch.ones(1),
        )
    with pytest.raises(ValueError, match="already"):
        install(layer, prefill_type=Prefill)
