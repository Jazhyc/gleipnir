"""Last-query attention keeps decode rows, picks final prefill rows, fails closed."""

from dataclasses import dataclass
from types import SimpleNamespace

import pytest
import torch

from gleipnir.serving.last_query_attention import install, selected_rows


@dataclass
class Prefill:
    cum_seq_lens_q: torch.Tensor
    max_q_len: int
    max_seq_len: int


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
