"""Prepared metadata must preserve boundaries and reject stale certificates."""

import sys
from types import SimpleNamespace

import pytest
import torch

from gleipnir.packed_sequences import PackedSequenceLayout, packed_fa4_interface
from gleipnir.training_hotpath import (
    normalize_qk_without_input_copy,
    prepared_fa4_interface,
    prepared_kernel_kwargs,
)


@pytest.mark.parametrize("lengths", [(1, 2, 3), (63, 65, 256), (1025,), (7, 1, 9)])
def test_prepared_boundaries_and_chunks_match_original(lengths):
    layout = PackedSequenceLayout(lengths)
    original = layout.kernel_kwargs()
    actual = prepared_kernel_kwargs(layout, "cpu", chunk_size=64)
    for key, value in original.items():
        if isinstance(value, torch.Tensor):
            assert torch.equal(value, actual[key])
            assert value.dtype == actual[key].dtype
        else:
            assert value == actual[key]
    cumulative = actual["cu_seq_lens_q"]
    assert cumulative is actual["cu_seq_lens_k"]
    size, offsets, count, indices = cumulative._flash_qla_prepared_varlen
    counts = [(n + 63) // 64 for n in lengths]
    assert size == 64 and count == sum(counts)
    assert offsets.tolist() == list(PackedSequenceLayout(tuple(counts)).offsets)
    assert indices.tolist() == [
        [i, j] for i, n in enumerate(counts) for j in range(n)
    ]


def test_fa4_prepared_route_never_reads_tensor_values(monkeypatch):
    kwargs = prepared_kernel_kwargs(
        PackedSequenceLayout((1, 2, 3)), "cpu", chunk_size=64
    )
    q = torch.randn(1, 2, 6, 4, requires_grad=True)
    calls = []

    def kernel(query, key, value, **options):
        calls.append(options)
        return query + key + value

    def forbidden(*args, **kwargs):
        raise AssertionError("unexpected host read or fallback")

    monkeypatch.setattr(torch.Tensor, "tolist", forbidden)
    monkeypatch.setattr(torch, "equal", forbidden)
    output, _ = prepared_fa4_interface(forbidden, kernel)(
        None, q, q, q, None, **kwargs
    )
    assert output.shape == (1, 6, 2, 4)
    output.sum().backward()
    assert q.grad is not None and bool((q.grad == 3).all())
    assert calls[0]["causal"] is True
    assert calls[0]["cu_seqlens_q"] is kwargs["cu_seq_lens_q"]


@pytest.mark.parametrize("change", ["mutate", "different_k", "clone"])
def test_fa4_stale_or_untrusted_metadata_uses_original_validation(change):
    kwargs = prepared_kernel_kwargs(PackedSequenceLayout((2, 3)), "cpu", chunk_size=64)
    cumulative = kwargs["cu_seq_lens_q"]
    if change == "mutate":
        cumulative.view(-1)[-1] += 1
    elif change == "different_k":
        kwargs["cu_seq_lens_k"] = cumulative.clone()
        kwargs["cu_seq_lens_k"][-1] += 1
    else:
        kwargs["cu_seq_lens_q"] = cumulative.clone()
        kwargs["cu_seq_lens_q"][-1] += 1
    q = torch.randn(1, 2, 5, 4)

    def kernel(*args, **kwargs):
        raise AssertionError("invalid metadata reached kernel")

    original = packed_fa4_interface(None, kernel)
    with pytest.raises(ValueError):
        prepared_fa4_interface(original, kernel)(None, q, q, q, None, **kwargs)


@pytest.mark.parametrize("field,value", [("max_length_q", 4), ("dropout", 0.1)])
def test_prepared_route_rejects_inconsistent_metadata(field, value):
    kwargs = prepared_kernel_kwargs(PackedSequenceLayout((2, 3)), "cpu", chunk_size=64)
    kwargs[field] = value
    q = torch.randn(1, 2, 5, 4)
    with pytest.raises(ValueError):
        prepared_fa4_interface(None, None)(None, q, q, q, None, **kwargs)


def test_normalization_retains_fp32_output_without_materializing_input(monkeypatch):
    x = torch.randn(3, 128, dtype=torch.bfloat16, requires_grad=True)
    calls = []

    def l2norm(input, *, output_dtype):
        calls.append((input, output_dtype))
        return input.float() / input.float().square().sum(-1, keepdim=True).sqrt()

    monkeypatch.setitem(
        sys.modules, "fla.modules.l2norm", SimpleNamespace(l2norm=l2norm)
    )
    y = normalize_qk_without_input_copy(x)
    y.square().sum().backward()
    assert calls[0][0] is x and calls[0][1] == torch.float32
    assert y.dtype == torch.float32
    assert x.grad is not None and x.grad.dtype == torch.bfloat16
