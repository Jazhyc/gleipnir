"""Preserve causal paged lengths and buffers when translating native APIs."""

import pytest

from gleipnir.serving_fa4 import make_paged_fa4_forward, subdivide_paged_kv


def test_subdivision_preserves_interleaved_cache_and_shuffled_page_payloads():
    import torch

    cache = torch.arange(3 * 2 * 640 * 4 * 8).reshape(3, 2, 640, 4, 8)
    k, v = cache.unbind(1)
    table = torch.tensor([[2, 0], [1, 2]], dtype=torch.int32)
    small_k, small_v, mapped = subdivide_paged_kv(k, v, table)
    assert mapped.tolist() == [
        [20, 21, 22, 23, 24, 0, 1, 2, 3, 4],
        [10, 11, 12, 13, 14, 20, 21, 22, 23, 24],
    ]
    for original, small in ((k, small_k), (v, small_v)):
        assert (
            original.untyped_storage().data_ptr() == small.untyped_storage().data_ptr()
        )
        assert torch.equal(original[table].flatten(1, 2), small[mapped].flatten(1, 2))
    with pytest.raises(ValueError, match="NHD"):
        subdivide_paged_kv(k.transpose(1, 2).contiguous().transpose(1, 2), v, table)


def test_external_fa4_preserves_paged_lengths_causal_mask_and_output_buffer():
    calls = []

    def native(*args, **kwargs):
        calls.append((args, kwargs))
        return "output", "lse", None, None, None

    wrapper = make_paged_fa4_forward(native)
    assert wrapper(
        "q",
        "k",
        "v",
        block_table="pages",
        seqused_k="lengths",
        cu_seqlens_q="queries",
        causal=True,
        fa_version=4,
        out="buffer",
        window_size=(-1, -1),
        return_softmax_lse=True,
        q_descale="unit_q",
        k_descale="unit_k",
        v_descale="unit_v",
    ) == ("output", "lse")
    call = calls[0][1]
    assert call["page_table"] == "pages" and call["seqused_k"] == "lengths"
    assert call["cu_seqlens_q"] == "queries" and call["causal"] is True
    assert call["out"] == "buffer" and call["window_size_left"] is None
    assert not {"q_descale", "k_descale", "v_descale"} & call.keys()
    with pytest.raises(ValueError, match="another version"):
        wrapper("q", "k", "v", fa_version=2)
    with pytest.raises(ValueError, match="unsupported"):
        wrapper("q", "k", "v", dropout_p=0.1)
