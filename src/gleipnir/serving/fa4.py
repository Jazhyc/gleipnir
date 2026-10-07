"""Translate the supported vLLM paged forward API to pinned native FA4."""

from typing import Any


def subdivide_paged_kv(k: Any, v: Any, table: Any, page_size: int = 128) -> tuple:
    """Expose NHD hybrid cache pages as smaller views, including K/V gaps."""
    import torch

    physical = k.shape[1]
    if physical == page_size:
        return k, v, table
    if physical % page_size or k.shape != v.shape or k.ndim != 4:
        raise ValueError("unsupported FA4 physical page geometry")
    heads, dimension = k.shape[2:]
    token_stride = heads * dimension
    page_stride = page_size * token_stride
    expected = (token_stride, dimension, 1)
    if (
        k.stride()[1:] != expected
        or v.stride() != k.stride()
        or k.stride(0) % page_stride
        or k.stride(0) < physical * token_stride
    ):
        raise ValueError("FA4 subdivision requires the NHD cache layout")
    subdivisions = physical // page_size
    spacing = k.stride(0) // page_stride
    # Hybrid caches interleave whole physical K/V pages. Expose those gaps as
    # unused native pages; the remapped table only indexes each valid K/V span.
    count = (k.shape[0] - 1) * spacing + subdivisions
    shape = (count, page_size, heads, dimension)
    strides = (page_stride, *expected)
    small_k = k.as_strided(shape, strides)
    small_v = v.as_strided(shape, strides)
    offsets = torch.arange(subdivisions, dtype=table.dtype, device=table.device)
    small_table = (table.unsqueeze(-1) * spacing + offsets).flatten(1)
    return small_k, small_v, small_table


def make_paged_fa4_forward(native: Any) -> Any:
    def forward(q: Any, k: Any, v: Any, **kwargs: Any) -> Any:
        if kwargs.pop("fa_version", 4) != 4:
            raise ValueError("external FA4 received another version")
        for name, neutral in (
            ("dropout_p", 0),
            ("alibi_slopes", None),
            ("dynamic_causal", None),
            ("output_scale", None),
            ("cp_world_size", 1),
            ("cp_rank", 0),
            ("cp_tot_seqused_k", None),
        ):
            value = kwargs.pop(name, neutral)
            if value is not None and value != neutral and value is not False:
                raise ValueError(f"unsupported external FA4 option: {name}")
        window = kwargs.pop("window_size", (-1, -1))
        return_lse = kwargs.pop("return_softmax_lse", False)
        kwargs["page_table"] = kwargs.pop("block_table", None)
        kwargs["learnable_sink"] = kwargs.pop("s_aux", None)
        kwargs["return_lse"] = return_lse
        kwargs["window_size_left"] = None if window[0] < 0 else window[0]
        kwargs["window_size_right"] = None if window[1] < 0 else window[1]
        # vLLM supplies these tensors even for BF16; its own FA4 branch ignores
        # them. The worker separately requires unit BF16 layer scales.
        for name in ("q_descale", "k_descale", "v_descale"):
            kwargs.pop(name, None)
        if kwargs["page_table"] is not None and hasattr(k, "shape"):
            k, v, kwargs["page_table"] = subdivide_paged_kv(k, v, kwargs["page_table"])
        output, lse, _, _, _ = native(q, k, v, **kwargs)
        return (output, lse) if return_lse else output

    return forward
