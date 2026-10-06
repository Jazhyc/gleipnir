"""Forward-only causal D256 cuDNN MXFP8 over a BF16 paged serving cache.

Gather and quantize K/V together, emitting only forward payloads and packed SF
atoms. The cache remains BF16; packing and allocation are inside measured calls.
"""

import hashlib
from functools import lru_cache
from pathlib import Path
from unittest.mock import patch

import torch
import triton
import triton.language as tl

from gleipnir.nvidia_mxfp8_attention import _cudnn
from gleipnir.nvidia_mxfp8_fused_quantize import _scale
from gleipnir.serving_mxfp8_source import hardware_exp2_source

MAX_BATCH = 128
MAX_LENGTH = 32768


@triton.jit(do_not_specialize=["BATCH"])
def _produce(
    X,
    OUT,
    SF,
    CU,
    TABLE,
    CAPACITY,
    TOTAL,
    STRIDE_B,
    STRIDE_H,
    STRIDE_S,
    TABLE_STRIDE,
    HEADS: tl.constexpr,
    BATCH,
    PAGE: tl.constexpr,
    PAGED: tl.constexpr,
    COLUMN: tl.constexpr,
):
    tile, head, batch = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    start, end = tl.load(CU + batch), tl.load(CU + batch + 1)
    length = end - start
    if tile * 128 < length:
        rows = tile * 128 + tl.arange(0, 128)
        cols = tl.arange(0, 256)
        if PAGED:
            pages = tl.load(
                TABLE + batch * TABLE_STRIDE + rows // PAGE, rows < length, other=0
            )
            ptr = (
                X
                + pages[:, None].to(tl.int64) * STRIDE_B
                + head * STRIDE_H
                + (rows[:, None] % PAGE) * STRIDE_S
                + cols[None, :]
            )
        else:
            ptr = (
                X
                + (start + rows[:, None]).to(tl.int64) * STRIDE_S
                + head * STRIDE_H
                + cols[None, :]
            )
        value = tl.load(ptr, rows[:, None] < length, other=0).to(tl.float32)
        if COLUMN:
            groups = tl.reshape(value, (4, 32, 256))
            sf, inverse = _scale(tl.max(tl.abs(groups), 1))
            payload = tl.reshape(groups * inverse[:, None, :], (128, 256))
            scales = tl.reshape(tl.trans(sf), (1024,))
        else:
            groups = tl.reshape(value, (128, 8, 32))
            sf, inverse = _scale(tl.max(tl.abs(groups), 2))
            payload = tl.reshape(groups * inverse[:, :, None], (128, 256))
            scales = tl.reshape(sf, (1024,))
        # Initialize the final partial tile, even with an overallocated KV
        # bound, so a masked PV contraction cannot encounter poisoned tails.
        live = (rows < length) | ((batch == BATCH - 1) & (start + rows < TOTAL))
        tl.store(
            OUT
            + ((start + rows[:, None]).to(tl.int64) * HEADS + head) * 256
            + cols[None, :],
            payload.to(tl.float8e4nv),
            live[:, None],
        )
        previous = tl.arange(0, 128)
        a = tl.load(CU + previous, previous < batch, other=0)
        b = tl.load(CU + previous + 1, previous < batch, other=0)
        prefix = tl.sum(tl.cdiv(b - a, 128), 0)
        i = tl.arange(0, 1024)
        if COLUMN:
            d = i // 512 * 128 + (i // 4 % 4) * 32 + i // 16 % 32
            index = d * 4 + i % 4
        else:
            r = (i // 4 % 4) * 32 + i // 16 % 32
            c = i // 512 * 4 + i % 4
            index = r * 8 + c
        packed = tl.gather(scales, index, 0)
        tl.store(SF + (head * CAPACITY + prefix + tile) * 1024 + i, packed)


def produce(
    source: torch.Tensor,
    cumulative: torch.Tensor,
    maximum: int,
    *,
    table: torch.Tensor | None = None,
    total: int | None = None,
    column: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pack one orientation, directly gathering HND cache pages when supplied."""
    batch = cumulative.numel() - 1
    if not 1 <= batch <= MAX_BATCH or not 1 <= maximum <= MAX_LENGTH:
        raise ValueError("MXFP8 serving envelope exceeded")
    if source.dtype != torch.bfloat16 or source.stride(-1) != 1:
        raise ValueError("MXFP8 producer requires BF16 with contiguous D")
    if source.shape[-1] != 256 or cumulative.dtype != torch.int32:
        raise ValueError("MXFP8 producer requires D256 and int32 cumulative lengths")
    paged = table is not None
    heads = source.shape[1]
    if paged:
        if source.ndim != 4 or total is None or total < batch:
            raise ValueError(
                "paged MXFP8 requires HND cache and a bounded token capacity"
            )
        stride_b, stride_h, stride_s, _ = source.stride()
        page = source.shape[2]
        table_stride = table.stride(0)
    else:
        if source.ndim != 3:
            raise ValueError("query MXFP8 requires THD input")
        total = source.shape[0]
        stride_s, stride_h, _ = source.stride()
        stride_b, page, table_stride = 0, 1, 0
    capacity = triton.cdiv(total, 128) + batch
    payload = torch.empty(
        (total, heads, 256), dtype=torch.float8_e4m3fn, device=source.device
    )
    scales = torch.empty(
        heads * capacity * 1024, dtype=torch.uint8, device=source.device
    )
    _produce[(triton.cdiv(maximum, 128), heads, batch)](
        source,
        payload,
        scales,
        cumulative,
        table if paged else cumulative,
        capacity,
        total,
        stride_b,
        stride_h,
        stride_s,
        table_stride,
        heads,
        batch,
        page,
        paged,
        column,
        num_warps=8,
        enable_fp_fusion=False,
    )
    return payload, scales


@triton.jit
def _token_offsets(LENGTHS, OFFSETS, BATCH):
    rows = tl.arange(0, 128)
    lengths = tl.load(LENGTHS + rows, rows < BATCH, other=0)
    offsets = tl.cumsum(lengths, 0)
    tl.store(OFFSETS, 0)
    tl.store(OFFSETS + rows + 1, offsets, rows < BATCH)


@lru_cache(maxsize=4)
def forward_plan(device: torch.device):
    """Compile packed query/history with bottom-right causal alignment."""
    from cudnn.api_base import TensorDesc
    from cudnn.frost.template_loader import load_template
    from cudnn.sdpa.fwd import SdpaFwdDslSm100, api_dsl

    _cudnn()

    def desc(heads, dtype):
        return TensorDesc(
            dtype,
            (MAX_BATCH, heads, MAX_LENGTH, 256),
            (MAX_LENGTH * heads * 256, 256, heads * 256, 1),
            (3, 1, 2, 0),
            device,
        )

    original_loader = api_dsl._load_sm100_kernel_module

    def loader(flavor, params, **kwargs):
        original = original_loader(flavor, params, **kwargs)
        if flavor != (256, 256) or kwargs.get("pertensor") or kwargs.get("rubin"):
            raise ValueError("MXFP8 hardware-exp2 variant requires SM100 D256")
        source = hardware_exp2_source(Path(original.__file__).read_text())
        digest = hashlib.sha256(source.encode()).hexdigest()
        target = (
            Path(__file__).resolve().parents[2]
            / ".cache/kernels/nvidia_mxfp8/generated"
            / ("serving_exp2_" + digest + ".py")
        )
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            target.write_text(source)
        elif target.read_text() != source:
            raise ValueError("MXFP8 derived-source identity drift")
        return load_template(str(target), params, tag="gleipnir_serving_exp2_" + digest)

    with patch.object(api_dsl, "_load_sm100_kernel_module", loader):
        plan = SdpaFwdDslSm100(
            desc(16, torch.float8_e4m3fn),
            desc(4, torch.float8_e4m3fn),
            desc(4, torch.float8_e4m3fn),
            desc(16, torch.bfloat16),
            is_causal=True,
            causal_bottom_right=True,
            scale_softmax=0.0625,
            thd=True,
            seq_q_lens_present=False,
            seq_kv_lens_present=True,
            cu_seq_q_lens=True,
            cu_seq_kv_lens=True,
            cga=1,
            split_kv=1,
            dtype_o=torch.bfloat16,
        )
        plan.check_support()
        plan.compile()
    return plan


def paged_forward(
    *,
    query: torch.Tensor,
    kv_cache: torch.Tensor,
    block_tables: torch.Tensor,
    cum_seq_lens_q: torch.Tensor,
    cum_seq_lens_kv: torch.Tensor,
    seq_lens: torch.Tensor,
    max_q_len: int,
    max_kv_len: int,
    batch_size: int,
    out: torch.Tensor,
    **kwargs,
) -> torch.Tensor:
    """Adapt only the audited BF16 causal TRTLLM prefill call; never fall back."""
    if (
        query.shape[1:] != (16, 256)
        or query.dtype != torch.bfloat16
        or kv_cache.dtype != torch.bfloat16
        or kv_cache.ndim != 5
        or kv_cache.shape[1:3] != (2, 4)
        or out.dtype != torch.bfloat16
        or batch_size != cum_seq_lens_q.numel() - 1
        or cum_seq_lens_kv.numel() != batch_size + 1
        or seq_lens.shape != (batch_size,)
        or seq_lens.dtype != torch.int32
    ):
        raise ValueError(
            "MXFP8 serving requires BF16 causal D256 GQA 16/4 and HND cache"
        )
    for name, neutral in (
        ("bmm1_scale", 0.0625),
        ("bmm2_scale", 1.0),
        ("window_left", -1),
        ("sinks", None),
        ("o_sf_scale", None),
        ("kv_cache_sf", None),
        ("causal", True),
    ):
        if kwargs.get(name, neutral) != neutral:
            raise ValueError(f"unsupported MXFP8 serving option: {name}")
    key, value = kv_cache.unbind(1)
    # TRTLLM's cum_seq_lens_kv counts cache pages. cuDNN's packed THD
    # offsets count tokens; derive them from exact lengths, including tails.
    token_offsets = torch.empty_like(cum_seq_lens_kv)
    _token_offsets[(1,)](seq_lens, token_offsets, batch_size)
    qr, sfq = produce(query, cum_seq_lens_q, max_q_len)
    capacity = batch_size * max_kv_len
    kr, sfk = produce(
        key, token_offsets, max_kv_len, table=block_tables, total=capacity
    )
    vc, sfv = produce(
        value,
        token_offsets,
        max_kv_len,
        table=block_tables,
        total=capacity,
        column=True,
    )
    plan = forward_plan(query.device)
    workspace = torch.empty(
        max(1, plan.scratch_workspace_bytes()), device=query.device, dtype=torch.uint8
    )
    plan.execute(
        qr,
        kr,
        vc,
        out,
        seq_q_lens=cum_seq_lens_q,
        seq_kv_lens=token_offsets,
        sf_q=sfq,
        sf_k=sfk,
        sf_v=sfv,
        workspace=workspace,
    )
    return out
