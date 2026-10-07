"""Opt-in inference activation producers for the pinned FROST FP4 recipe."""

import torch
import triton
import triton.language as tl

from gleipnir.cudnn_fp4_gemm import PackedNvfp4


@triton.jit(do_not_specialize=["M"])
def _vendor_finish(SCALE, INV, SF, Q, M, K: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    scale = tl.load(SCALE + i, i < M, other=0)
    inverse = tl.where(scale > 0, tl.div_rn(1.0, scale), 1.0)
    tl.store(INV + i, inverse, i < M)
    zero = scale == 0
    # The vendor producer emits invalid SFs for all-zero rows. Repair their
    # exact mathematical zero payload, and initialize scale padding too.
    if tl.sum(zero.to(tl.int32), 0) > 0:
        for start in range(tl.cdiv(K // 16, 64)):
            g = start * 64 + tl.arange(0, 64)
            offset = (
                (
                    (i[:, None] // 128 * (K // 64) + g[None, :] // 4) * 32
                    + i[:, None] % 32
                )
                * 4
                + (i[:, None] % 128) // 32
            ) * 4 + g[None, :] % 4
            tl.store(SF + offset, 0.0, zero[:, None] & (g[None, :] < K // 16))
        for start in range(tl.cdiv(K // 2, 256)):
            j = start * 256 + tl.arange(0, 256)
            tl.store(
                Q + i[:, None] * (K // 2) + j[None, :],
                0,
                zero[:, None] & (i[:, None] < M) & (j[None, :] < K // 2),
            )


def vendor_pack(x: torch.Tensor) -> PackedNvfp4:
    """Use installed CUDA per-token NVFP4, with explicit inverse/tail handling."""
    from flashinfer.quantization import SfLayout, nvfp4_quantize

    if (
        x.ndim != 2
        or x.dtype != torch.bfloat16
        or not x.is_cuda
        or not x.is_contiguous()
        or x.shape[1] not in {2560, 4096, 9216}
        or x.shape[0] < 1
    ):
        raise ValueError("vendor FROST packing requires supported BF16 activation rows")
    # A host constant avoids the pinned wrapper's GPU tensor .item() path.
    codes, scales, token_scale = nvfp4_quantize(
        x,
        1.0 / 2688.0,
        sfLayout=SfLayout.layout_128x4,
        backend="cuda",
        per_token_activation=True,
        enable_pdl=False,
    )
    m, k = x.shape
    scales = scales.view(torch.float8_e4m3fn).reshape(
        triton.cdiv(m, 128) * 128, k // 16
    )
    inverse = torch.empty(m, device=x.device, dtype=torch.float32)
    _vendor_finish[(triton.cdiv(m, 128) * 8,)](
        token_scale, inverse, scales, codes, m, k, 16, num_warps=4
    )
    return PackedNvfp4(codes.view(torch.float4_e2m1fn_x2), scales, inverse)
