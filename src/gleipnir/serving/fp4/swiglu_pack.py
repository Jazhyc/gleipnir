"""Pack the fused kernel's BF16 activation with the validated rowwise emitter."""

import torch
import triton
import triton.language as tl

from gleipnir.cudnn_fp4_gemm import PackedNvfp4
from gleipnir.serving_fp4_fusion import _buffers, _emit


@triton.jit(do_not_specialize=["M"])
def _activated_pack(X, Q, SF, INV, M, K: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    j = tl.arange(0, BLOCK)
    value = tl.load(X + row * K + j, (row < M) & (j < K), other=0).to(tl.float32)
    _emit(value, Q, SF, INV, row, M, K, BLOCK)


def activated_pack(x: torch.Tensor) -> PackedNvfp4:
    """Retain whole-row amax, BF16 input and hardware FP4 RNE conversion."""
    if x.ndim != 2 or x.shape[1] != 9216:
        raise ValueError("fused SwiGLU packing requires the pinned width 9216")
    q, sf, inv = _buffers(x, x.shape[1])
    _activated_pack[(triton.cdiv(x.shape[0], 128) * 128,)](
        x,
        q,
        sf,
        inv,
        x.shape[0],
        x.shape[1],
        triton.next_power_of_2(x.shape[1]),
        num_warps=8,
        enable_fp_fusion=False,
    )
    return PackedNvfp4(q.view(torch.float4_e2m1fn_x2), sf, inv)
