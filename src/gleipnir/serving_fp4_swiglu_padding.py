"""Single-launch padding of native SwiGLU operands, including scale bytes."""

import triton
import triton.language as tl


@triton.jit(do_not_specialize=["ROWS", "PADDED", "SF_COUNT"])
def _copy(
    Q, SF, INV, OUT_Q, OUT_SF, OUT_INV, ROWS, PADDED, SF_COUNT, BLOCK: tl.constexpr
):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    q = tl.load(Q + i, i < ROWS * 1280, other=0)
    tl.store(OUT_Q + i, q, i < PADDED * 1280)
    sf = tl.load(SF + i, i < SF_COUNT, other=0)
    tl.store(OUT_SF + i, sf, i < PADDED * 160)
    inv = tl.load(INV + i, i < ROWS, other=1)
    tl.store(OUT_INV + i, inv, i < PADDED)


def copy_padded(a, codes, sf, inv):
    """Copy existing physical scale layout unchanged and zero its new padding."""
    _copy[(triton.cdiv(codes.numel(), 1024),)](
        a.codes.view(codes.dtype),
        a.scales,
        a.inverse,
        codes,
        sf,
        inv,
        a.codes.shape[0],
        codes.shape[0],
        a.scales.numel(),
        1024,
    )
