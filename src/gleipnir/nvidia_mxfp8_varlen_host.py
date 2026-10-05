# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Packed, runtime-shaped host derived from NVIDIA's prepared MXFP8 host.

The original native dQ/dKdV kernels and their O2 compilation policy are retained.
Scale repacking and allocation belong to the explicit caller, outside this host.
This module is imported only by the isolated experimental NVIDIA backend.
"""

import cutlass
import cutlass.cute as cute
from cuda.bindings import driver
from cudnn.frost.compiled_cache import compile_cached


@cute.jit
def packed_backward_host(
    pointers: tuple,
    total: cutlass.Int32,
    maximum: cutlass.Int32,
    batch: cutlass.Int32,
    scale: cutlass.Float32,
    heads: cutlass.Constexpr,
    kernels: cutlass.Constexpr,
    stream: driver.CUstream,
):
    hq, hk = heads
    hr = hq // hk
    q_layout = cute.make_layout(
        (total, 256, hr, hk, 1), stride=(hq * 256, 1, 256, hr * 256, 0)
    )
    k_layout = cute.make_layout(
        (total, 256, 1, hk, 1), stride=(hk * 256, 1, 256, 256, 0)
    )
    tensors = []
    for i in cutlass.range_constexpr(13):
        layout = k_layout if cutlass.const_expr(i in (1, 2, 7, 8, 10)) else q_layout
        if cutlass.const_expr(i == 5):
            layout = cute.make_layout(
                (total, hr, hk, 1), stride=(1, total, total * hr, 0)
            )
        tensors.append(cute.make_tensor(pointers[i], layout))
    scales = []
    for i in cutlass.range_constexpr(11):
        scales.append(cute.make_tensor(pointers[13 + i], cute.make_layout((1,))))
    cu = cute.make_tensor(pointers[24], cute.make_layout((batch + 1,)))
    ws = cute.make_tensor(pointers[25], cute.make_layout((1,)))
    problem = (maximum, maximum, 256, ((hr, hk), batch))
    q, k, v, o, do, lse, dq, dk, dv, qt, kt, dot, do_half = tensors
    dq_kernel, dkdv_kernel = kernels
    dq_kernel(
        problem,
        q,
        k,
        kt,
        v,
        o,
        scales[0],
        scales[1],
        scales[2],
        scales[3],
        scales[4],
        dq,
        dk,
        dv,
        do,
        do_half,
        lse,
        cu,
        cu,
        scale,
        None,
        cutlass.Int32(0),
        ws,
        stream,
        False,
    )
    dkdv_kernel(
        problem,
        q,
        qt,
        k,
        v,
        o,
        scales[5],
        scales[6],
        scales[7],
        scales[8],
        scales[9],
        scales[10],
        dk,
        dv,
        do,
        dot,
        do_half,
        lse,
        cu,
        cu,
        scale,
        None,
        cutlass.Int32(0),
        ws,
        stream,
        True,
    )


def compile_packed_backward(heads: tuple[int, int], cache_key: str):
    """Compile once per head contract; no sequence extents enter the key."""
    from cudnn.sdpa.bwd.kernels.sm100 import _bprop_mxfp8_masks as masks
    from cudnn.sdpa.bwd.kernels.sm100.bprop_dkdv_d256_mxfp8 import (
        BlackwellFmhaBackwardDKDV256,
    )
    from cudnn.sdpa.bwd.kernels.sm100.bprop_dq_d256_mxfp8 import (
        BlackwellFmhaBackwardDQ256,
    )

    dq = BlackwellFmhaBackwardDQ256(
        cutlass.BFloat16,
        cutlass.Float32,
        (128, 128, 256),
        True,
        masks.MaskEnum.WINDOW_MASK,
        is_persistent=False,
        online_ds_scale=True,
        store_num_bits_per_copy=16,
    )
    dkdv = BlackwellFmhaBackwardDKDV256(
        cutlass.BFloat16,
        cutlass.Float32,
        (128, 128, 256),
        True,
        masks.MaskEnum.WINDOW_MASK_BWD,
        is_persistent=False,
        online_ds_scale=True,
        p_scale_log2=8,
    )
    fp8, bf16 = cutlass.Float8E4M3FN, cutlass.BFloat16
    types = (
        fp8,
        fp8,
        fp8,
        bf16,
        fp8,
        cutlass.Float32,
        bf16,
        bf16,
        bf16,
        fp8,
        fp8,
        fp8,
        bf16,
    )
    types += (cutlass.Float8E8M0FNU,) * 11 + (cutlass.Int32, cutlass.Uint8)
    pointers = tuple(
        cute.runtime.make_ptr(t, 16, cute.AddressSpace.gmem, assumed_align=16)
        for t in types
    )
    return compile_cached(
        packed_backward_host,
        pointers,
        cutlass.Int32(1),
        cutlass.Int32(1),
        cutlass.Int32(1),
        cutlass.Float32(1),
        heads,
        (dq, dkdv),
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=False),
        options="--enable-tvm-ffi --opt-level 2 --gpu-arch sm_100a",
        cache_key=cache_key,
        symbol="gleipnir_mxfp8_varlen_backward",
    )
