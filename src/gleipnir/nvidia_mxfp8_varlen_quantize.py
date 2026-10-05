# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Sequence-local packed MXFP8 quantizer derived from NVIDIA's fused producer.

Payloads stay THD. Each CTA reads its example's cumulative offsets, emits the
canonical per-example scale slab for backward and head-major packed tiles for
forward. Tail blocks never include values from the following example. Maximum
length and packed capacity are runtime values, absent from compiler cache keys.
"""

import hashlib
from functools import lru_cache
from pathlib import Path

import cutlass
import cutlass.cute as cute
import torch
from cuda.bindings import driver as cuda
from cudnn.frost.compiled_cache import compile_cached, positional_entry
from cudnn.gated_attention_block.kernels.quantize_mxfp8 import (
    COL_D_PER_LANE,
    COL_D_PER_UNIT,
    COL_TOKEN_BLOCKS,
    COL_TOKENS_PER_UNIT,
    DST_BYTES_PER_LANE,
    ELEMS_PER_LANE,
    LOADS_PER_LANE,
    SF_ATOM_BYTES,
    SF_BURST_BYTES,
    SF_TILE_ROWS,
    SRC_BYTES_PER_LANE,
    WARP,
    abs_max_tree,
    chunks_per_lane,
    e8m0_from_amax,
    e8m0_pair,
    f16x2_to_f32,
    fmax_f32,
    fp32_to_fp8_pack,
    fp32_to_fp8x2,
    lanes_per_row,
    ld_global,
    ld_global_v4,
    sf_atom_byte,
    sf_atom_offset,
    sf_tile_bytes,
    st_global_v4,
)
from cutlass.experimental import primitives as nvvm


def st_global_b16(addr, value):
    """16-bit global store of a ``Uint16`` (two packed e4m3 codes)."""
    nvvm.inline_ptx("st.global.b16 [$0], $1;", read_only_args=[addr, value])


@cute.kernel
def packed_quantize_kernel(
    # [T, H, D] bf16/f16, own token stride (slab slice or compact), head stride D
    mSrc: cute.Tensor,
    # [T, H, D] e4m3, own token stride (compact in the block), head stride D
    mDst: cute.Tensor,
    # [B*H*ceil(S/128)*4*D] uint8, F8_128x4 order per the module docstring
    mSf: cute.Tensor,
    cumulative: cute.Tensor,
    packed_sf: cute.Tensor,
    packed_tiles: cutlass.Int32,
    n_tiles: cutlass.Int32,  # ceil(S/128) == gridDim.y
    # B*H*n_tiles: the columnwise D-plane stride in atoms (unused rowwise)
    v_sf_groups: cutlass.Int32,
    h: cutlass.Constexpr[int],
    d: cutlass.Constexpr[int],
    axis_col: cutlass.Constexpr[bool],
    threads_per_cta: cutlass.Constexpr[int],
) -> None:
    tile_bytes = cutlass.const_expr(sf_tile_bytes(d))
    burst_lanes = cutlass.const_expr(tile_bytes // SF_BURST_BYTES)
    sSF = cutlass.Array(
        cutlass.Uint8, tile_bytes, alignment=16, space=cutlass.AddressSpace.smem
    )

    tidx = cutlass.Int32(cute.arch.thread_idx()[0])
    head = cutlass.Int32(cute.arch.block_idx()[0])
    global_tile = cutlass.Int32(cute.arch.block_idx()[1])
    b = cutlass.Int32(0)
    tile_prefix = cutlass.Int32(0)
    prefix = cutlass.Int32(0)
    found = cutlass.Int32(0)
    for candidate in cutlass.range(cumulative.shape[0] - 1):
        length = cumulative[candidate + 1] - cumulative[candidate]
        count = (length + 127) // 128
        if global_tile >= prefix and global_tile < prefix + count:
            b = candidate
            tile_prefix = prefix
            found = cutlass.Int32(1)
        prefix += count
    if found == 1:
        bh = b * cutlass.Int32(h) + head
        s_tile = global_tile - tile_prefix
        s0 = s_tile * cutlass.Int32(SF_TILE_ROWS)
        tok0 = cumulative[b]
        seq_len = cumulative[b + 1] - tok0
        last = seq_len - cutlass.Int32(
            1
        )  # tail rows clamp their LOADS here (never past the tensor)
        src_tok_stride = cutlass.Int64(mSrc.stride[0]) * cutlass.Int64(2)
        dst_tok_stride = cutlass.Int64(mDst.stride[0])
        src_base = mSrc.iterator.toint() + head.to(cutlass.Int64) * cutlass.Int64(
            mSrc.stride[1]
        ) * cutlass.Int64(2)
        dst_base = mDst.iterator.toint() + head.to(cutlass.Int64) * cutlass.Int64(
            mDst.stride[1]
        )

        if cutlass.const_expr(not axis_col):
            # ---- ROWWISE (Q/K): 32-element blocks along D
            # --------------------------------
            lanes = cutlass.const_expr(lanes_per_row(d))
            chunks = cutlass.const_expr(chunks_per_lane(d))
            rows_per_pass = cutlass.const_expr(threads_per_cta // lanes)
            passes = cutlass.const_expr(SF_TILE_ROWS // rows_per_pass)
            lane = tidx % cutlass.Int32(lanes)
            grp = tidx // cutlass.Int32(lanes)
            even = (lane & cutlass.Int32(1)) == cutlass.Int32(0)
            src_lane_off = lane.to(cutlass.Int64) * cutlass.Int64(SRC_BYTES_PER_LANE)
            dst_lane_off = lane.to(cutlass.Int64) * cutlass.Int64(DST_BYTES_PER_LANE)
            for p in cutlass.range_constexpr(passes):
                r = (
                    cutlass.Int32(p * rows_per_pass) + grp
                )  # row within the 128-row tile
                s = s0 + r
                valid = s < seq_len
                s_ld = s if valid else last
                tok = tok0 + s_ld
                src_row = src_base + tok.to(cutlass.Int64) * src_tok_stride
                dst_row = dst_base + tok.to(cutlass.Int64) * dst_tok_stride
                for c in cutlass.range_constexpr(chunks):
                    base = (
                        src_row
                        + cutlass.Int64((c * lanes) * SRC_BYTES_PER_LANE)
                        + src_lane_off
                    )
                    vals = []
                    for j in cutlass.range_constexpr(LOADS_PER_LANE):
                        for w in ld_global_v4(
                            base + cutlass.Int64(j * 16), cutlass.Int32
                        ):
                            lo, hi = f16x2_to_f32(w, dtype=mSrc.element_type)
                            vals.append(lo)
                            vals.append(hi)
                    # The lane pair (lane, lane^1) shares one 32-element block: bfly(1)
                    # completes the block amax.
                    amax = abs_max_tree(vals)
                    partner = cutlass.Float32(
                        nvvm.shfl_sync(
                            0xFFFFFFFF, amax, cutlass.Int32(1), 31, kind=nvvm.Shfl.BFLY
                        )
                    )
                    amax = fmax_f32(amax, partner)
                    amax = (
                        amax if valid else cutlass.Float32(0.0)
                    )  # tail row: SF 0x00, no data store
                    rcp, sf_byte = e8m0_from_amax(amax)
                    if valid:
                        scaled = []
                        for i in cutlass.range_constexpr(ELEMS_PER_LANE):
                            scaled.append(vals[i] * rcp)
                        packed = fp32_to_fp8_pack(scaled, dtype=cutlass.Float8E4M3FN)
                        st_global_v4(
                            dst_row
                            + cutlass.Int64((c * lanes) * DST_BYTES_PER_LANE)
                            + dst_lane_off,
                            [packed[0], packed[1], packed[2], packed[3]],
                            cutlass.Int32,
                        )
                    # SF SMEM byte for block column c_idx = d//32 of row r: (c//4)*512 +
                    # (r%32)*16 + (r//32)*4 + c%4 (sf_layout).
                    c_idx = (cutlass.Int32(c * lanes) + lane) // cutlass.Int32(2)
                    sf_off = sf_atom_offset(r, c_idx)
                    if even:
                        sSF.store(sf_byte.to(cutlass.Uint8), sf_off)
        else:
            # ---- COLUMNWISE (V): 32-token blocks along S
            # -----------------------------------
            warps = cutlass.const_expr(threads_per_cta // WARP)
            slices = cutlass.const_expr(d // COL_D_PER_UNIT)
            units_per_warp = cutlass.const_expr(COL_TOKEN_BLOCKS * slices // warps)
            lane = tidx % cutlass.Int32(WARP)
            warp = tidx // cutlass.Int32(WARP)
            for i in cutlass.range_constexpr(units_per_warp):
                u = warp * cutlass.Int32(units_per_warp) + cutlass.Int32(i)
                tb = u // cutlass.Int32(slices)  # 32-token block within the tile
                ds = u % cutlass.Int32(slices)  # 64-d slice
                d0 = ds * cutlass.Int32(COL_D_PER_UNIT) + lane * cutlass.Int32(
                    COL_D_PER_LANE
                )
                s_blk = s0 + tb * cutlass.Int32(COL_TOKENS_PER_UNIT)
                n_valid = (
                    seq_len - s_blk
                )  # tokens t < n_valid are in range (may be <= 0 or >= 32)
                # Token addresses walk INCREMENTALLY (one 64-bit add per token, not a
                # 64-bit
                # multiply each);
                # a tail token's load is redirected to the batch's last row and its word
                # zeroed.
                src_addr = (
                    src_base
                    + (tok0 + s_blk).to(cutlass.Int64) * src_tok_stride
                    + d0.to(cutlass.Int64) * cutlass.Int64(2)
                )
                src_last = (
                    src_base
                    + (tok0 + last).to(cutlass.Int64) * src_tok_stride
                    + d0.to(cutlass.Int64) * cutlass.Int64(2)
                )
                dst_addr = (
                    dst_base
                    + (tok0 + s_blk).to(cutlass.Int64) * dst_tok_stride
                    + d0.to(cutlass.Int64)
                )
                words = []
                valids = []
                dsts = []
                for t in cutlass.range_constexpr(COL_TOKENS_PER_UNIT):
                    valid = cutlass.Int32(t) < n_valid  # warp-uniform
                    w = ld_global(src_addr if valid else src_last, cutlass.Int32)
                    words.append(
                        w if valid else cutlass.Int32(0)
                    )  # tail token: contributes 0 to the block amax
                    valids.append(valid)
                    dsts.append(dst_addr)
                    src_addr = src_addr + src_tok_stride
                    dst_addr = dst_addr + dst_tok_stride
                los = []
                his = []
                for t in cutlass.range_constexpr(COL_TOKENS_PER_UNIT):
                    lo, hi = f16x2_to_f32(words[t], dtype=mSrc.element_type)
                    los.append(lo)
                    his.append(hi)
                rcp0, rcp1, sf_pair = e8m0_pair(abs_max_tree(los), abs_max_tree(his))
                for t in cutlass.range_constexpr(COL_TOKENS_PER_UNIT):
                    if valids[t]:
                        st_global_b16(
                            dsts[t], fp32_to_fp8x2(los[t] * rcp0, his[t] * rcp1)
                        )
                # SF SMEM bytes: plane p = d//128 at p*512 + ((d%128)%32)*16 +
                # ((d%128)//32)*4 + tb (sf_layout); d0+1 sits 16 B after d0.
                plane = d0 // cutlass.Int32(SF_TILE_ROWS)
                dm = d0 % cutlass.Int32(SF_TILE_ROWS)
                sf_off = sf_atom_byte(dm, tb, base=plane * cutlass.Int32(SF_ATOM_BYTES))
                sSF.store((sf_pair & cutlass.Int32(0xFF)).to(cutlass.Uint8), sf_off)
                sSF.store(
                    ((sf_pair >> 8) & cutlass.Int32(0xFF)).to(cutlass.Uint8),
                    sf_off + cutlass.Int32(16),
                )

        # ---- SF burst: the whole tile leaves SMEM as 16-B lanes
        # ---------------------------
        nvvm.barrier_cta_sync()
        if tidx < cutlass.Int32(burst_lanes):
            smem_off = tidx * cutlass.Int32(SF_BURST_BYTES)
            words4 = sSF.load(
                smem_off, vector_size=SF_BURST_BYTES, alignment=16
            ).bitcast(cutlass.Int32)
            tile_idx = (bh * n_tiles + s_tile).to(cutlass.Int64)
            sf_base = mSf.iterator.toint()
            if cutlass.const_expr(not axis_col):
                # Q/K: the tile is 4*D contiguous bytes.
                gaddr = (
                    sf_base
                    + tile_idx * cutlass.Int64(tile_bytes)
                    + smem_off.to(cutlass.Int64)
                )
            else:
                # V: plane p (512 B, lanes p*32..p*32+31) lands p * v_sf_groups atoms
                # away.
                burst_plane = (
                    tidx // cutlass.Int32(SF_ATOM_BYTES // SF_BURST_BYTES)
                ).to(cutlass.Int64)
                burst_within = (
                    (tidx % cutlass.Int32(SF_ATOM_BYTES // SF_BURST_BYTES))
                    * cutlass.Int32(SF_BURST_BYTES)
                ).to(cutlass.Int64)
                gaddr = (
                    sf_base
                    + burst_plane
                    * (v_sf_groups.to(cutlass.Int64) * cutlass.Int64(SF_ATOM_BYTES))
                    + tile_idx * cutlass.Int64(SF_ATOM_BYTES)
                    + burst_within
                )
            st_global_v4(
                gaddr, [words4[0], words4[1], words4[2], words4[3]], cutlass.Int32
            )
            if s0 < seq_len:
                packed_tile = (
                    head.to(cutlass.Int64) * packed_tiles + tile_prefix + s_tile
                )
                packed_addr = (
                    packed_sf.iterator.toint() + packed_tile * tile_bytes + smem_off
                )
                st_global_v4(
                    packed_addr,
                    [words4[0], words4[1], words4[2], words4[3]],
                    cutlass.Int32,
                )


@cute.jit
def packed_quantize_launch(
    src: cute.Tensor,
    dst: cute.Tensor,
    sf: cute.Tensor,
    cumulative: cute.Tensor,
    packed_sf: cute.Tensor,
    packed_tiles: cutlass.Int32,
    n_tiles: cutlass.Int32,
    groups: cutlass.Int32,
    n_bh: cutlass.Int32,
    h: cutlass.Constexpr,
    columnwise: cutlass.Constexpr,
    stream: cuda.CUstream,
):
    packed_quantize_kernel(
        src,
        dst,
        sf,
        cumulative,
        packed_sf,
        packed_tiles,
        n_tiles,
        groups,
        h,
        256,
        columnwise,
        256,
    ).launch(grid=(h, packed_tiles, 1), block=(256, 1, 1), stream=stream)


@lru_cache(maxsize=8)
def recipe(heads: int, columnwise: bool):
    tok = cute.sym_int()

    def fake(dtype, shape, stride):
        return cute.runtime.make_fake_tensor(
            dtype=dtype, shape=shape, stride=stride, assumed_align=16
        )

    src = fake(cutlass.BFloat16, (tok, heads, 256), (heads * 256, 256, 1))
    dst = fake(cutlass.Float8E4M3FN, (tok, heads, 256), (heads * 256, 256, 1))
    sf = fake(cutlass.Uint8, (cute.sym_int(),), (1,))
    cu = fake(cutlass.Int32, (cute.sym_int(),), (1,))
    packed = fake(cutlass.Uint8, (cute.sym_int(),), (1,))
    digest = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    entry = compile_cached(
        packed_quantize_launch,
        src,
        dst,
        sf,
        cu,
        packed,
        *(cutlass.Int32(0) for _ in range(4)),
        heads,
        columnwise,
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=False),
        options="--enable-tvm-ffi --gpu-arch sm_100a",
        cache_key=f"gleipnir-packed-quantize:{digest}:{heads}:{columnwise}",
        symbol="gleipnir_mxfp8_varlen_quantize",
    )
    fn = positional_entry(entry)
    if fn is None:
        raise RuntimeError("packed quantizer requires TVM FFI entry")
    return entry, fn


def quantize(
    source: torch.Tensor, cumulative: torch.Tensor, maximum: int, columnwise: bool
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Explicitly allocate payload and both scale forms; no host length reads."""
    source = source.contiguous()
    total, heads, _ = source.shape
    batch = cumulative.numel() - 1
    tiles = (maximum + 127) // 128
    capacity = (total + 127) // 128 + batch
    payload = torch.empty_like(source, dtype=torch.float8_e4m3fn)
    sf = torch.empty(
        batch * heads * tiles * 1024, device=source.device, dtype=torch.uint8
    )
    packed = torch.empty(
        heads * capacity * 1024, device=source.device, dtype=torch.uint8
    )
    _, fn = recipe(heads, columnwise)
    fn(
        source,
        payload,
        sf,
        cumulative,
        packed,
        capacity,
        tiles,
        batch * heads * tiles,
        batch * heads,
        torch.cuda.current_stream().cuda_stream,
    )
    return payload, sf, packed
