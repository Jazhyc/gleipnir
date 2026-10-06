"""Bounded inference adaptation of NVIDIA's pinned NVFP4 SwiGLU template."""

# ruff: noqa: E501 -- Exact upstream anchors and generated CuTe statements.

from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path

UPSTREAM = Path(
    ".cache/kernels/nvidia_mxfp8/frontend/cudnn/gemm/cutedsl/dense/swiglu/"
    "dense_blockscaled_gemm_persistent_swiglu_interleaved_quant.py"
)


def interleaved_rows(n: int) -> list[int]:
    """Map natural gate/up rows to 32-wide up/gate pairs, without quantization."""
    if n <= 0 or n % 64:
        raise ValueError("gate/up width must be positive and divisible by 64")
    half = n // 2
    return [
        row
        for start in range(0, half, 32)
        for offset in (half, 0)
        for row in range(start + offset, start + offset + 32)
    ]


def adapt_source(source: str) -> str:
    """Preserve licensing; remove AB12 writes and scale both inputs before SiLU.

    Generate a separate module instead of mutating shared NVIDIA kernel files.
    Identity-coordinate partitioning matches the output register layout, so
    each lane loads the appropriate row scale, including partial M tiles.
    """

    def replace(old: str, new: str) -> None:
        nonlocal source
        if source.count(old) != 1:
            raise ValueError(f"pinned SwiGLU source drift at {old[:70]!r}")
        source = source.replace(old, new)

    replace(
        "        tCgC = thr_mma.partition_C(gC_mnl)",
        """        tCgC = thr_mma.partition_C(gC_mnl)
        coordC = cute.make_identity_tensor(mC_mnl.shape)
        gCoordC = cute.local_tile(coordC, cute.slice_(self.mma_tiler_c, (None, None, 0)), (None, None, None))
        tCoordC = thr_mma.partition_C(gCoordC)""",
    )
    replace(
        "            tTR_rC = cute.make_rmem_tensor(tTR_rAcc_up.shape, self.c_dtype)",
        """            coord_epi = cute.flat_divide(tCoordC[((None, None), 0, 0, None, None, None)], epi_tile)
            coord_regs = tiled_copy_t2r.get_slice(epi_tidx).partition_D(coord_epi)
            tTR_rC = cute.make_rmem_tensor(tTR_rAcc_up.shape, self.c_dtype)""",
    )
    replace(
        "                    acc_vec_up_ = acc_vec_up_ * alpha\n                    acc_vec_gate_ = acc_vec_gate_ * alpha",
        """                    coords = coord_regs[(None, None, None, 0, sfc_subtile_idx, *mma_tile_coord_mnl)]
                    coords = tiled_copy_r2s.retile(coords)
                    for vi in cutlass.range_constexpr(cute.size(acc_vec_up)):
                        row = coords[vi][0]
                        scale = cutlass.Float32(0.0)
                        if row < mC_mnl.shape[0]:
                            scale = norm_const_tensor[row]
                        acc_vec_up[vi] = (acc_vec_up[vi].to(cutlass.BFloat16).to(cutlass.Float32) * scale).to(cutlass.BFloat16).to(cutlass.Float32)
                        acc_vec_gate[vi] = (acc_vec_gate[vi].to(cutlass.BFloat16).to(cutlass.Float32) * scale).to(cutlass.BFloat16).to(cutlass.Float32)
                    acc_vec_up_ = acc_vec_up.load()
                    acc_vec_gate_ = acc_vec_gate.load()""",
    )
    begin = source.index("                    # Store AB12 to shared memory for bprop")
    end = source.index("                    # SwiGelu", begin)
    source = source[:begin] + source[end:]

    # Inference also releases backward-only shared storage and pipeline state.
    def remove_between(begin: str, end: str) -> None:
        nonlocal source
        if source.count(begin) != 1:
            raise ValueError("pinned SwiGLU backward-storage source drift")
        a = source.index(begin)
        b = source.index(end, a)
        source = source[:a] + source[b:]

    remove_between(
        "            sAB12: cute.struct.Align[",
        "            # (MMA, MMA_M, MMA_K, STAGE)",
    )
    remove_between(
        "        sAB12 = storage.sAB12.get_tensor(",
        "        # Shared memory for amax reduction",
    )
    remove_between(
        "            tTR_rAB12 = cute.make_rmem_tensor(",
        "            #\n            # Persistent tile scheduling loop",
    )
    remove_between(
        "            ab12_producer_group = pipeline.CooperativeGroup(",
        "            # norm_const when used in sfc",
    )
    remove_between(
        "                bSG_gAB12 = bSG_gAB12_mnl[",
        "                # Set tensor memory buffer for current tile",
    )
    replace(
        "                bSG_gAB12 = cute.group_modes(bSG_gAB12, 1, cute.rank(bSG_gAB12))\n",
        "",
    )
    replace("            ab12_pipeline.producer_tail()\n", "")
    replace(
        "epi_bytes = c_bytes_per_stage * num_c_stage + ab12_bytes_per_stage * num_ab12_stage + amax_bytes",
        "epi_bytes = c_bytes_per_stage * num_c_stage + amax_bytes",
    )
    return source


def load_kernel(root: Path, archive: Path):
    """Archive exact generated source and import its isolated module."""
    original = (root / UPSTREAM).read_text()
    generated = adapt_source(original)
    archive.mkdir(parents=True, exist_ok=True)
    path = archive / "nvfp4_swiglu_inference.py"
    path.write_text(generated)
    spec = importlib.util.spec_from_file_location("gleipnir_nvfp4_swiglu_trial", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Sm100BlockScaledPersistentDenseGemmKernel, {
        "upstream_sha256": hashlib.sha256(original.encode()).hexdigest(),
        "generated_sha256": hashlib.sha256(generated.encode()).hexdigest(),
    }


def prepare_weight(weight):
    """Permute packed data and swizzled SF bytes once, keeping the inverse."""
    import torch

    from gleipnir.cudnn_fp4_gemm import PackedNvfp4

    n, packed_k = weight.codes.shape
    k = packed_k * 2
    rows = torch.tensor(interleaved_rows(n), device=weight.codes.device)
    r = torch.arange(n, device=rows.device)[:, None]
    c = torch.arange(k // 16, device=rows.device)[None, :]

    def offsets(rr):
        return (
            ((rr // 128 * (k // 64) + c // 4) * 32 + rr % 32) * 4 + rr % 128 // 32
        ) * 4 + c % 4

    scales = torch.empty_like(weight.scales).view(torch.uint8).reshape(-1)
    scales[offsets(r)] = weight.scales.view(torch.uint8).reshape(-1)[
        offsets(rows[:, None])
    ]
    codes = weight.codes.view(torch.uint8)[rows].contiguous().view(weight.codes.dtype)
    return PackedNvfp4(codes, scales.view(weight.scales.dtype), weight.inverse)


class FusedSwiGlu:
    """Cache exact-M native plans; use original FP4 operands and FP32 descale."""

    def __init__(
        self, kernel, reference, tile=(128, 256), cluster=(1, 1), *, vector=False
    ):
        self.kernel, self.reference = kernel, reference
        self.tile, self.cluster = tile, cluster
        self.vector = vector
        self.plans = {}

    def __call__(self, a, b):
        import cutlass.cute as cute
        import torch
        import triton
        from cuda.bindings import driver as cuda
        from cudnn.gemm.cutedsl.dense.swiglu.api import GemmSwigluSm100
        from cutlass.cute.runtime import make_fake_stream

        from gleipnir.cudnn_fp4_epilogue import _row_scale
        from gleipnir.cudnn_fp4_gemm import PackedNvfp4

        logical_m, n = a.codes.shape[0], b.codes.shape[0]
        m = triton.cdiv(logical_m, self.tile[0]) * self.tile[0]
        if m != logical_m:
            codes = torch.zeros(
                m, a.codes.shape[1], device=a.codes.device, dtype=torch.uint8
            )
            codes[:logical_m].copy_(a.codes.view(torch.uint8))
            inverse = torch.ones(m, device=a.codes.device, dtype=torch.float32)
            inverse[:logical_m].copy_(a.inverse)
            sf = torch.zeros(
                triton.cdiv(m, 128) * 128,
                a.scales.shape[1] if a.scales.ndim == 2 else a.codes.shape[1] // 8,
                device=a.codes.device,
                dtype=a.scales.dtype,
            )
            sf.reshape(-1)[: a.scales.numel()].copy_(a.scales.reshape(-1))
            a = PackedNvfp4(codes.view(a.codes.dtype), sf, inverse)
        scale = torch.empty(m, device=a.codes.device, dtype=torch.float32)
        _row_scale[(triton.cdiv(m, 1024),)](a.inverse, b.inverse, scale, m, 1024)
        out = torch.empty(m, n // 2, device=a.codes.device, dtype=torch.bfloat16)
        # The unused descriptor preserves upstream shared-memory planning, but
        # the adaptation never stores the training-only AB12 tensor.
        unused = torch.empty_strided(
            (1, n, 1), (n, 1, n), device=a.codes.device, dtype=torch.bfloat16
        )
        aa, bb = a.codes.unsqueeze(-1), b.codes.unsqueeze(-1)
        k = a.codes.shape[1] * 2
        sa = a.scales.view(1, triton.cdiv(m, 128), k // 64, 32, 4, 4)
        sb = b.scales.view(1, triton.cdiv(n, 128), k // 64, 32, 4, 4)
        cc = out.unsqueeze(-1)
        if m not in self.plans:
            api = GemmSwigluSm100(
                aa,
                bb,
                unused,
                cc,
                sample_sfa=sa,
                sample_sfb=sb,
                mma_tiler_mn=self.tile,
                cluster_shape_mn=self.cluster,
            )
            api.norm_const_desc = api._make_tensor_desc(
                scale, name="row_scale", canonical=True
            )
            kernel = self.kernel(16, self.tile, self.cluster, self.vector, 2)
            fake = api._make_fake_cute_tensor_from_desc
            hw = __import__("cutlass").utils.HardwareInfo()
            self.plans[m] = cute.compile(
                kernel,
                fake(api.a_desc, assumed_align=16),
                fake(api.b_desc, assumed_align=16),
                fake(api.sfa_desc, assumed_align=16),
                fake(api.sfb_desc, assumed_align=16),
                fake(api.c_desc, assumed_align=16),
                fake(api.ab12_desc, assumed_align=8),
                None,
                None,
                fake(api.norm_const_desc, assumed_align=16),
                1.0,
                hw.get_max_active_clusters(self.cluster[0] * self.cluster[1]),
                make_fake_stream(use_tvm_ffi_env_stream=False),
                options="--enable-tvm-ffi",
            )
        self.plans[m](
            aa,
            bb,
            sa,
            sb,
            cc,
            unused,
            None,
            None,
            scale,
            1.0,
            cuda.CUstream(torch.cuda.current_stream().cuda_stream),
        )
        return out[:logical_m]
