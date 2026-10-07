"""Experimental FP4 SwiGLU output with unit global and local E4M3 scales."""

# ruff: noqa: E501 -- Pinned generated CuTe source statements.

from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path

from gleipnir.serving_fp4_swiglu import UPSTREAM
from gleipnir.serving_fp4_swiglu_overhead import overhead_source


def native_output_source(original: str) -> str:
    """Emit FP4 register values directly; do not use whole-row dynamic scaling."""
    source = overhead_source(original)
    anchor = "                    # Generate amax\n"
    a = source.index(anchor)
    # Include the decorative comment just before this block.
    a = source.rfind("                    #\n", 0, a)
    b = source.index("                # Perform amax reduction after all subtiles", a)
    direct = """                    # Direct local-block NVFP4 output; BF16 activation boundary retained.
                    for vi in cutlass.range_constexpr(cute.size(tCompute)):
                        tCompute[vi] = tCompute[vi].to(cutlass.BFloat16).to(cutlass.Float32)
                    for gi in cutlass.range_constexpr(cute.size(tCompute) // 16):
                        maximum = cutlass.Float32(0.0)
                        for ei in cutlass.range_constexpr(16):
                            val = tCompute[gi * 16 + ei]
                            maximum = cute.arch.fmax(maximum, cute.arch.fmax(val, -val))
                        unrounded_sf = nvvm.inline_ptx(
                            "div.rn.f32 $0, $1, $2;", write_only_types=[cutlass.Float32],
                            read_only_args=[maximum, cutlass.Float32(6.0)],
                        )
                        sf_value = unrounded_sf.to(cutlass.Float8E4M3FN)
                        sf_float = sf_value.to(cutlass.Float32)
                        inv_sf = cutlass.Float32(1.0)
                        denom_sf = cutlass.Float32(1.0)
                        if sf_float > 0.0:
                            inv_sf = cute.arch.rcp_approx(sf_float)
                            denom_sf = sf_float
                        rr = coords[gi * 16][0]
                        col = coords[gi * 16][1]
                        if rr < cute.ceil_div(mC_mnl.shape[0], 128) * 128:
                            sf_offset = (((rr // 128 * 144 + col // 64) * 32 + rr % 32) * 4 + rr % 128 // 32) * 4 + col // 16 % 4
                            if rr < mC_mnl.shape[0]:
                                norm_const_tensor[3][sf_offset] = sf_value
                            else:
                                norm_const_tensor[3][sf_offset] = cutlass.Float32(0.0).to(cutlass.Float8E4M3FN)
                        if rr < mC_mnl.shape[0]:
                            for wi in cutlass.range_constexpr(2):
                                word = cutlass.Uint32(0)
                                for bi in cutlass.range_constexpr(4):
                                    pi = wi * 4 + bi
                                    low = nvvm.inline_ptx(
                                        "{ .reg .f32 q, r, nq; mul.rn.f32 q, $1, $3; neg.f32 nq, q; fma.rn.f32 r, nq, $2, $1; fma.rn.f32 $0, r, $3, q; }",
                                        write_only_types=[cutlass.Float32],
                                        read_only_args=[tCompute[gi * 16 + pi * 2], denom_sf, inv_sf],
                                    )
                                    high = nvvm.inline_ptx(
                                        "{ .reg .f32 q, r, nq; mul.rn.f32 q, $1, $3; neg.f32 nq, q; fma.rn.f32 r, nq, $2, $1; fma.rn.f32 $0, r, $3, q; }",
                                        write_only_types=[cutlass.Float32],
                                        read_only_args=[tCompute[gi * 16 + pi * 2 + 1], denom_sf, inv_sf],
                                    )
                                    packed = nvvm.inline_ptx(
                                        "{ .reg .b8 p; cvt.rn.satfinite.e2m1x2.f32 p, $2, $1; mov.b32 $0, {p, p, p, p}; }",
                                        write_only_types=[cutlass.Uint32], read_only_args=[low, high],
                                    )
                                    word = word | ((packed & cutlass.Uint32(255)) << (bi * 8))
                                norm_const_tensor[2][rr * 1152 + col // 8 + wi] = word

"""
    source = source[:a] + direct + source[b:]
    return source.replace(
        "import cutlass.cute.math as math",
        "from cutlass.experimental import primitives as nvvm\nimport cutlass.cute.math as math",
    )


def load_kernel(root: Path, archive: Path):
    """Archive the precise separately generated, licensed kernel source."""
    original = (root / UPSTREAM).read_text()
    generated = native_output_source(original)
    archive.mkdir(parents=True, exist_ok=True)
    path = archive / "nvfp4_swiglu_native_output.py"
    path.write_text(generated)
    spec = importlib.util.spec_from_file_location("gleipnir_native_swiglu_output", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Sm100BlockScaledPersistentDenseGemmKernel, {
        "upstream_sha256": hashlib.sha256(original.encode()).hexdigest(),
        "generated_sha256": hashlib.sha256(generated.encode()).hexdigest(),
    }


class NativeOutputSwiGlu:
    """Symbolic-M GEMM + SwiGLU + local-block FP4; no BF16 activation stores."""

    def __init__(self, kernel):
        self.kernel, self.plan = kernel, None
        self.compile_count = 0
        self.inverse = None

    def compile(self):
        import cutlass
        import cutlass.cute as cute
        import torch
        from cutlass.cute.runtime import make_fake_stream, make_fake_tensor

        m = cute.sym_int()

        def fake(dtype, shape, stride):
            return make_fake_tensor(dtype, shape, stride=stride, assumed_align=16)

        self.plan = cute.compile(
            self.kernel(16, (256, 192), (2, 1), True, 2),
            fake(cutlass.Float4E2M1FN, (m, 2560, 1), (2560, 1, m * 2560)),
            fake(cutlass.Float4E2M1FN, (18432, 2560, 1), (2560, 1, 18432 * 2560)),
            fake(cutlass.Float8E4M3FN, (cute.sym_int(),), (1,)),
            fake(cutlass.Float8E4M3FN, (18432 * 160,), (1,)),
            fake(cutlass.BFloat16, (m, 9216, 1), (9216, 1, m * 9216)),
            fake(cutlass.BFloat16, (1, 18432, 1), (18432, 1, 18432)),
            None,
            None,
            (
                fake(cutlass.Float32, (m,), (1,)),
                fake(cutlass.Float32, (1,), (1,)),
                fake(cutlass.Uint32, (cute.sym_int(),), (1,)),
                fake(cutlass.Float8E4M3FN, (cute.sym_int(),), (1,)),
            ),
            1.0,
            cutlass.utils.HardwareInfo().get_max_active_clusters(2),
            make_fake_stream(use_tvm_ffi_env_stream=False),
            options="--enable-tvm-ffi",
        )
        self.inverse = torch.ones(32768, device="cuda", dtype=torch.float32)
        self.compile_count += 1

    def __call__(self, a, b):
        import torch
        import triton
        from cuda.bindings import driver as cuda

        from gleipnir.cudnn_fp4_gemm import PackedNvfp4

        m = a.codes.shape[0]
        if not 1 <= m <= 32768 or b.codes.shape != (18432, 1280):
            raise ValueError("native SwiGLU output outside measured envelope")
        if self.plan is None:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("compile native FP4 output before graph capture")
            self.compile()
        codes = torch.empty((m, 4608), device=a.codes.device, dtype=torch.uint8)
        sf = torch.empty(
            (triton.cdiv(m, 128) * 128, 576),
            device=a.codes.device,
            dtype=torch.float8_e4m3fn,
        )
        # Descriptor only: the adapted epilogue never writes this BF16 storage.
        descriptor = torch.empty(
            (m, 9216, 1), device=a.codes.device, dtype=torch.bfloat16
        )
        unused = torch.empty((1, 18432, 1), device=a.codes.device, dtype=torch.bfloat16)
        self.plan(
            a.codes.unsqueeze(-1),
            b.codes.unsqueeze(-1),
            a.scales.reshape(-1),
            b.scales.reshape(-1),
            descriptor,
            unused,
            None,
            None,
            (
                a.inverse,
                b.inverse,
                codes.view(torch.uint32).reshape(-1),
                sf.reshape(-1),
            ),
            1.0,
            cuda.CUstream(torch.cuda.current_stream().cuda_stream),
        )
        return PackedNvfp4(codes.view(torch.float4_e2m1fn_x2), sf, self.inverse[:m])
