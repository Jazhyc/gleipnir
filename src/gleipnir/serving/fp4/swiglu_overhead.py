"""Isolated symbolic-M SwiGLU trial with inverse scaling in the epilogue."""

# ruff: noqa: E501 -- Exact pinned CuTe source anchors.

from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path

from gleipnir.serving_fp4_swiglu import UPSTREAM, adapt_source


def overhead_source(original: str) -> str:
    """Retain the old arithmetic, with inverse tensors replacing scale storage."""
    source = adapt_source(original)
    old_scale = """                    for vi in cutlass.range_constexpr(cute.size(acc_vec_up)):
                        row = coords[vi][0]
                        scale = cutlass.Float32(0.0)
                        if row < mC_mnl.shape[0]:
                            scale = norm_const_tensor[row]
"""
    new_scale = """                    row = coords[0][0]
                    scale = cutlass.Float32(0.0)
                    if row < mC_mnl.shape[0]:
                        scale = cute.arch.rcp_approx(norm_const_tensor[0][row] * norm_const_tensor[1][0])
                    for vi in cutlass.range_constexpr(cute.size(acc_vec_up)):
"""
    replacements = {
        "if cutlass.const_expr(self.cta_tile_shape_mnk_c[1] == 192):": "if cutlass.const_expr(self.cta_tile_shape_mnk[1] == 192):",
        old_scale: new_scale,
    }
    for old, new in replacements.items():
        if source.count(old) != 1:
            raise ValueError(f"SwiGLU overhead source drift: {old}")
        source = source.replace(old, new)
    old = "norm_const_tensor: Optional[cute.Tensor]"
    if source.count(old) != 2:
        raise ValueError("SwiGLU inverse argument source drift")
    return source.replace(old, "norm_const_tensor: tuple")


def load_kernel(root: Path, archive: Path):
    """Generate a licensed, separately archived source without vendor mutation."""
    original = (root / UPSTREAM).read_text()
    source = overhead_source(original)
    archive.mkdir(parents=True, exist_ok=True)
    path = archive / "nvfp4_swiglu_overhead.py"
    path.write_text(source)
    spec = importlib.util.spec_from_file_location("gleipnir_swiglu_overhead", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Sm100BlockScaledPersistentDenseGemmKernel, {
        "upstream_sha256": hashlib.sha256(original.encode()).hexdigest(),
        "generated_sha256": hashlib.sha256(source.encode()).hexdigest(),
    }


class OverheadSwiGlu:
    """One symbolic-M native plan; no precomputed row-scale launch or buffer."""

    def __init__(self, kernel, *, pad: bool = False):
        self.kernel, self.pad = kernel, pad
        self.plan = None
        self.plans = {}
        self.compile_count = 0

    def compile(self):
        import cutlass
        import cutlass.cute as cute
        from cutlass.cute.runtime import make_fake_stream, make_fake_tensor

        m = cute.sym_int(divisibility=256 if self.pad else 1)

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
            (fake(cutlass.Float32, (m,), (1,)), fake(cutlass.Float32, (1,), (1,))),
            1.0,
            cutlass.utils.HardwareInfo().get_max_active_clusters(2),
            make_fake_stream(use_tvm_ffi_env_stream=False),
            options="--enable-tvm-ffi",
        )
        self.compile_count += 1

    def __call__(self, a, b):
        import torch
        from cuda.bindings import driver as cuda

        logical_m = a.codes.shape[0]
        if not 1 <= logical_m <= 32768 or b.codes.shape != (18432, 1280):
            raise ValueError("SwiGLU overhead outside measured envelope")
        if self.pad and logical_m % 256:
            a = padded_operand(a)
        m = a.codes.shape[0]
        if self.plan is None:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("compile symbolic SwiGLU before graph capture")
            self.compile()
        out = torch.empty((m, 9216), dtype=torch.bfloat16, device=a.codes.device)
        unused = torch.empty((1, 18432, 1), dtype=torch.bfloat16, device=a.codes.device)
        self.plan(
            a.codes.unsqueeze(-1),
            b.codes.unsqueeze(-1),
            a.scales.reshape(-1),
            b.scales.reshape(-1),
            out.unsqueeze(-1),
            unused,
            None,
            None,
            (a.inverse, b.inverse),
            1.0,
            cuda.CUstream(torch.cuda.current_stream().cuda_stream),
        )
        return out[:logical_m]


def padded_operand(a):
    """Initialize and copy all three packed buffers in one launch."""
    import torch
    import triton

    from gleipnir.cudnn_fp4_gemm import PackedNvfp4
    from gleipnir.serving_fp4_swiglu_padding import copy_padded

    rows = a.codes.shape[0]
    m = triton.cdiv(rows, 256) * 256
    codes = torch.empty((m, 1280), dtype=torch.uint8, device=a.codes.device)
    sf = torch.empty((m, 160), dtype=a.scales.dtype, device=a.codes.device)
    inv = torch.empty(m, dtype=torch.float32, device=a.codes.device)
    copy_padded(a, codes, sf, inv)
    return PackedNvfp4(codes.view(a.codes.dtype), sf, inv)
