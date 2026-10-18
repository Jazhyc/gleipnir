"""Config-derived native FP4 SwiGLU output for Qwen3.5-family MLPs.

The 4B kernel (``gleipnir.serving.fp4.swiglu_native_output``) is hash-bound,
compiled for gate/up 18432x2560 with an MMA 256x192 tile, and stores codes
and scale factors with the 4B output width baked in (``rr * 1152``,
``rr // 128 * 144``). Its stores guard rows, not columns, so a tile that does
not divide the gate/up width would write out of bounds; 34,816 (27B) is not
a multiple of 192. This module reuses the 4B source transformation and
weight interleaving by import, substitutes the served output width, and
compiles a tile that divides the gate/up width exactly (checked).

Arithmetic is the 4B native output's: BF16 activation boundary, then FP4 E2M1
with one E4M3 scale per 16 values and unit global inverse (no whole-row
scaling), so it changes activation quantization relative to the fused
producer and is reported, not gated.
"""

from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
from types import MethodType
from typing import Any

import torch

from gleipnir.kernels.fp4.cudnn_fp4_gemm import PackedNvfp4
from gleipnir.serving.fp4.integration import packed_linear
from gleipnir.serving.fp4.swiglu import UPSTREAM, prepare_weight
from gleipnir.serving.fp4.swiglu_native_output import native_output_source

TILES = ((256, 256), (256, 128))
ENVELOPE = 32768
_STATE: dict[str, Any] = {"plan": None, "band": None, "audit": None}


def native_source(original: str, mlp: int) -> str:
    """The 4B native-output source with the served SwiGLU output width."""
    if mlp % 64:
        raise ValueError("SwiGLU output width must be a multiple of 64")
    source = native_output_source(original)
    for old, new in (
        ("rr // 128 * 144 + col // 64", f"rr // 128 * {mlp // 64} + col // 64"),
        ("rr * 1152 + col // 8 + wi", f"rr * {mlp // 8} + col // 8 + wi"),
    ):
        if source.count(old) != 1:
            raise ValueError(f"native SwiGLU store anchor drift: {old}")
        source = source.replace(old, new)
    return source


def load_kernel(root: Path, archive: Path, mlp: int):
    """Generate, archive and import the served-width kernel source."""
    original = (root / UPSTREAM).read_text()
    generated = native_source(original, mlp)
    archive.mkdir(parents=True, exist_ok=True)
    path = archive / f"nvfp4_swiglu_native_output_{mlp}.py"
    path.write_text(generated)
    name = f"gleipnir_geometry_swiglu_{mlp}"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Sm100BlockScaledPersistentDenseGemmKernel, {
        "upstream_sha256": hashlib.sha256(original.encode()).hexdigest(),
        "generated_sha256": hashlib.sha256(generated.encode()).hexdigest(),
    }


class NativeSwiGlu:
    """Symbolic-M GEMM + SwiGLU + local-block FP4 at a served MLP geometry."""

    def __init__(self, kernel, hidden: int, mlp: int, tile=(256, 256), cluster=(2, 1)):
        if tuple(tile) not in TILES or (2 * mlp) % tile[1]:
            raise ValueError(f"tile {tile} must divide the gate/up width {2 * mlp}")
        if hidden % 64:
            raise ValueError("hidden width must be a multiple of 64")
        self.kernel, self.hidden, self.mlp = kernel, hidden, mlp
        self.tile, self.cluster = tuple(tile), tuple(cluster)
        self.plan, self.compile_count = None, 0
        self.inverse = None

    def compile(self):
        import cutlass
        import cutlass.cute as cute
        from cutlass.cute.runtime import make_fake_stream, make_fake_tensor

        k, n, out = self.hidden, 2 * self.mlp, self.mlp
        m = cute.sym_int()

        def fake(dtype, shape, stride):
            return make_fake_tensor(dtype, shape, stride=stride, assumed_align=16)

        self.plan = cute.compile(
            self.kernel(16, self.tile, self.cluster, True, 2),
            fake(cutlass.Float4E2M1FN, (m, k, 1), (k, 1, m * k)),
            fake(cutlass.Float4E2M1FN, (n, k, 1), (k, 1, n * k)),
            fake(cutlass.Float8E4M3FN, (cute.sym_int(),), (1,)),
            fake(cutlass.Float8E4M3FN, (n * (k // 16),), (1,)),
            fake(cutlass.BFloat16, (m, out, 1), (out, 1, m * out)),
            fake(cutlass.BFloat16, (1, n, 1), (n, 1, n)),
            None,
            None,
            (
                fake(cutlass.Float32, (m,), (1,)),
                fake(cutlass.Float32, (1,), (1,)),
                fake(cutlass.Uint32, (cute.sym_int(),), (1,)),
                fake(cutlass.Float8E4M3FN, (cute.sym_int(),), (1,)),
            ),
            1.0,
            cutlass.utils.HardwareInfo().get_max_active_clusters(
                self.cluster[0] * self.cluster[1]
            ),
            make_fake_stream(use_tvm_ffi_env_stream=False),
            options="--enable-tvm-ffi",
        )
        self.inverse = torch.ones(ENVELOPE, device="cuda", dtype=torch.float32)
        self.compile_count += 1

    def __call__(self, a: PackedNvfp4, b: PackedNvfp4) -> PackedNvfp4:
        import triton
        from cuda.bindings import driver as cuda

        m = a.codes.shape[0]
        if not 1 <= m <= ENVELOPE or tuple(b.codes.shape) != (
            2 * self.mlp,
            self.hidden // 2,
        ):
            raise ValueError("native SwiGLU output outside the compiled geometry")
        if self.plan is None:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("compile native FP4 output before graph capture")
            self.compile()
        device = a.codes.device
        codes = torch.empty((m, self.mlp // 2), device=device, dtype=torch.uint8)
        sf = torch.empty(
            (triton.cdiv(m, 128) * 128, self.mlp // 16),
            device=device,
            dtype=torch.float8_e4m3fn,
        )
        # Descriptor only: the adapted epilogue never writes this BF16 storage.
        descriptor = torch.empty((m, self.mlp, 1), device=device, dtype=torch.bfloat16)
        unused = torch.empty((1, 2 * self.mlp, 1), device=device, dtype=torch.bfloat16)
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


@torch.library.custom_op("gleipnir::geometry_swiglu_native_output", mutates_args=())
def native_producer(
    codes: torch.Tensor,
    sf: torch.Tensor,
    inverse: torch.Tensor,
    weight: torch.Tensor,
    weight_sf: torch.Tensor,
    weight_inverse: torch.Tensor,
    original_weight: torch.Tensor,
    original_sf: torch.Tensor,
    original_inverse: torch.Tensor,
    layer: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    from gleipnir.serving.fp4_geometry_producers import silu_producer

    plan, (low, high) = _STATE["plan"], _STATE["band"]
    if plan is None or plan.plan is None:
        raise RuntimeError("native SwiGLU called before its plan was compiled")
    m = codes.shape[0]
    if not low <= m <= high:
        gateup = packed_linear(
            codes, sf, inverse, original_weight, original_sf, original_inverse
        )
        result = silu_producer(gateup)
        backend = "fused_producer"
    else:
        a = plan(
            PackedNvfp4(codes.view(torch.float4_e2m1fn_x2), sf, inverse),
            PackedNvfp4(weight, weight_sf, weight_inverse),
        )
        result = (a.codes.view(torch.uint8), a.scales, a.inverse)
        backend = "native"
    if _STATE["audit"] is not None:
        _STATE["audit"](layer, m, backend)
    return result


@native_producer.register_fake
def _fake_native(
    codes,
    sf,
    inverse,
    weight,
    weight_sf,
    weight_inverse,
    original_weight,
    original_sf,
    original_inverse,
    layer,
):
    m, mlp = codes.shape[0], original_weight.shape[0] // 2
    return (
        codes.new_empty((m, mlp // 2), dtype=torch.uint8),
        codes.new_empty(
            (((m + 127) // 128) * 128, mlp // 16), dtype=torch.float8_e4m3fn
        ),
        codes.new_empty((m,), dtype=torch.float32),
    )


def _forward(self, x):
    from gleipnir.serving.fp4_geometry_producers import _project

    if not isinstance(x, PackedNvfp4):
        raise ValueError("native SwiGLU requires the fused normalization producer")
    weight, original = self._swiglu_weight, self.gate_up_proj
    codes, sf, inverse = native_producer(
        x.codes,
        x.scales,
        x.inverse,
        weight.codes,
        weight.scales,
        weight.inverse,
        original.weight,
        original.weight_scale,
        original.weight_inverse,
        self._swiglu_layer,
    )
    return _project(self.down_proj, PackedNvfp4(codes, sf, inverse))


def install(model, text: Any, plan: NativeSwiGlu, band: tuple[int, int], audit) -> dict:
    """Interleave serving copies of every gate/up weight; keep the originals.

    Requires ``fp4_geometry_producers.install`` first (packed norm inputs and
    the fused SwiGLU producer for rows outside ``band``).
    """
    from gleipnir.serving import fp4_geometry_producers as producers

    if (plan.hidden, plan.mlp) != (text.hidden_size, text.intermediate_size):
        raise ValueError("native SwiGLU plan does not match the served geometry")
    if not 1 <= band[0] <= band[1] <= ENVELOPE:
        raise ValueError(f"native SwiGLU row band outside the envelope: {band}")
    _STATE.update(plan=plan, band=tuple(band), audit=audit)
    count = 0
    for name, _norm in model.named_modules():
        if not name.endswith(".post_attention_layernorm"):
            continue
        mlp = model.get_submodule(name.rsplit(".", 1)[0]).mlp
        if getattr(mlp.forward, "__func__", None) is not producers._mlp_forward:
            raise ValueError("native SwiGLU requires the fused producer install")
        projection = mlp.gate_up_proj
        mlp._swiglu_weight = prepare_weight(
            PackedNvfp4(
                projection.weight, projection.weight_scale, projection.weight_inverse
            )
        )
        mlp._swiglu_layer = count
        mlp.forward = MethodType(_forward, mlp)
        count += 1
    if count != len(text.layer_types):
        raise ValueError(f"incomplete native SwiGLU scope: {count}")
    return {
        "mlp_count": count,
        "band": list(band),
        "tile": list(plan.tile),
        "cluster": list(plan.cluster),
        "weight_requantization": False,
    }
