"""Opt-in NVIDIA FROST row-descaling epilogue, with raw BF16 rounding retained."""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from gleipnir.cudnn_fp4_gemm import Nvfp4Gemm, PackedNvfp4


@triton.jit(do_not_specialize=["ROWS"])
def _row_scale(IA, IB, SCALE, ROWS, BLOCK: tl.constexpr):
    r = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    a = tl.load(IA + r, r < ROWS, other=1)
    b = tl.load(IB)
    tl.store(SCALE + r, 1.0 / (a * b), r < ROWS)


class Nvfp4ScaledGemm:
    """Adapt the pinned projection graph with a rowwise FP32 multiply epilogue.

    ``runtime_m`` opts into NVIDIA's symbolic M launch support. K/N, scale
    layouts, tile configuration and BF16 rounding remain fixed by the plan.
    """

    def __init__(self, m: int, k: int, n: int, *, runtime_m: bool = False) -> None:
        import cudnn
        from cudnn.gemm.frost.compiler import jit_from_cudnn_graph

        # Planning freezes graph mutation. Recreate the same operand/layout
        # contract, retaining the original arithmetic and tile control.
        self.runtime_m = runtime_m
        self.reference = Nvfp4Gemm(m, k, n)
        self.plan = self.reference.plan
        graph = cudnn.pygraph(
            io_data_type=cudnn.data_type.BFLOAT16,
            intermediate_data_type=cudnn.data_type.FLOAT,
            compute_data_type=cudnn.data_type.FLOAT,
        )
        self.graph = graph

        def operand(name, source, *, scales=False):
            return graph.tensor(
                name=name,
                dim=source.get_dim(),
                stride=source.get_stride(),
                data_type=source.get_data_type(),
                **(
                    {"reordering_type": cudnn.tensor_reordering.F8_128x4}
                    if scales
                    else {}
                ),
            )

        self.a_tensor = operand("A", self.plan.a)
        self.b_tensor = operand("B", self.plan.b)
        self.sfa_tensor = operand("SFA", self.plan.sfa, scales=True)
        self.sfb_tensor = operand("SFB", self.plan.sfb, scales=True)
        ad = graph.block_scale_dequantize(
            input=self.a_tensor, descale=self.sfa_tensor, block_size=[1, 16]
        )
        bd = graph.block_scale_dequantize(
            input=self.b_tensor, descale=self.sfb_tensor, block_size=[16, 1]
        )
        raw = graph.matmul(A=ad, B=bd, name="gleipnir_row_scale_matmul").set_data_type(
            cudnn.data_type.BFLOAT16
        )
        self.scale_tensor = graph.tensor(
            name="gleipnir_row_descale",
            dim=[1, m, 1],
            stride=[m, 1, 1],
            data_type=cudnn.data_type.FLOAT,
        )
        self.output_tensor = (
            graph.mul(a=raw, b=self.scale_tensor, name="gleipnir_scaled_output")
            .set_output(True)
            .set_data_type(cudnn.data_type.BFLOAT16)
        )
        graph.validate()
        self.jit = jit_from_cudnn_graph(graph, config=self.plan.jit.config)
        self.workspace_bytes = int(getattr(self.jit, "workspace_bytes", 0) or 0)
        self.workspace = torch.empty(
            self.workspace_bytes, device="cuda", dtype=torch.uint8
        )

    def _operand_rows(self, a: PackedNvfp4, b: PackedNvfp4) -> int:
        m = a.codes.shape[0] if self.runtime_m else self.plan.m
        if m <= 0:
            raise ValueError("fused descale requires positive activation rows")
        if a.inverse.numel() != m or b.inverse.numel() != 1:
            raise ValueError("fused descale requires per-row A and scalar B inverses")
        if a.codes.shape != (m, self.plan.k // 2) or b.codes.shape != (
            self.plan.n,
            self.plan.k // 2,
        ):
            raise ValueError("fused descale operand geometry mismatch")
        return m

    def __call__(self, a: PackedNvfp4, b: PackedNvfp4) -> torch.Tensor:
        from cudnn.frost.workspace import Workspace
        from cudnn.gated_attention_block.kernels.proj_gemm import _rank3, _sf_view

        m = self._operand_rows(a, b)
        scale = torch.empty(m, device=a.codes.device, dtype=torch.float32)
        _row_scale[(triton.cdiv(m, 1024),)](
            a.inverse, b.inverse, scale, m, 1024
        )
        out = torch.empty(
            m, self.plan.n, device=a.codes.device, dtype=torch.bfloat16
        )
        bindings = {
            self.a_tensor: _rank3(a.codes, "a"),
            self.b_tensor: _rank3(b.codes, "b"),
            self.sfa_tensor: _sf_view(self.plan, a.scales, "sf_a", m),
            self.sfb_tensor: _sf_view(self.plan, b.scales, "sf_b", self.plan.n),
            self.scale_tensor: scale.view(1, m, 1),
            self.output_tensor: out.unsqueeze(0),
        }
        kwargs = {"stream": torch.cuda.current_stream(out.device).cuda_stream}
        if self.workspace_bytes:
            kwargs["workspace"] = Workspace(
                self.workspace,
                self.workspace_bytes,
                "gleipnir_fused_descale",
                device=out.device.index,
            )
        self.jit(bindings, **kwargs)
        return out
