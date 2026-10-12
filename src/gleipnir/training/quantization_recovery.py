"""Opt-in merged-weight QAD; native deployment forwards and approximate STEs.

No base parameter or adapter state-dict key is replaced. Packed weights are
cached only until their FP32 adapters change. Imports of GPU kernels are lazy.
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from types import MethodType
from typing import Any

import torch
import torch.nn.functional as F


def effective_weight(module: Any) -> torch.Tensor:
    """Apply the export contract: FP32 accumulation, then BF16 conversion."""
    base = getattr(module, "base_layer", module).weight
    if not hasattr(module, "lora_A"):
        return base.detach()
    names = module.active_adapters
    if len(names) != 1 or module.merged or module.disable_adapters:
        raise ValueError("QAD requires one active, unmerged adapter")
    name = names[0]
    if module.use_dora.get(name, False):
        raise ValueError("QAD does not support DoRA")
    with torch.no_grad(), torch.autocast(base.device.type, enabled=False):
        delta = module.lora_B[name].weight.float() @ module.lora_A[name].weight.float()
        return (base.float() + delta * module.scaling[name]).to(torch.bfloat16)


def adapter_operands(module: Any) -> tuple[torch.Tensor, torch.Tensor, float]:
    """Supply empty operands for frozen projections without LoRA."""
    if hasattr(module, "lora_A"):
        name = module.active_adapters[0]
        return (
            module.lora_A[name].weight,
            module.lora_B[name].weight,
            module.scaling[name],
        )
    weight = module.weight
    return (
        weight.new_empty((0, weight.shape[1])),
        weight.new_empty((weight.shape[0], 0)),
        0.0,
    )


class _LowRankSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, a, b, output, xq, wq, scale):
        retained_x = xq if a.numel() else xq.new_empty((0, xq.shape[-1]))
        ctx.save_for_backward(retained_x, wq, a, b)
        ctx.scale = scale
        ctx.input_shape = x.shape
        return output

    @staticmethod
    def backward(ctx, grad):
        xq, wq, a, b = ctx.saved_tensors
        g = grad.reshape(-1, grad.shape[-1])
        # This is an approximate STE, with FP32 gradient storage. Avoid full dW
        # and the huge differentiable merged-weight graph for rank-128 training.
        with torch.autocast(g.device.type, dtype=torch.bfloat16, enabled=g.is_cuda):
            dx = g @ wq
            da = db = None
            if a.numel():
                if ctx.needs_input_grad[1]:
                    da = ((g @ b).t() @ xq) * ctx.scale
                if ctx.needs_input_grad[2]:
                    db = (g.t() @ (xq @ a.t())) * ctx.scale
        return (
            dx.reshape(ctx.input_shape),
            None if da is None else da.to(a.dtype),
            None if db is None else db.to(b.dtype),
            None,
            None,
            None,
            None,
        )


def low_rank_ste(x, module, output, xq, wq):
    a, b, scale = adapter_operands(module)
    return _LowRankSTE.apply(x, a, b, output, xq, wq, scale)


def fp8_decode(x: torch.Tensor, *, weight: bool) -> tuple[torch.Tensor, torch.Tensor]:
    """Serving PTPC E4M3: rows are weight channels or activation tokens."""
    with torch.no_grad():
        scale = x.float().abs().amax(-1, keepdim=True).clamp_min(1e-12) / 448
        codes = (x.float() / scale).clamp(-448, 448).to(torch.float8_e4m3fn)
        decoded = (codes.float() * scale).to(torch.bfloat16)
    return decoded, scale


class ProjectionGroup:
    """Share a fused weight scale while retaining separate HF projection names."""

    def __init__(self, modules: tuple[Any, ...], precision: str):
        self.modules, self.precision = modules, precision
        self.key = None
        self.operator_check = None
        self.packed = self.decoded = self.interleaved = None
        self.widths = [getattr(m, "base_layer", m).weight.shape[0] for m in modules]

    def prepare(self):
        from gleipnir.kernels.fp4.cudnn_fp4_gemm import pack_operand

        key = tuple(
            (p.data_ptr(), p._version) for m in self.modules for p in m.parameters()
        )
        if self.key != key:
            merged = torch.cat([effective_weight(m) for m in self.modules]).contiguous()
            if self.precision == "fp4":
                with torch.no_grad():
                    self.packed = pack_operand(merged, weight=True)
                    self.decoded = decode_fp4(self.packed)
            else:
                self.decoded, self.scales = fp8_decode(merged, weight=True)
                self.codes = (
                    (merged.float() / self.scales)
                    .clamp(-448, 448)
                    .to(torch.float8_e4m3fn)
                )
            self.interleaved = None
            self.key = key

    def forward(self, x, index: int):
        self.prepare()
        flat = x.reshape(-1, x.shape[-1]).contiguous()
        start, end = sum(self.widths[:index]), sum(self.widths[: index + 1])
        with torch.no_grad():
            if self.precision == "fp4":
                activation = pack_activation(flat)
                out = native_linear(activation, self.packed)[:, start:end]
                xq = (
                    decode_fp4(activation)
                    if hasattr(self.modules[index], "lora_A")
                    else flat.new_empty((0, flat.shape[-1]))
                )
                if self.operator_check is None:
                    from gleipnir.kernels.fp4.cudnn_fp4_gemm import decode_operand

                    rows, channels = min(17, len(flat)), min(64, end - start)
                    raw_a = decode_operand(activation)[:rows]
                    raw_a *= activation.inverse[:rows, None]
                    raw_b = decode_operand(self.packed)[start : start + channels]
                    raw_b *= self.packed.inverse
                    with torch.autocast(flat.device.type, enabled=False):
                        raw = (raw_a @ raw_b.t()).to(torch.bfloat16).float()
                    reference = (
                        raw / (activation.inverse[:rows, None] * self.packed.inverse)
                    ).to(torch.bfloat16)
                    measured = out[:rows, :channels].float()
                    relative = float(
                        (measured - reference.float()).norm()
                        / reference.float().norm().clamp_min(1e-12)
                    )
                    self.operator_check = {
                        "relative_l2": relative,
                        "rows": rows,
                        "channels": channels,
                        "limit": 0.01,
                        "passed": bool(torch.isfinite(measured).all())
                        and relative <= 0.01,
                    }
                    if not self.operator_check["passed"]:
                        raise ValueError(
                            f"QAD native FP4/reference mismatch: {relative}"
                        )
            else:
                xq, scale = fp8_decode(flat, weight=False)
                codes = (flat.float() / scale).clamp(-448, 448).to(torch.float8_e4m3fn)
                # FP32 accumulation, then both FP32 descales, then BF16 output.
                # The native serving CUTLASS operator is checked against this
                # quantized reference, rather than a BF16 dequantized GEMM.
                with torch.autocast(flat.device.type, enabled=False):
                    out = (
                        (codes.float() @ self.codes[start:end].float().t())
                        * scale
                        * self.scales[start:end].t()
                    ).to(torch.bfloat16)
        result = low_rank_ste(
            flat, self.modules[index], out, xq, self.decoded[start:end]
        )
        return result.reshape(*x.shape[:-1], end - start)


_PLANS: dict[tuple, Any] = {}


def pack_activation(x):
    from gleipnir.kernels.fp4.cudnn_fp4_gemm import pack_operand

    return pack_operand(x, row_amax=True, chunked_rows=True, hardware_packing=True)


def decode_fp4(operand):
    from gleipnir.kernels.fp4.decode import decode_bf16

    return decode_bf16(operand)


def native_linear(a, b):
    from gleipnir.kernels.fp4.cudnn_fp4_epilogue import Nvfp4ScaledGemm

    key = (a.codes.device.index, a.codes.shape[1] * 2, b.codes.shape[0])
    if key not in _PLANS:
        _PLANS[key] = Nvfp4ScaledGemm(128, key[1], key[2], runtime_m=True)
    return _PLANS[key](a, b)


class _ExactForward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, surrogate, native):
        return native

    @staticmethod
    def backward(ctx, grad):
        return grad, None


class _PackedMLP(torch.autograd.Function):
    """Retain FP4 operands; reconstruct the same SiLU STE locally in backward."""

    @staticmethod
    def forward(ctx, x, ga, gb, ua, ub, da, db, pair, down, producer):
        from gleipnir.serving.fp4.fusion import silu_pack
        from gleipnir.serving.fp4.swiglu import prepare_weight
        from gleipnir.serving.fp4.swiglu_pack import activated_pack

        flat = x.reshape(-1, x.shape[-1]).contiguous()
        pair.prepare()
        down.prepare()
        a = pack_activation(flat)
        if len(flat) < 1536:
            h = silu_pack(native_linear(a, pair.packed))
        else:
            if pair.interleaved is None:
                pair.interleaved = prepare_weight(pair.packed)
            if len(flat) <= 4096:
                h = activated_pack(producer.overhead(a, pair.interleaved))
            else:
                h = producer(a, pair.interleaved)
        out = native_linear(h, down.packed)
        ctx.save_for_backward(
            a.codes,
            a.scales,
            a.inverse,
            h.codes,
            h.scales,
            h.inverse,
            ga,
            gb,
            ua,
            ub,
            da,
            db,
        )
        ctx.pair, ctx.down, ctx.input_shape = pair, down, x.shape
        return out.reshape_as(x)

    @staticmethod
    def backward(ctx, grad):
        from gleipnir.kernels.fp4.cudnn_fp4_gemm import PackedNvfp4

        ac, asc, ai, hc, hsc, hi, *masters = ctx.saved_tensors
        a, h = PackedNvfp4(ac, asc, ai), PackedNvfp4(hc, hsc, hi)
        pair, down = ctx.pair, ctx.down
        with torch.no_grad():
            xq, hq = decode_fp4(a), decode_fp4(h)
            gateup = native_linear(a, pair.packed)
            output = native_linear(h, down.packed)
        with torch.enable_grad():
            x = xq.detach().requires_grad_(True)
            operands = [p.detach().requires_grad_(True) for p in masters]
            ga, gb, ua, ub, da, db = operands
            width = pair.widths[0]
            gate = _LowRankSTE.apply(
                x,
                ga,
                gb,
                gateup[:, :width],
                xq,
                pair.decoded[:width],
                adapter_operands(pair.modules[0])[2],
            )
            up = _LowRankSTE.apply(
                x,
                ua,
                ub,
                gateup[:, width:],
                xq,
                pair.decoded[width:],
                adapter_operands(pair.modules[1])[2],
            )
            hidden = F.silu(gate) * up
            out = _LowRankSTE.apply(
                hidden,
                da,
                db,
                output,
                hq,
                down.decoded,
                adapter_operands(down.modules[0])[2],
            )
            gradients = torch.autograd.grad(out, (x, *operands), grad.reshape_as(out))
        return gradients[0].reshape(ctx.input_shape), *gradients[1:], None, None, None


def mxfp8_ste(x, *, column: bool = False):
    """E8M0 groups of 32, with a rounding STE and detached scales."""
    if column:
        length = x.shape[0]
        padded = F.pad(x, (0, 0, 0, 0, 0, (-length) % 32))
        groups = padded.transpose(0, 2).reshape(x.shape[2], x.shape[1], -1, 32)
    else:
        groups = x.reshape(*x.shape[:-1], -1, 32)
    with torch.no_grad():
        scale = torch.pow(
            2.0,
            torch.ceil(
                torch.log2(
                    groups.float().abs().amax(-1, keepdim=True).clamp_min(1e-12) / 448
                )
            ),
        )
        decoded = ((groups.float() / scale).to(torch.float8_e4m3fn).float() * scale).to(
            x.dtype
        )
        if column:
            decoded = (
                decoded.reshape(x.shape[2], x.shape[1], -1)
                .transpose(0, 2)[:length]
                .contiguous()
            )
        else:
            decoded = decoded.reshape_as(x)
    return _ExactForward.apply(x, decoded)


def install(model: Any, root: Path) -> dict[str, Any]:
    """Install merged QAD projections and the exact serving SwiGLU row policy."""
    from gleipnir.serving.fp4.swiglu_native_output import (
        NativeOutputSwiGlu,
        load_kernel,
    )
    from gleipnir.serving.fp4.swiglu_overhead import OverheadSwiGlu
    from gleipnir.serving.fp4.swiglu_overhead import load_kernel as load_medium

    native, _ = load_kernel(
        root, root / "results/quantization-recovery01/swiglu_source"
    )
    producer = NativeOutputSwiGlu(native)
    producer.compile()
    medium, _ = load_medium(
        root, root / "results/quantization-recovery01/swiglu_medium_source"
    )
    producer.overhead = OverheadSwiGlu(medium, pad=False)
    producer.overhead.compile()
    counts = {"mlp": 0, "gdn": 0, "attention": 0}
    groups = []

    def project(modules, precision):
        group = ProjectionGroup(tuple(modules), precision)
        groups.append(group)
        for i, module in enumerate(modules):

            def forward(self, x, *args, _index=i, _group=group, **kwargs):
                return _group.forward(x, _index)

            module.forward = MethodType(forward, module)
        return group

    for module in list(model.modules()):
        if type(module).__name__ == "Qwen3_5MLP":
            pair = project([module.gate_proj, module.up_proj], "fp4")
            down = project([module.down_proj], "fp4")

            def mlp(self, x, _pair=pair, _down=down):
                operands = [
                    p
                    for projection in (self.gate_proj, self.up_proj, self.down_proj)
                    for p in adapter_operands(projection)[:2]
                ]
                return _PackedMLP.apply(x, *operands, _pair, _down, producer)

            module.forward = MethodType(mlp, module)
            counts["mlp"] += 1
        elif type(module).__name__ == "Qwen3_5GatedDeltaNet":
            project([module.in_proj_qkv, module.in_proj_z], "fp4")
            project([module.out_proj], "fp4")
            counts["gdn"] += 1
        elif type(module).__name__ == "Qwen3_5Attention":
            project([module.q_proj, module.k_proj, module.v_proj], "fp8")
            project([module.o_proj], "fp8")
            counts["attention"] += 1
    if counts != {"mlp": 32, "gdn": 24, "attention": 8}:
        raise ValueError(f"QAD projection scope drift: {counts}")
    model._gleipnir_qad_groups = groups
    return {
        **counts,
        "effective_weight": "FP32_accumulate_BF16_merge_before_quantization",
        "backward": "low_rank_STE_BF16_contractions_FP32_gradients",
    }


@contextmanager
def attention_runtime():
    """Use serving MXFP8 forward with an isolated quantized-QKV FA4 STE."""
    from flash_attn.cute import flash_attn_varlen_func
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    from experiments.filtered_fp4_baseline.compiler_repair import install_repair
    from gleipnir.serving.mxfp8 import forward_plan, produce
    from gleipnir.training.packing import PackedSequenceLayout

    install_repair()
    original = ALL_ATTENTION_FUNCTIONS["sdpa"]

    def attention(module, query, key, value, attention_mask, **kwargs):
        if attention_mask is not None or query.shape[0] != 1:
            raise ValueError("QAD attention requires explicit unpadded boundaries")
        offsets = kwargs.get("cu_seq_lens_q")
        if offsets is None or not torch.equal(offsets, kwargs["cu_seq_lens_k"]):
            raise ValueError("QAD self-attention boundaries missing or unequal")
        cuts = offsets.tolist()
        layout = PackedSequenceLayout(
            tuple(b - a for a, b in zip(cuts[:-1], cuts[1:], strict=True))
        )
        if (
            cuts[0] != 0
            or layout.total_tokens != query.shape[2]
            or kwargs.get("dropout", 0) != 0
        ):
            raise ValueError("QAD attention boundary/dropout drift")
        q, k, v = [x[0].transpose(0, 1).contiguous() for x in (query, key, value)]
        maximum = max(layout.lengths)
        with torch.no_grad():
            qr, sq = produce(q, offsets, maximum)
            kr, sk = produce(k, offsets, maximum)
            vc, sv = produce(v, offsets, maximum, column=True)
            plan = forward_plan(q.device)
            out = torch.empty_like(q)
            workspace = torch.empty(
                max(1, plan.scratch_workspace_bytes()),
                device=q.device,
                dtype=torch.uint8,
            )
            plan.execute(
                qr,
                kr,
                vc,
                out,
                seq_q_lens=offsets,
                seq_kv_lens=offsets,
                sf_q=sq,
                sf_k=sk,
                sf_v=sv,
                workspace=workspace,
            )
        if torch.is_grad_enabled():
            # V scale groups must restart at every original sequence boundary.
            qd = mxfp8_ste(q)
            kd = mxfp8_ste(k)
            vd = torch.cat(
                [
                    mxfp8_ste(v[a:b], column=True)
                    for a, b in zip(cuts[:-1], cuts[1:], strict=True)
                ]
            )
            surrogate = flash_attn_varlen_func(
                qd,
                kd,
                vd,
                cu_seqlens_q=offsets,
                cu_seqlens_k=offsets,
                max_seqlen_q=maximum,
                max_seqlen_k=maximum,
                softmax_scale=0.0625,
                causal=True,
            )
            if isinstance(surrogate, tuple):
                surrogate = surrogate[0]
            out = _ExactForward.apply(surrogate, out)
        return out.unsqueeze(0), None

    attention._gleipnir_packed_boundaries = True
    ALL_ATTENTION_FUNCTIONS.register("sdpa", attention)
    try:
        yield
    finally:
        ALL_ATTENTION_FUNCTIONS.register("sdpa", original)
