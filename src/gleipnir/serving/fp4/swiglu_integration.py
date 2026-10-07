"""Audited large-row fused producer; retain the selected producer for tiny M."""

from types import MethodType

import torch

from gleipnir import serving_fp4_integration as preparation
from gleipnir.cudnn_fp4_gemm import PackedNvfp4
from gleipnir.serving_fp4_swiglu import prepare_weight
from gleipnir.serving_fp4_swiglu_pack import activated_pack

_PLAN = None
_AUDIT = None


@torch.library.custom_op("gleipnir::nvfp4_gemm_swiglu_pack", mutates_args=())
def fused_producer(
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
    m = codes.shape[0]
    if _PLAN is None:
        raise ValueError("fused SwiGLU called outside its selected row policy")
    if m < 1536:
        gateup = preparation.packed_linear(
            codes, sf, inverse, original_weight, original_sf, original_inverse
        )
        result = preparation.silu_producer(gateup)
        _AUDIT(layer, m, "reference_small_rows")
        return result
    padded = ((m + 255) // 256) * 256
    if torch.cuda.is_current_stream_capturing() and padded not in _PLAN.plans:
        raise RuntimeError("warm native SwiGLU plan before CUDA graph capture")
    activated = _PLAN(
        PackedNvfp4(codes.view(torch.float4_e2m1fn_x2), sf, inverse),
        PackedNvfp4(weight, weight_sf, weight_inverse),
    )
    a = activated_pack(activated)
    if preparation._AUDIT is not None:
        preparation._AUDIT("silu", (m, 18432))
    _AUDIT(layer, m, "fused")
    return a.codes.view(torch.uint8), a.scales, a.inverse


@fused_producer.register_fake
def _fake_producer(
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
    m = codes.shape[0]
    return (
        codes.new_empty((m, 4608), dtype=torch.uint8),
        codes.new_empty((((m + 127) // 128) * 128, 576), dtype=torch.float8_e4m3fn),
        codes.new_empty((m,), dtype=torch.float32),
    )


def _forward(self, x):
    if not isinstance(x, PackedNvfp4):
        raise ValueError("fused MLP requires the recorded normalization producer")
    weight = self._swiglu_weight
    original = self.gate_up_proj
    codes, sf, inverse = fused_producer(
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
    return preparation._project(self.down_proj, PackedNvfp4(codes, sf, inverse))


def install(model, plan, audit) -> dict:
    """Interleave reconstructed frozen serving copies once; retain original weights."""
    global _PLAN, _AUDIT
    _PLAN, _AUDIT = plan, audit
    count = 0
    for name, _norm in model.named_modules():
        if not name.endswith(".post_attention_layernorm"):
            continue
        mlp = model.get_submodule(name.rsplit(".", 1)[0]).mlp
        if not mlp._fp4_norm or not mlp._fp4_silu:
            raise ValueError("fused SwiGLU requires combined normalization/packing")
        projection = mlp.gate_up_proj
        mlp._swiglu_weight = prepare_weight(
            PackedNvfp4(
                projection.weight,
                projection.weight_scale,
                projection.weight_inverse,
            )
        )
        mlp._swiglu_layer = count
        mlp.forward = MethodType(_forward, mlp)
        count += 1
    if count != 32:
        raise ValueError("incomplete fused SwiGLU model scope")
    return {"mlp_count": count, "minimum_rows": 1536, "weight_requantization": False}
