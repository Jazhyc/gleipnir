"""Forward-only GEMM/LoRA-add/head-norm/RoPE/FP8 epilogue pilot.

The production LoRA masters and BF16 weights stay unchanged. A frozen FP8
weight copy is an explicit experimental operand, not a serving artifact.
No native projection backward is claimed by this standalone pilot.
"""

import torch


def quantize_weight(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Independent canonical rowwise MXFP8 weights with power-of-two scales."""
    rows, inner = weight.shape
    if weight.dtype != torch.bfloat16 or inner % 32:
        raise ValueError("projection weights require BF16 and K divisible by 32")
    blocks = weight.float().reshape(rows, inner // 32, 32)
    amax = blocks.abs().amax(-1)
    exponent = torch.ceil(torch.log2(amax / 448.0)).clamp(-126, 127)
    exponent = torch.where(amax == 0, 0, exponent)
    scale = 2.0**exponent
    payload = (blocks / scale[..., None]).to(torch.float8_e4m3fn).reshape(weight.shape)
    return payload, (exponent + 127).to(torch.uint8)


def project_prepare(
    hidden,
    weight,
    cumulative,
    maximum,
    *,
    heads,
    head_stride=256,
    update=None,
    norm_weight=None,
    cos=None,
    sin=None,
    weight_sf=None,
    eps=1e-6,
):
    """Project straight to compact FP8 plus native scale layouts; no BF16 slab."""
    from gleipnir.nvidia_mxfp8_fused_quantize import _prepare

    if hidden.ndim != 2 or not hidden.is_contiguous() or hidden.dtype != torch.bfloat16:
        raise ValueError(
            "projection pilot requires contiguous BF16 [T,K] hidden states"
        )
    total, inner = hidden.shape
    if (
        inner % 32
        or weight.shape[1] != inner
        or weight.shape[0] < (heads - 1) * head_stride + 256
    ):
        raise ValueError("projection geometry/weight span is invalid")
    mxfp8 = weight_sf is not None
    expected_dtype = torch.float8_e4m3fn if mxfp8 else torch.bfloat16
    if weight.dtype != expected_dtype or not weight.is_contiguous():
        raise ValueError("projection weight precision/layout mismatch")
    if mxfp8 and (
        weight_sf.shape != (weight.shape[0], inner // 32)
        or weight_sf.dtype != torch.uint8
        or not weight_sf.is_contiguous()
    ):
        raise ValueError("projection weight scale layout mismatch")
    if update is not None and (
        update.shape != (total, heads, 256)
        or update.dtype not in (torch.bfloat16, torch.float32)
        or not update.is_contiguous()
    ):
        raise ValueError("projection LoRA update shape/precision mismatch")
    transform = norm_weight is not None
    if transform and (
        norm_weight.requires_grad
        or norm_weight.shape != (256,)
        or cos is None
        or sin is None
        or cos.shape != sin.shape
        or cos.shape[0] != total
        or cos.shape[1] % 2
        or cos.shape[1] > 256
        or not cos.is_contiguous()
        or not sin.is_contiguous()
    ):
        raise ValueError("projection norm/rotary contract mismatch")
    tensors = [weight, cumulative, update, norm_weight, cos, sin, weight_sf]
    if any(t is not None and t.device != hidden.device for t in tensors):
        raise ValueError("projection operands must share a device")
    from gleipnir.nvidia_mxfp8_fused_attention import validate_layout

    validate_layout(hidden, cumulative, maximum)
    batch = cumulative.numel() - 1
    tiles = (maximum + 127) // 128
    capacity = (total + 127) // 128 + batch
    payload = torch.empty(
        (total, heads, 256), device=hidden.device, dtype=torch.float8_e4m3fn
    )
    sizes = (
        batch * heads * tiles * 4096,
        batch * heads * tiles * 2048,
        batch * heads * tiles * 2048,
        heads * capacity * 1024,
        heads * capacity * 1024,
    )
    scales = tuple(
        torch.empty(n, device=hidden.device, dtype=torch.uint8) for n in sizes
    )
    _prepare[(tiles, heads, batch)](
        hidden,
        payload,
        payload,
        *scales,
        cumulative,
        maximum,
        capacity,
        heads,
        batch,
        True,
        norm_weight if transform else hidden,
        cos if transform else hidden,
        sin if transform else hidden,
        inner,
        256,
        eps,
        cos.shape[1] if transform else 0,
        transform,
        weight,
        update if update is not None else hidden,
        weight_sf if mxfp8 else hidden,
        inner,
        head_stride,
        True,
        mxfp8,
        update is not None,
        num_warps=8,
        enable_fp_fusion=False,
    )
    return payload, payload, *scales
