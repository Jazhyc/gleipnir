"""Experimental native MXFP8 backward with hardware-exp2 forward statistics."""

import hashlib
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path
from unittest.mock import patch

import torch


@lru_cache(maxsize=4)
def forward_with_statistics(device: torch.device):
    """Retain the serving kernel arithmetic while emitting token-major LSE."""
    from cudnn.api_base import TensorDesc
    from cudnn.frost.template_loader import load_template
    from cudnn.sdpa.fwd import SdpaFwdDslSm100, api_dsl

    from experiments.filtered_fp4_baseline.compiler_repair import install_repair
    from gleipnir.nvidia_mxfp8_attention import _cudnn
    from gleipnir.serving import mxfp8 as serving

    install_repair()
    _cudnn()
    maximum, batch = serving.MAX_LENGTH, serving.MAX_BATCH

    def descriptor(heads, dtype):
        return TensorDesc(
            dtype,
            (batch, heads, maximum, 256),
            (maximum * heads * 256, 256, heads * 256, 1),
            (3, 1, 2, 0),
            device,
        )

    original = api_dsl._load_sm100_kernel_module

    def loader(flavor, params, **kwargs):
        upstream = original(flavor, params, **kwargs)
        if flavor != (256, 256) or kwargs.get("pertensor") or kwargs.get("rubin"):
            raise ValueError("native QAT backward requires the SM100 D256 kernel")
        source = serving.hardware_exp2_source(Path(upstream.__file__).read_text())
        digest = hashlib.sha256(source.encode()).hexdigest()
        target = Path(".cache/kernels/nvidia_mxfp8/generated") / f"qat_lse_{digest}.py"
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and target.read_text() != source:
            raise ValueError("native QAT generated-source drift")
        target.write_text(source)
        return load_template(str(target.resolve()), params, tag="qat_lse_" + digest)

    lse = TensorDesc(
        torch.float32, (batch, 16, maximum), (maximum * 16, 1, 16), (1, 2, 0), device
    )
    with patch.object(api_dsl, "_load_sm100_kernel_module", loader):
        plan = SdpaFwdDslSm100(
            descriptor(16, torch.float8_e4m3fn),
            descriptor(4, torch.float8_e4m3fn),
            descriptor(4, torch.float8_e4m3fn),
            descriptor(16, torch.bfloat16),
            sample_lse=lse,
            is_causal=True,
            causal_bottom_right=True,
            scale_softmax=0.0625,
            thd=True,
            seq_q_lens_present=False,
            seq_kv_lens_present=True,
            cu_seq_q_lens=True,
            cu_seq_kv_lens=True,
            cga=1,
            split_kv=1,
            dtype_o=torch.bfloat16,
        )
        plan.check_support()
        plan.compile()
    return plan


def native_attention(q, k, v, cumulative, maximum, *, scale=0.0625):
    """Use validated packed backward layouts with a matching forward/LSE pair."""
    import gleipnir.nvidia_mxfp8_varlen_attention as packed
    import gleipnir.nvidia_mxfp8_varlen_repack as repack

    if q.shape[1:] != (16, 256) or k.shape[1:] != (4, 256) or v.shape != k.shape:
        raise ValueError("native QAT backward fixes Q/KV heads to 16/4 and D256")
    if maximum > 32768:
        raise ValueError("native QAT backward exceeds the serving context envelope")

    def plan(hq, hk, device):
        if (hq, hk) != (16, 4):
            raise ValueError("native QAT head contract drift")
        return forward_with_statistics(device)

    # The serving forward returns quantized V even for singleton sequences.
    # Keep that forward; the existing singleton backward is its identity STE.
    with (
        patch.object(packed, "forward_plan", plan),
        patch.object(
            repack,
            "singleton_forward",
            lambda *args: None,
        ),
    ):
        return packed.packed_attention(q, k, v, cumulative, maximum, scale=scale)


@contextmanager
def attention_runtime():
    """Route packed Qwen attention through native MXFP8 forward and backward."""
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    from gleipnir.nvidia_mxfp8_varlen_attention import packed_mxfp8_interface

    original = ALL_ATTENTION_FUNCTIONS["sdpa"]
    attention = packed_mxfp8_interface(original, native_attention)
    attention._gleipnir_packed_boundaries = True
    ALL_ATTENTION_FUNCTIONS.register("sdpa", attention)
    try:
        yield
    finally:
        ALL_ATTENTION_FUNCTIONS.register("sdpa", original)
