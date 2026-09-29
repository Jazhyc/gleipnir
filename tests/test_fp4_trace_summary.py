"""Trace categorization avoids counting attention templates as GEMM work."""

from experiments.fp4_inference.trace_summary import category


def test_attention_template_cutlass_types_do_not_become_linear():
    assert (
        category("flash::flash_fwd_splitkv_kernel<cutlass::bfloat16_t>") == "attention"
    )
    assert (
        category("triton_red_fused_abs_cutlass_scaled_mm_max")
        == "fused_elementwise_reduction"
    )
    assert category("cutlass_80_tensorop_bf16_gemm") == "linear_gemm"
    assert category("chunk_gated_delta_rule_fwd_kernel_h_blockdim64") == "gdn_and_conv"
