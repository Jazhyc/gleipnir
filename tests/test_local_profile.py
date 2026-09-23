import pytest

from experiments.local_inference.profile_summary import kernel_family, summarize_trace


def test_conservative_kernel_families():
    assert kernel_family("ampere_bf16_gemm") == "matrix_multiplication"
    assert kernel_family("flash::flash_fwd_splitkv_kernel") == "flash_attention"
    assert kernel_family("chunk_fwd_kernel_o") == "named_gdn_and_convolution"
    assert kernel_family("reshape_and_cache_flash_kernel") == "kv_write_and_bookkeeping"
    assert kernel_family("triton_poi_fused_0") == "other"


def test_reject_cpu_only_trace():
    with pytest.raises(ValueError, match="CPU-only"):
        summarize_trace({"traceEvents": [{"cat": "cpu_op", "dur": 1000}]})


def test_kernel_duration_and_union_exclude_cpu_and_handle_overlap():
    trace = {
        "traceEvents": [
            {"cat": "kernel", "ph": "X", "name": "a", "ts": 0, "dur": 10},
            {"cat": "kernel", "ph": "X", "name": "a", "ts": 5, "dur": 10},
            {"cat": "kernel", "ph": "X", "name": "b", "ts": 20, "dur": 5},
            {"cat": "cpu_op", "ph": "X", "name": "cpu", "ts": 0, "dur": 1000},
        ]
    }
    result = summarize_trace(trace)
    assert result["kernel_count"] == 3
    assert result["summed_kernel_seconds"] == 25 / 1e6
    assert result["kernel_active_union_seconds"] == 20 / 1e6
    assert result["kernel_active_fraction"] == 0.8
    assert result["kernels"][0]["percent_summed_kernel_time"] == 80
