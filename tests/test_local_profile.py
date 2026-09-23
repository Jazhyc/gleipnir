import pytest

from experiments.local_inference import profile
from experiments.local_inference.profile_summary import kernel_family, summarize_trace
from experiments.local_inference.shape_summary import summarize_shapes


def test_shape_correlation_uses_external_id_not_cpu_duration():
    events = [
        {
            "cat": "cpu_op",
            "name": "aten::mm",
            "dur": 999,
            "args": {"External id": 7, "Input Dims": [[2048, 2560], [2560, 18432]]},
        },
        {
            "cat": "kernel",
            "ph": "X",
            "name": "gemm",
            "dur": 12,
            "args": {"External id": 7},
        },
        {"cat": "kernel", "ph": "X", "name": "flash", "dur": 8, "args": {}},
    ]
    report = summarize_shapes(events)
    assert report["shapes"][0]["projection"] == "mlp.gate_up_proj"
    assert report["shapes"][0]["percent_all_kernel_time"] == 60
    assert report["matched_mm_seconds"] == 12 / 1e6
    with pytest.raises(ValueError, match="No GPU"):
        summarize_shapes(events[:1])


def test_reject_competing_cupti_subscribers(tmp_path, monkeypatch):
    target = tmp_path / "capture"
    monkeypatch.setattr(
        profile.sys,
        "argv",
        ["profile", "--output", str(target), "--kind", "cuda", "--early-cupti"],
    )
    with pytest.raises(SystemExit) as error:
        profile.main()
    assert error.value.code == 2
    assert not target.exists()


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
