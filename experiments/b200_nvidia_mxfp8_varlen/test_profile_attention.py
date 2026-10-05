"""Profiler attribution must exclude synthetic GPU scope ranges."""

import pytest

from experiments.b200_nvidia_mxfp8_varlen.profile_attention import attribute_kernels


def event(cat, name, start, duration, external, thread=1):
    return {
        "cat": cat,
        "name": name,
        "ts": start,
        "dur": duration,
        "ph": "X",
        "pid": 7,
        "tid": thread,
        "args": {"External id": external},
    }


def test_nested_launches_and_gpu_annotations_are_not_double_counted():
    events = [
        event("user_annotation", "mxfp8::quantize", 0, 20, 1),
        event("cpu_op", "aten::empty", 1, 2, 2),
        event("kernel", "actual_quantize", 100, 3, 1),
        event("gpu_user_annotation", "mxfp8::quantize", 100, 30, 1),
        event("cpu_op", "aten::copy", 21, 2, 3),
        event("kernel", "actual_copy", 110, 4, 3),
        event("kernel", "unmapped_kernel", 115, 5, 99),
    ]
    result = attribute_kernels(events)
    assert len(result["kernels"]) == 3
    assert result["gpu_kernel_stages"] == {
        "mxfp8::quantize": {"calls": 1, "us": 3},
        "other": {"calls": 1, "us": 4},
        "unmapped": {"calls": 1, "us": 5},
    }


def test_cpu_threads_do_not_inherit_each_others_scopes():
    result = attribute_kernels(
        [
            event("user_annotation", "mxfp8::quantize", 0, 20, 1),
            event("cpu_op", "aten::copy", 1, 2, 2, thread=2),
            event("kernel", "actual_copy", 100, 3, 2),
        ]
    )
    assert result["kernels"][0]["scope"] == "other"


def test_duplicate_external_ids_fail():
    with pytest.raises(ValueError, match="duplicate"):
        attribute_kernels(
            [event("cpu_op", "a", 0, 1, 1), event("cpu_op", "b", 2, 1, 1)]
        )


def test_launch_correlation_overrides_coarse_external_id():
    events = [
        event("cpu_op", "autograd_forward", 0, 20, 1),
        event("user_annotation", "mxfp8::quantize", 1, 10, 2),
        event("cuda_runtime", "cudaLaunchKernel", 2, 1, 1),
        event("kernel", "actual_quantize", 100, 3, 1),
    ]
    for e in events[2:]:
        e["args"]["correlation"] = 42
    assert attribute_kernels(events)["kernels"][0]["scope"] == "mxfp8::quantize"
