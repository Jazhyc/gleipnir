"""Focused checks for asynchronous GPU attribution and duration conservation."""

import pytest

from experiments.fp4_stability.profile_breakdown import cpu_contexts, summarize_trace


def cpu(name, external_id, start, duration, tid=1):
    return {
        "cat": "cpu_op",
        "ph": "X",
        "name": name,
        "pid": 1,
        "tid": tid,
        "ts": start,
        "dur": duration,
        "args": {"External id": external_id},
    }


def kernel(name, external_id, duration):
    return {
        "cat": "kernel",
        "ph": "X",
        "name": name,
        "pid": 0,
        "tid": 7,
        "ts": 1000,
        "dur": duration,
        "args": {"External id": external_id},
    }


def test_external_id_tracks_nested_scope_after_cpu_call_returns():
    events = [
        cpu("FrozenFp4Function", 1, 0, 10),
        cpu("aten::mm", 2, 1, 3),
        kernel("gemm", 2, 500_000),
    ]
    assert summarize_trace(events)["kernel_scopes"] == [
        {"scope": "native_fp4_forward", "calls": 1, "seconds": 0.5}
    ]


def test_other_partition_conserves_activity_and_excludes_annotations():
    events = [
        cpu("aten::copy_", 1, 0, 10),
        cpu("aten::mm", 2, 20, 10),
        kernel("direct_copy_kernel_cuda", 1, 100_000),
        kernel("nvjet_matrix", 2, 200_000),
        kernel("unknown_kernel", 99, 300_000),
        {
            "cat": "gpu_user_annotation",
            "ph": "X",
            "name": "annotation",
            "dur": 9_000_000,
        },
    ]
    result = summarize_trace(events)
    assert sum(x["calls"] for x in result["other_families"]) == 3
    assert sum(x["seconds"] for x in result["other_families"]) == pytest.approx(0.6)
    assert result["unmapped_kernels"] == {"calls": 1, "seconds": 0.3}


def test_fla_distinguishes_original_checkpoint_and_backward():
    events = [
        cpu("ChunkGatedDeltaRuleFunction", 1, 0, 10),
        cpu("autograd::engine::evaluate_function: AddBackward0", 2, 20, 40, tid=2),
        cpu("ChunkGatedDeltaRuleFunction", 3, 21, 10, tid=2),
        cpu("ChunkGatedDeltaRuleFunctionBackward", 4, 40, 10, tid=2),
        kernel("chunk_fwd_kernel_o", 1, 100_000),
        kernel("chunk_fwd_kernel_o", 3, 200_000),
        kernel("recompute_w_u_fwd_kernel", 4, 300_000),
    ]
    phases = {x["phase"]: x["seconds"] for x in summarize_trace(events)["fla_phases"]}
    assert phases == {
        "original_forward": 0.1,
        "checkpoint_forward": 0.2,
        "backward": 0.3,
    }


def test_sibling_cpu_scope_does_not_inherit_fla():
    events = [
        cpu("ChunkGatedDeltaRuleFunction", 1, 0, 10),
        cpu("aten::mul", 2, 20, 10),
        kernel("elementwise_kernel", 2, 100_000),
    ]
    assert summarize_trace(events)["kernel_scopes"][0]["scope"] == "other"


def test_duplicate_external_ids_rejected():
    with pytest.raises(ValueError, match="duplicate CPU external ID"):
        cpu_contexts([cpu("aten::mm", 1, 0, 1), cpu("aten::mm", 1, 2, 1)])


def test_multiple_cpu_processes_rejected_instead_of_misattributed():
    first = cpu("aten::mm", 1, 0, 1)
    second = {**cpu("aten::mm", 2, 0, 1), "pid": 2}
    with pytest.raises(ValueError, match="single CPU process"):
        cpu_contexts([first, second])
