"""Trace accounting excludes annotations and respects CPU thread ancestry."""

from experiments.b200_mlp_gemm.analyze_hotpath_profile import analyze


def event(name, cat, ts, dur, tid=1, **args):
    return dict(name=name, cat=cat, ts=ts, dur=dur, tid=tid, ph="X", args=args)


def test_sync_parent_attribution_does_not_cross_threads_or_expired_ranges():
    events = [
        event("ChunkGatedDeltaRuleFunction", "cpu_op", 0, 100),
        event("cudaStreamSynchronize", "cuda_runtime", 10, 5),
        event("cudaStreamSynchronize", "cuda_runtime", 20, 5, tid=2),
        event("aten::repeat_interleave", "cpu_op", 110, 30),
        event("cudaStreamSynchronize", "cuda_runtime", 115, 5),
    ]
    r = analyze(events)
    assert r["stream_synchronization_calls"] == 3
    assert r["synchronization_callers"]["flashqla_metadata"]["calls"] == 1
    assert r["synchronization_callers"]["sequence_id_construction"]["calls"] == 1
    assert r["synchronization_callers"]["unattributed"]["calls"] == 1


def test_copy_kernel_matches_operand_types_without_counting_cpu_or_annotations():
    events = [
        event(
            "aten::copy_", "cpu_op", 0, 100,
            **{
                "External id": 7,
                "Input type": ["float", "c10::BFloat16", "Scalar"],
                "Input Dims": [[1, 200, 32, 128], [1, 200, 32, 128], []],
            },
        ),
        event("copy_kernel", "kernel", 100, 1000, **{"External id": 7}),
        event("GEMM", "kernel", 200, 2000),
        event("copy_kernel", "gpu_user_annotation", 0, 10000),
    ]
    r = analyze(events)
    assert r["cuda_kernel_calls"] == 2
    assert r["summed_kernel_ms"] == 3
    assert r["explicit_copy_groups"] == [
        {
            "types_destination_source": ["float", "c10::BFloat16"],
            "rank": 4,
            "last_dimension": [128],
            "calls": 1,
            "gpu_ms": 1,
        }
    ]
