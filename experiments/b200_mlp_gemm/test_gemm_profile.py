"""GEMM attribution must preserve ambiguous shapes and count CUDA kernels once."""

import pytest

from experiments.b200_mlp_gemm.analyze_gemm_profile import analyze, shape_candidates


def module(name, shape, requires_grad=False):
    return {"name": name, "shape": shape, "requires_grad": requires_grad}


def test_shape_attribution_keeps_ambiguity_and_weight_gradients():
    modules = [
        module("x.linear_attn.in_proj_z", [4096, 2560]),
        module("x.self_attn.q_proj.base_layer", [4096, 2560]),
        module("x.mlp.up_proj.lora_A.default", [128, 2560], True),
    ]
    assert shape_candidates([[4096, 2560], [2560, 4096]], modules) == [
        "full_attention_frozen_projections",
        "gdn_frozen_projections",
    ]
    assert shape_candidates([[128, 1000], [1000, 2560]], modules) == ["lora_adapters"]
    assert shape_candidates([[1000, 2560], [2560, 128]], modules) == ["lora_adapters"]
    assert shape_candidates([[1, 2, 3], [1, 3, 4]], modules) == []


def test_external_id_connects_shapes_to_kernel_only_duration():
    events = [
        {
            "ph": "X",
            "cat": "cpu_op",
            "name": "aten::mm",
            "ts": 0,
            "dur": 500,
            "args": {"External id": 42, "Input Dims": [[1000, 2560], [2560, 4096]]},
        },
        {
            "ph": "X",
            "cat": "kernel",
            "name": "nvjet_test",
            "dur": 100,
            "args": {"External id": 42},
        },
        {"ph": "X", "cat": "kernel", "name": "_pack_row_blocks", "dur": 100},
        {"ph": "X", "cat": "gpu_user_annotation", "name": "nvjet_test", "dur": 100},
    ]
    result = analyze(events, [module("x.linear_attn.in_proj_z", [4096, 2560])])
    row = result["groups"]["gdn_frozen_projections"]
    assert row["calls"] == 1 and row["gpu_ms"] == pytest.approx(0.1)
    assert row["fraction_of_total_kernel_time"] == pytest.approx(0.5)
    assert row["fraction_of_ordinary_gemm_time"] == 1.0


@pytest.mark.parametrize(
    ("forward_name", "backward_name"),
    [("CompiledFunction", "CompiledFunctionBackward"), ("aten::mm", "MmBackward0")],
)
def test_attention_call_and_autograd_sequence_disambiguate_backward(
    forward_name, backward_name
):
    modules = [
        module("x.linear_attn.in_proj_z", [4096, 2560]),
        module("x.self_attn.q_proj.base_layer", [4096, 2560]),
    ]

    def event(name, cat, ts, dur, **args):
        return {
            "ph": "X",
            "name": name,
            "cat": cat,
            "ts": ts,
            "dur": dur,
            "pid": 1,
            "tid": 1,
            "args": args,
        }

    events = [
        event("Torch-Compiled Region: 0/7", "user_annotation", 0, 100),
        event(
            forward_name,
            "cpu_op",
            10,
            50,
            **{"External id": 1, "Sequence number": 7},
        ),
        event("ChunkGatedDeltaRuleFunction", "cpu_op", 70, 10, **{"External id": 2}),
        event(
            backward_name,
            "cpu_op",
            200,
            100,
            **{"External id": 3, "Sequence number": 7},
        ),
        event(
            "aten::mm",
            "cpu_op",
            220,
            10,
            **{"External id": 42, "Input Dims": [[1000, 2560], [2560, 4096]]},
        ),
        event("nvjet_test", "kernel", 230, 50, **{"External id": 42}),
    ]
    result = analyze(events, modules)
    assert result["groups"]["gdn_frozen_projections"]["gpu_ms"] == pytest.approx(0.05)
    assert result["decoder_context_evidence"] == {"Torch-Compiled Region: 0/7": ["gdn"]}
