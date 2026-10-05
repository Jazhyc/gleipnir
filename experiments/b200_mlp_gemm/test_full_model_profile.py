"""Profiler reuse cannot silently accept drift or promote instrumented timings."""

from copy import deepcopy
from types import SimpleNamespace

import pytest

from experiments.b200_mlp_gemm.full_model_profile import (
    ProfileUpdates,
    validate_profile_reference,
)
from experiments.b200_mlp_gemm.test_timing_screen import metadata, student_config
from gleipnir.packed_training import validate_packed_training_config


def test_only_scoped_profile_can_reuse_timing_receipt():
    config = student_config()
    config["training"]["startup_validation_reference"] = "receipt.json"
    config["training"]["startup_validation_reference_sha256"] = "bound-receipt"
    config["training"]["packing_learning_gradient_tolerance"] = 0.05
    with pytest.raises(ValueError, match="timing-only"):
        validate_packed_training_config(config)
    config["training"]["native_fp4_mlp_profile"] = True
    assert validate_packed_training_config(config)
    config["training"]["native_fp4_mlp_profile"] = "true"
    with pytest.raises(ValueError, match="timing-only"):
        validate_packed_training_config(config)


@pytest.mark.parametrize(
    "change", ["mode", "hardware", "epilogue", "leakage", "finite", "authority"]
)
def test_profile_reference_rejects_drift(change):
    original = metadata()
    original["selective_torch_compile"]["mode"] = "default"
    original["sequence_packing"]["learning_gradient_tolerance"] = 0.05
    original["quantization"]["full_bf16_lora"]["native_fp4_mlp"].update(
        hardware_packing=True,
        fused_descale=True,
    )
    for key in ("eager_canary", "compiled_canary"):
        original["sequence_packing"][key].update(
            independent_loss=1.11,
            packed_loss=1.06,
            cases=[
                {
                    "repeat_max_abs": 0.0,
                    "perturb_max_abs": 0.0,
                    "cross_input_grad_max_abs": 0.0,
                    "own_input_grad_max_abs": 1.0,
                }
            ],
        )
    validate_profile_reference(original)
    changed = deepcopy(original)
    native = changed["quantization"]["full_bf16_lora"]["native_fp4_mlp"]
    if change == "mode":
        changed["selective_torch_compile"]["mode"] = "reduce-overhead"
    elif change == "hardware":
        native["hardware_packing"] = False
    elif change == "epilogue":
        native["fused_descale"] = False
    elif change == "leakage":
        changed["sequence_packing"]["eager_canary"]["cases"][0][
            "cross_input_grad_max_abs"
        ] = 1.0
    elif change == "finite":
        changed["sequence_packing"]["eager_canary"]["adapter_gradient_relative_l2"] = (
            float("nan")
        )
    else:
        changed["sequence_packing"]["timing_authority"] = None
    with pytest.raises(ValueError):
        validate_profile_reference(changed)


def test_profiler_callback_selects_only_fixed_updates(monkeypatch, tmp_path):
    import torch

    from experiments.b200_mlp_gemm.full_model_profile import UPDATES

    entered = []

    class FakeContext:
        def __enter__(self):
            entered.append(1)

        def __exit__(self, *args):
            pass

    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(torch.profiler, "profile", lambda **kwargs: FakeContext())
    monkeypatch.setattr(torch.profiler, "record_function", lambda name: FakeContext())
    callback = ProfileUpdates(tmp_path)
    for update in range(1, 21):
        callback.on_step_begin(None, SimpleNamespace(global_step=update - 1), None)
        if update in UPDATES:
            assert callback.profiler is not None
            callback.close()
        else:
            assert callback.profiler is None
    assert len(entered) == 2 * len(UPDATES)


def test_kernel_accounting_excludes_annotations_and_merges_overlap():
    from experiments.b200_mlp_gemm.analyze_full_profile import analyze, kernel_group

    def event(name, category, start, duration):
        return {"name": name, "cat": category, "ph": "X", "ts": start, "dur": duration}

    rows = [
        event("cudnn_kernel_frost_Float4E2M1FN_block_scale_matmul", "kernel", 0, 100),
        event("nvjet_sm100_tst", "kernel", 50, 100),
        event("tilelang_fused_chunk_gdr_bwd_kernel", "kernel", 300, 80),
        event("Memcpy HtoD", "gpu_memcpy", 160, 50),
        event("duplicated FP4 kernel label", "gpu_user_annotation", 0, 100),
        event("at::mm", "cpu_op", 0, 300),
        event("warmed_optimizer_update_11", "user_annotation", 0, 500),
        event("cudaLaunchKernel", "cuda_runtime", 10, 40),
    ]
    result = analyze(rows)
    assert result["kernel_count"] == 3
    assert result["summed_kernel_ms"] == pytest.approx(0.28)
    assert result["kernel_interval_union_ms"] == pytest.approx(0.23)
    assert result["device_interval_union_ms"] == pytest.approx(0.28)
    assert result["no_device_event_ms_within_span"] == pytest.approx(0.1)
    assert result["profiled_cpu_update_range_ms"] == pytest.approx(0.5)
    assert result["cuda_apis"]["cudaLaunchKernel"]["cpu_ms"] == pytest.approx(0.04)
    assert kernel_group("_pack_row_blocks") == "mlp_fp4_dynamic_conversion"
    assert kernel_group("FlashAttentionForwardSm100") == "fa4_full_attention"
    assert kernel_group("opaque_kernel") == "other_or_unclassified"
