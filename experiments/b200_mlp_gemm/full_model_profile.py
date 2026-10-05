"""Profile fixed warmed optimizer updates with checksum-bound validation reuse."""

from __future__ import annotations

import hashlib
import json
import os
import time
from copy import deepcopy
from pathlib import Path
from typing import Any

import torch
from transformers import TrainerCallback

from gleipnir.packed_benchmark import summarize
from gleipnir.validated_startup import validation_reference as bf16_validation_reference

ROOT = Path(__file__).resolve().parents[2]
REFERENCE = (
    ROOT / "results/b200_mlp_gemm/warmed03/causal_adapter/training_metadata.json"
)
REFERENCE_SHA = "14ab15279bb8895cf32353117d5c1cf957ad45d2b0ca9d7205067db27d77edeb"
UPDATES = (11, 15, 20)


def profile_validation_reference(path: Path, **kwargs: Any) -> dict:
    """Reuse failed numerical receipts only in the explicit bounded FP4 profile."""
    if path.resolve() != REFERENCE or kwargs.get("expected_sha256") != REFERENCE_SHA:
        raise ValueError("FP4 profile requires its checksum-bound warmed receipt")
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != REFERENCE_SHA:
        raise ValueError("FP4 profile validation checksum drift")
    metadata = json.loads(raw)
    validated = validate_profile_reference(metadata)
    if (
        kwargs.get("packed_attention_backend") != "flash_attention_4"
        or kwargs.get("packed_attention_version") != "4.0.0b33"
        or kwargs.get("learning_gradient_tolerance") != 0.05
    ):
        raise ValueError("FP4 profile precision/acceptance identity changed")
    # Check the unchanged B200/FA4/FlashQLA runtime with the existing verifier.
    # Its BF16 numerical receipts do not become the FP4 validation reference.
    control = bf16_validation_reference(
        ROOT
        / "results/b200_bf16_fa4_accepted/flash_attention_4"
        / "causal_adapter/training_metadata.json",
        expected_sha256="185fa498f8ec31f07ae58a8213584c7a7f5b41738393ae75d97f91229a993364",
        verify_runtime=True,
    )
    launch = json.loads((REFERENCE.parents[1] / "summary.json").read_text())
    for name in (
        "cudnn_fp4_mlp.py",
        "cudnn_fp4_gemm.py",
        "cudnn_fp4_epilogue.py",
        "nvfp4_pack.py",
        "attention_backends.py",
    ):
        relative = f"src/gleipnir/{name}"
        if (
            hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
            != launch["source_sha256"][relative]
        ):
            raise ValueError(f"validated kernel source changed: {relative}")
    result = {
        **control,
        "policy": (
            "reuse_explicit_timing_only_fp4_resident"
            if os.environ.get("GLEIPNIR_FP4_RESIDENT_ROOT")
            else "reuse_explicit_timing_only_fp4_profile"
        ),
        "reference_path": str(path),
        "reference_sha256": REFERENCE_SHA,
        "learning_gradient_tolerance": 0.05,
        "reference_backend": metadata["gated_delta_backend"],
        "reference_adaptive_canary": metadata["adaptive_microbatching"][
            "gradient_canary"
        ],
        "reference_compile_canary": metadata["selective_torch_compile"]["canary"],
        "reference_packing": {
            k: deepcopy(metadata["sequence_packing"][k])
            for k in ("eager_canary", "compiled_canary", "preflight")
        },
        "timing_only": True,
        "reference_timing_authority": metadata["sequence_packing"]["timing_authority"],
        "initial_master_sha256": validated["initial_master_sha256"],
    }
    return result


def validate_profile_reference(metadata: dict) -> dict:
    """Reject unvalidated execution, changed arithmetic, or missing isolation."""
    candidate = summarize(metadata, 10, accept_timing=True)
    native = metadata["quantization"]["full_bf16_lora"]["native_fp4_mlp"]
    if (
        not native.get("hardware_packing")
        or not native.get("fused_descale")
        or metadata["selective_torch_compile"].get("mode") != "default"
        or metadata["sequence_packing"]["learning_gradient_tolerance"] != 0.05
    ):
        raise ValueError("reference does not validate the current FP4 profile recipe")
    from gleipnir.packed_training_screen import _accept_canary

    for name in ("eager_canary", "compiled_canary"):
        _accept_canary(
            deepcopy(metadata["sequence_packing"][name]),
            0.05,
            metadata["sequence_packing"]["timing_authority"],
        )
    return candidate


class ProfileUpdates(TrainerCallback):
    """Record complete forward/backward/finite-check/clip/optimizer work."""

    def __init__(self, destination: Path) -> None:
        self.destination = destination
        self.profiler = None
        self.steps: list[dict] = []
        self.started = 0.0
        self.range = None

    def on_step_begin(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
        update = state.global_step + 1
        if update in UPDATES:
            self.destination.mkdir(parents=True, exist_ok=True)
            torch.cuda.synchronize()
            self.started = time.perf_counter()
            self.profiler = torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                record_shapes=False,
                with_stack=False,
                profile_memory=False,
            )
            self.profiler.__enter__()
            self.range = torch.profiler.record_function(
                f"warmed_optimizer_update_{update}"
            )
            self.range.__enter__()

    def on_step_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
        if self.profiler is None:
            return
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - self.started
        self.range.__exit__(None, None, None)
        self.range = None
        self.profiler.__exit__(None, None, None)
        output = self.destination / f"update{state.global_step}"
        output.mkdir(exist_ok=False)
        self.profiler.export_chrome_trace(str(output / "trace.json"))
        operators = [
            {
                "name": e.key,
                "calls": e.count,
                "self_cpu_us": e.self_cpu_time_total,
                "self_device_us": e.self_device_time_total,
                "device_us": e.device_time_total,
            }
            for e in self.profiler.key_averages()
        ]
        (output / "operators.json").write_text(json.dumps(operators, indent=2) + "\n")
        self.steps.append(
            {"update": state.global_step, "wall_seconds_with_profiler": elapsed}
        )
        (self.destination / "profile.json").write_text(
            json.dumps(
                {
                    "updates": self.steps,
                    "selected_updates": list(UPDATES),
                    "instrumented_time_is_not_speed_result": True,
                    "scope": (
                        "complete warmed logical update, including optimizer; "
                        "no baseline trace"
                    ),
                },
                indent=2,
            )
            + "\n"
        )
        self.profiler = None
        print(f"warmed_profile update={state.global_step} exported", flush=True)

    def close(self) -> None:
        if self.range is not None:
            self.range.__exit__(None, None, None)
            self.range = None
        if self.profiler is not None:
            self.profiler.__exit__(None, None, None)
            self.profiler = None
