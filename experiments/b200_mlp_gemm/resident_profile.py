"""One bounded CPU/CUDA trace with GEMM operand shapes, without altering kernels."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from transformers import TrainerCallback


class GemmProfile(TrainerCallback):
    def __init__(self, root: Path) -> None:
        self.root, self.profiler = root, None

    def on_step_begin(self, args: Any, state: Any, control: Any, **kw: Any) -> None:
        if state.global_step == 14:
            torch.cuda.synchronize()
            self.profiler = torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                record_shapes=True,
                with_stack=False,
                profile_memory=False,
            )
            self.profiler.__enter__()

    def on_step_end(self, args: Any, state: Any, control: Any, **kw: Any) -> None:
        if self.profiler is None:
            return
        torch.cuda.synchronize()
        self.profiler.__exit__(None, None, None)
        self.profiler.export_chrome_trace(str(self.root / "gemm_trace.json"))
        (self.root / "operators.json").write_text(
            json.dumps(
                [
                    {
                        "name": e.key,
                        "shapes": e.input_shapes,
                        "calls": e.count,
                        "self_cpu_us": e.self_cpu_time_total,
                        "self_device_us": e.self_device_time_total,
                    }
                    for e in self.profiler.key_averages(group_by_input_shape=True)
                ],
                indent=2,
            )
            + "\n"
        )
        self.profiler = None
        print("resident_gemm_profile update=15 exported", flush=True)

    def close(self) -> None:
        if self.profiler is not None:
            self.profiler.__exit__(None, None, None)
            self.profiler = None
