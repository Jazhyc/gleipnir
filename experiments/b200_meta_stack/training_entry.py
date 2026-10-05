"""Profile a single warmup update; measured updates remain unprofiled."""

from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path
from unittest.mock import patch

import torch
from transformers import Trainer, TrainerCallback


class ProfileUpdate(TrainerCallback):
    def __init__(self, update: int, destination: Path) -> None:
        self.update = update
        self.destination = destination
        self.profiler = None

    def on_step_begin(self, args, state, control, **kwargs):
        if state.global_step + 1 == self.update:
            self.profiler = torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                record_shapes=False,
                with_stack=False,
            )
            self.profiler.__enter__()

    def on_step_end(self, args, state, control, **kwargs):
        if self.profiler is not None:
            self.profiler.__exit__(None, None, None)
            self.destination.mkdir(parents=True, exist_ok=False)
            self.profiler.export_chrome_trace(str(self.destination / "trace.json"))
            (self.destination / "operators.txt").write_text(
                self.profiler.key_averages().table(
                    sort_by="self_device_time_total", row_limit=100
                )
            )
            self.profiler = None


def main() -> None:
    update = int(os.environ["GLEIPNIR_PROFILE_UPDATE"])
    destination = Path(os.environ["GLEIPNIR_PROFILE_OUTPUT"])
    original = Trainer.__init__

    def initialize(self, *args, **kwargs):
        kwargs["callbacks"] = [
            *(kwargs.get("callbacks") or []),
            ProfileUpdate(update, destination),
        ]
        original(self, *args, **kwargs)

    sys.argv[0] = "experiments/deception_distillation/train_student_sft.py"
    with patch.object(Trainer, "__init__", initialize):
        runpy.run_path(sys.argv[0], run_name="__main__")


if __name__ == "__main__":
    main()
