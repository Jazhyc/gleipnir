"""Adopted default worker with attachable tracing and per-step wall timing."""

import os
import time
from typing import Any

from experiments.b200_host_overhead.tracing import allow_profiler_attach
from experiments.b200_last_layer_trim.worker import LastQueryPrecisionWorker


class HostTimingWorker(LastQueryPrecisionWorker):
    def load_model(self, *, load_dummy_weights: bool = False) -> None:
        allow_profiler_attach()
        self.step_timings: list[dict] = []
        super().load_model(load_dummy_weights=load_dummy_weights)

    def execute_model(self, scheduler_output: Any) -> Any:
        started = time.perf_counter()
        result = super().execute_model(scheduler_output)
        self.step_timings.append(
            {
                "seconds": time.perf_counter() - started,
                "tokens": int(scheduler_output.total_num_scheduled_tokens),
            }
        )
        return result

    def host_timing(self, *, reset: bool = False) -> dict:
        timings = self.step_timings
        if reset:
            self.step_timings = []
        return {"pid": os.getpid(), "steps": timings}
