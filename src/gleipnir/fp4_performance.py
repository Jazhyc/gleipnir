"""Bounded profiling of a warmed precision-training workload."""

from __future__ import annotations

import json
import time
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch


def profile_backward(action: Callable[[], dict[str, Any]], output: Path) -> dict:
    """Trace one warmed forward/backward/clip without advancing the optimizer."""
    torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=False,
        profile_memory=False,
        with_stack=False,
    ) as profiler:
        with torch.profiler.record_function("fp4_profile_forward_backward_clip"):
            measurement = action()
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    profiler.export_chrome_trace(str(output / "profile_trace.json"))
    operators = [
        {
            "name": event.key,
            "calls": event.count,
            "self_cpu_us": event.self_cpu_time_total,
            "self_device_us": event.self_device_time_total,
            "device_us": event.device_time_total,
        }
        for event in profiler.key_averages()
    ]
    kernels: dict[str, dict] = defaultdict(lambda: {"calls": 0, "device_us": 0.0})
    for event in profiler.events():
        if event.device_type == torch.autograd.DeviceType.CUDA:
            kernels[event.name]["calls"] += 1
            kernels[event.name]["device_us"] += event.device_time_total
    report = {
        "scope": (
            "one warmed logical batch; forward/backward/finite checks/clip; "
            "no optimizer update; profiler overhead included"
        ),
        "wall_seconds_with_profiler": elapsed,
        "measurement": measurement,
        "operators": sorted(operators, key=lambda item: -item["self_device_us"]),
        "kernels": sorted(
            [{"name": name, **value} for name, value in kernels.items()],
            key=lambda item: -item["device_us"],
        ),
    }
    (output / "profile.json").write_text(json.dumps(report, indent=2) + "\n")
    return {
        key: value
        for key, value in report.items()
        if key not in {"operators", "kernels"}
    }
