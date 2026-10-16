"""Diagnostic host-overhead audit for measured training updates.

Enabled only by ``GLEIPNIR_HOST_PROFILE_DIR``; the instrumented run's timings
are diagnostics, never speed receipts. Two views:

- ``sync_audit`` counts every synchronizing CUDA call (PyTorch sync-debug
  mode) by issuing source line and by the nearest project/library frame.
- ``profile_update`` traces one update and reports device busy time (union of
  kernel/memcpy/memset intervals) against wall time, plus CUDA API counts.
"""

from __future__ import annotations

import json
import os
import time
import traceback
import warnings
from collections import Counter
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import torch

ENVIRONMENT = "GLEIPNIR_HOST_PROFILE_DIR"
OWNERS = ("gleipnir", "flash_qla", "transformers", "peft", "experiments")
API_CALLS = (
    "cudaStreamSynchronize",
    "cudaDeviceSynchronize",
    "cudaEventSynchronize",
    "cudaMemcpyAsync",
    "cudaLaunchKernel",
    "cuLaunchKernel",
    "cuLaunchKernelEx",
)


def output_dir() -> Path | None:
    value = os.environ.get(ENVIRONMENT)
    return Path(value) if value else None


def _owner(stack: list[traceback.FrameSummary]) -> str:
    for frame in reversed(stack):
        if "/torch/" in frame.filename or "warnings.py" in frame.filename:
            continue
        if any(f"/{name}/" in frame.filename for name in OWNERS):
            return f"{frame.filename.split('/site-packages/')[-1]}:{frame.lineno}"
    return "other"


@contextmanager
def sync_audit() -> Iterator[Counter]:
    """Count synchronizing CUDA operations issued inside the block."""
    counts: Counter = Counter()
    previous_mode = torch.cuda.get_sync_debug_mode()
    previous_show = warnings.showwarning

    def record(message, category, filename, lineno, file=None, line=None):
        if "synchronizing" in str(message):
            stack = traceback.extract_stack()[:-1]
            counts[
                f"{filename.split('/site-packages/')[-1]}:{lineno} <- {_owner(stack)}"
            ] += 1
        else:
            previous_show(message, category, filename, lineno, file, line)

    with warnings.catch_warnings():
        warnings.simplefilter("always")
        warnings.showwarning = record
        torch.cuda.set_sync_debug_mode(1)
        try:
            yield counts
        finally:
            torch.cuda.set_sync_debug_mode(previous_mode)
            warnings.showwarning = previous_show


def interval_union(intervals: list[tuple[float, float]]) -> float:
    total, end = 0.0, None
    for start, stop in sorted(intervals):
        if end is None or start > end:
            total += stop - start
            end = stop
        elif stop > end:
            total += stop - end
            end = stop
    return total


def profile_update(action: Callable[[], Any]) -> tuple[Any, dict]:
    """Run one update under the PyTorch profiler and summarize host overhead."""
    from torch.autograd import DeviceType
    from torch.profiler import ProfilerActivity, profile

    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        start = time.perf_counter()
        result = action()
        torch.cuda.synchronize()
        wall = time.perf_counter() - start
    events = prof.events()
    device = [
        (e.time_range.start, e.time_range.end)
        for e in events
        if e.device_type == DeviceType.CUDA
    ]
    span = (max(b for _, b in device) - min(a for a, _ in device)) / 1e6
    busy = interval_union(device) / 1e6
    api = Counter()
    api_seconds = Counter()
    for e in events:
        if e.device_type == DeviceType.CPU and e.name in API_CALLS:
            api[e.name] += 1
            api_seconds[e.name] += (e.time_range.end - e.time_range.start) / 1e6
    return result, {
        "wall_seconds": wall,
        "device_events": len(device),
        "device_busy_seconds": busy,
        "device_span_seconds": span,
        "device_idle_in_span_seconds": span - busy,
        "device_idle_fraction_of_wall": 1 - busy / wall,
        "cuda_api_calls": dict(api),
        "cuda_api_seconds": dict(api_seconds),
    }


def write(directory: Path, name: str, value: dict) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / name).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
