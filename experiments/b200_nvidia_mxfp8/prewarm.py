"""Prepare exact NVIDIA plans in independent processes without training a model."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from experiments.b200_nvidia_mxfp8.run import ROOT, environment
from gleipnir.monitoring_systems_screen import sha256_file


def selected_lengths(path: Path, expected_sha256: str) -> list[int]:
    """Require the frozen manifest and collect its recorded direct-token lengths."""
    if sha256_file(path) != expected_sha256:
        raise ValueError("prewarm selection checksum drift")
    rows = [json.loads(line) for line in path.read_text().splitlines() if line]
    lengths = [row["student_direct_tokens"] for row in rows]
    if not lengths or any(type(n) is not int or n < 1 for n in lengths):
        raise ValueError("invalid prewarm sequence lengths")
    # Singleton attention is analytical and has no cuDNN plan.
    return sorted({n for n in lengths if n > 1}, reverse=True)


def partitions(lengths: list[int], workers: int) -> list[list[int]]:
    """Distribute each unique length exactly once, balancing long and short shapes."""
    if not 1 <= workers <= 16:
        raise ValueError("prewarm requires one to sixteen workers")
    return [lengths[index::workers] for index in range(workers)]


def worker(task_path: Path) -> None:
    """Build both selected plans; do not execute attention or touch model weights."""
    import torch
    from cudnn.frost.compiled_cache import stats

    from gleipnir.nvidia_mxfp8_attention import _plan

    torch.set_num_threads(1)
    task = json.loads(task_path.read_text())
    receipt = task_path.with_suffix(".jsonl")
    with receipt.open("x") as handle:
        for length in task["lengths"]:
            for backward in (False, True):
                before = dict(stats())
                started = time.perf_counter()
                _plan(length, 16, 4, 0.0625, backward, 0)
                row = {
                    "length": length,
                    "backward": backward,
                    "seconds": time.perf_counter() - started,
                    "cache_before": before,
                    "cache_after": dict(stats()),
                }
                handle.write(json.dumps(row) + "\n")
                handle.flush()
                print(json.dumps(row), flush=True)
            _plan.cache_clear()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--screen", type=Path)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--worker-task", type=Path)
    args = parser.parse_args()
    if args.worker_task is not None:
        worker(args.worker_task)
        return
    if args.screen is None:
        parser.error("--screen is required for the prewarm coordinator")
    screen_path = args.screen.resolve()
    screen = json.loads(screen_path.read_text())
    source = screen["source_job"]
    lengths = selected_lengths(
        Path(source["selection_manifest"]), source["selection_sha256"]
    )
    tasks = partitions(lengths, args.workers)
    output = screen_path.parent / "prewarm"
    output.mkdir(exist_ok=False)
    env = environment(screen["config"])
    report = {
        "status": "starting",
        "screen_sha256_at_start": sha256_file(screen_path),
        "source_sha256": sha256_file(Path(__file__)),
        "selection_sha256": source["selection_sha256"],
        "lengths": lengths,
        "workers": args.workers,
        "contract": {
            "query_heads": 16,
            "kv_heads": 4,
            "dimension": 256,
            "scale": 0.0625,
            "device": 0,
        },
        "scope": "plan preparation only; no attention execution or model updates",
        "cache_paths": {k: v for k, v in env.items() if "CACHE" in k},
    }
    report_path = output / "summary.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    (output / "executed_prewarm.py").write_bytes(Path(__file__).read_bytes())
    processes = []
    handles = []
    started = time.perf_counter()
    try:
        for index, lengths_for_worker in enumerate(tasks):
            task_path = output / f"worker{index}.json"
            task_path.write_text(json.dumps({"lengths": lengths_for_worker}) + "\n")
            handle = (output / f"worker{index}.log").open("x")
            handles.append(handle)
            processes.append(
                subprocess.Popen(
                    [
                        sys.executable,
                        "-m",
                        "experiments.b200_nvidia_mxfp8.prewarm",
                        "--worker-task",
                        str(task_path),
                    ],
                    cwd=ROOT,
                    env=env,
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                )
            )
        report["status"] = "running"
        report["worker_pids"] = [p.pid for p in processes]
        report_path.write_text(json.dumps(report, indent=2) + "\n")
        codes = [process.wait() for process in processes]
        report.update(
            status="complete" if all(c == 0 for c in codes) else "failed",
            returncodes=codes,
            seconds=time.perf_counter() - started,
        )
        report_path.write_text(json.dumps(report, indent=2) + "\n")
        if any(codes):
            raise SystemExit(
                "prewarm worker failed; original screen remains independent"
            )
    finally:
        for handle in handles:
            handle.close()


if __name__ == "__main__":
    main()
