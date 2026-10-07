"""Bounded recovery of resumable evaluators that stop saving predictions.

Own only a freshly created subprocess session; never attach to arbitrary GPU
PIDs. Nonzero exits and invalid caches fail closed rather than being retried.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class RecoveryPolicy:
    startup_seconds: float = 900
    stall_seconds: float = 600
    poll_seconds: float = 30
    terminate_seconds: float = 15
    max_restarts: int = 2

    def __post_init__(self) -> None:
        if (
            min(
                self.startup_seconds,
                self.stall_seconds,
                self.poll_seconds,
                self.terminate_seconds,
            )
            <= 0
            or self.max_restarts < 0
        ):
            raise ValueError("invalid evaluation recovery policy")


def prediction_counts(paths: list[Path]) -> tuple[int, ...]:
    """Count complete records only; log chatter and file touching are not progress.

    The evaluator atomically replaces its prediction files. Invalid complete
    JSON records fail closed; a trailing partial record is not counted and the
    evaluator's strict resume validator must reject it on restart.
    """
    counts = []
    for path in paths:
        count = 0
        if path.exists():
            with path.open() as stream:
                for line in stream:
                    if not line.endswith("\n"):
                        break
                    json.loads(line)
                    count += 1
        counts.append(count)
    return tuple(counts)


def live_group(pgid: int) -> bool:
    """Ignore dead zombies while verifying no owned worker remains runnable."""
    for path in Path("/proc").glob("[0-9]*/stat"):
        try:
            fields = path.read_text().rsplit(")", 1)[1].split()
            if int(fields[2]) == pgid and fields[0] != "Z":
                return True
        except (FileNotFoundError, ProcessLookupError):
            continue
    return False


def stop_group(process: subprocess.Popen, grace: float) -> None:
    """TERM then KILL the isolated session, including orphaned vLLM workers."""
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(process.pid, sig)
        except ProcessLookupError:
            break
        deadline = time.monotonic() + grace
        while live_group(process.pid) and time.monotonic() < deadline:
            process.poll()
            time.sleep(min(0.1, grace))
        if not live_group(process.pid):
            break
    process.wait(timeout=grace)
    if live_group(process.pid):
        raise RuntimeError(
            "owned evaluation workers survived cleanup; refusing restart"
        )


@contextmanager
def termination_cleanup() -> Iterator[None]:
    """Turn supervisor termination into stack unwinding and worker cleanup."""

    def terminate(signum: int, frame: object) -> None:
        raise SystemExit(128 + signum)

    previous = signal.signal(signal.SIGTERM, terminate)
    try:
        yield
    finally:
        signal.signal(signal.SIGTERM, previous)


@termination_cleanup()
def run_resumable_evaluation(
    command: list[str],
    paths: list[Path],
    events_path: Path,
    *,
    policy: RecoveryPolicy | None = None,
) -> None:
    """Restart only no-progress hangs, retaining all output and child diagnostics."""
    policy = policy or RecoveryPolicy()
    if not paths or len(set(paths)) != len(paths):
        raise ValueError("explicit unique prediction paths are required")
    events_path.parent.mkdir(parents=True, exist_ok=True)

    def record(event: str, **details: object) -> None:
        row = {"event": event, "time_unix": time.time(), **details}
        with events_path.open("a") as log:
            log.write(json.dumps(row, sort_keys=True) + "\n")
            log.flush()
            os.fsync(log.fileno())
        print(f"evaluation_watchdog: {json.dumps(row, sort_keys=True)}", flush=True)

    record("start", policy=asdict(policy), paths=[str(p) for p in paths])
    for attempt in range(policy.max_restarts + 1):
        counts = prediction_counts(paths)
        advanced = False
        last_progress = time.monotonic()
        process = subprocess.Popen(command, start_new_session=True)
        record("attempt", attempt=attempt, pid=process.pid, saved_counts=counts)
        try:
            while True:
                code = process.poll()
                if code is not None:
                    if code:
                        record("failed_exit", attempt=attempt, returncode=code)
                        raise subprocess.CalledProcessError(code, command)
                    final_counts = prediction_counts(paths)
                    if any(
                        now < before
                        for now, before in zip(final_counts, counts, strict=True)
                    ):
                        raise ValueError("prediction count decreased at evaluator exit")
                    record(
                        "complete",
                        attempt=attempt,
                        saved_counts=final_counts,
                    )
                    return
                current = prediction_counts(paths)
                if any(
                    now < before for now, before in zip(current, counts, strict=True)
                ):
                    raise ValueError(
                        "prediction count decreased; refusing automatic recovery"
                    )
                if current != counts:
                    counts, advanced, last_progress = current, True, time.monotonic()
                    record("progress", attempt=attempt, saved_counts=counts)
                timeout = policy.stall_seconds if advanced else policy.startup_seconds
                if time.monotonic() - last_progress >= timeout:
                    record(
                        "stall",
                        attempt=attempt,
                        pid=process.pid,
                        saved_counts=counts,
                        no_progress_seconds=timeout,
                    )
                    break
                time.sleep(policy.poll_seconds)
        finally:
            stop_group(process, policy.terminate_seconds)
        if attempt == policy.max_restarts:
            record("retry_budget_exhausted", attempt=attempt)
            raise TimeoutError("evaluation stalled after bounded automatic retries")
        record("restart", next_attempt=attempt + 1, saved_counts=counts)
