"""Cooperative operator tracing for a GPU worker without debugger attachment.

Diagnostic synchronization changes scheduling. These logs are never benchmark
measurements; launch-only end records mean enqueued, not GPU-completed.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import os
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import Any


class OperatorTrace:
    """Flush each begin record before calling an operator or synchronizing."""

    def __init__(
        self,
        directory: Path,
        mode: str,
        synchronize: Callable[[], None],
        capturing: Callable[[], bool],
    ) -> None:
        if mode not in {"launch", "sync"}:
            raise ValueError("trace mode must be launch or sync")
        directory.mkdir(parents=True, exist_ok=True)
        self.directory, self.mode = directory, mode
        self.synchronize, self.capturing = synchronize, capturing
        self.events = (directory / "operators.jsonl").open("a", buffering=1)
        self.active: ContextVar[int | None] = ContextVar(
            "operator_trace_batch", default=None
        )
        self.counter = itertools.count(1)
        self.batch_counter = itertools.count(1)

    def record(self, phase: str, **details: Any) -> None:
        self.events.write(
            json.dumps(
                {
                    "phase": phase,
                    "pid": os.getpid(),
                    "time_ns": time.monotonic_ns(),
                    "batch": self.active.get(),
                    **details,
                },
                separators=(",", ":"),
                allow_nan=False,
            )
            + "\n"
        )
        self.events.flush()

    @contextmanager
    def batch(self, **details: Any) -> Iterator[None]:
        token = self.active.set(next(self.batch_counter))
        self.record("batch_begin", **details)
        try:
            yield
        finally:
            self.record("batch_exit")
            self.active.reset(token)

    def invoke(self, label: str, fn: Callable[[], Any], **details: Any) -> Any:
        if self.active.get() is None or self.capturing():
            return fn()
        call = next(self.counter)
        self.record("begin", call=call, operator=label, **details)
        try:
            value = fn()
            if self.mode == "sync":
                self.record("synchronize", call=call, operator=label)
                self.synchronize()
        except BaseException as error:
            self.record(
                "error",
                call=call,
                operator=label,
                error=f"{type(error).__name__}: {error}",
            )
            raise
        self.record("end", call=call, operator=label, gpu_completed=self.mode == "sync")
        return value

    def save_inputs(self, request_ids: list[str], tokens: list[list[int]]) -> Path:
        if (
            len(request_ids) != len(tokens)
            or not tokens
            or any(not row for row in tokens)
        ):
            raise ValueError("batch inputs require one nonempty token row per request")
        data = json.dumps(
            {"request_ids": request_ids, "token_ids": tokens},
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
        digest = hashlib.sha256(data).hexdigest()
        path = self.directory / f"batch_{self.active.get()}_{digest}.json"
        path.write_bytes(data)
        self.record(
            "inputs",
            path=str(path),
            sha256=digest,
            lengths=[len(row) for row in tokens],
            total_tokens=sum(map(len, tokens)),
        )
        return path


class PatchSet:
    """Restore exact original attributes, including inherited-method lookup."""

    def __init__(self) -> None:
        self.entries: list[tuple[Any, str, bool, Any, Any]] = []

    def set(self, owner: Any, name: str, replacement: Any) -> None:
        own = name in vars(owner)
        original = vars(owner).get(name)
        setattr(owner, name, replacement)
        self.entries.append((owner, name, own, original, replacement))

    def restore(self) -> None:
        for owner, name, own, original, replacement in reversed(self.entries):
            if vars(owner).get(name) is not replacement:
                raise RuntimeError(f"trace patch changed while installed: {name}")
            if own:
                setattr(owner, name, original)
            else:
                delattr(owner, name)
        self.entries.clear()
