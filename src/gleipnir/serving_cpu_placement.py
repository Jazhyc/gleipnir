"""Reversible CPU affinity for identity-checked resident Linux processes."""

from __future__ import annotations

import os
import re
from pathlib import Path


def parse_cpu_list(value: str) -> set[int]:
    """Parse Linux sysfs CPU lists without accepting malformed ranges."""
    result = set()
    for part in value.strip().split(","):
        if not re.fullmatch(r"\d+(?:-\d+)?", part):
            raise ValueError("invalid CPU list")
        bounds = [int(v) for v in part.split("-")]
        first, last = bounds[0], bounds[-1]
        if last < first:
            raise ValueError("reversed CPU range")
        result.update(range(first, last + 1))
    return result


def start_ticks(path: Path) -> int:
    """Read process birth time even when the comm field contains spaces."""
    return int(path.read_text().rsplit(")", 1)[1].split()[19])


class ProcessAffinity:
    """Change every thread; restore saved masks and reject recycled PIDs."""

    def __init__(self, pid: int, *, proc: Path = Path("/proc")) -> None:
        self.pid = pid
        self.path = proc / str(pid)
        self.birth = start_ticks(self.path / "stat")
        self.original: dict[int, tuple[int, set[int]]] = {}
        self.leader_original = set(os.sched_getaffinity(pid))
        for task in (self.path / "task").iterdir():
            try:
                tid = int(task.name)
                self.original[tid] = (
                    start_ticks(task / "stat"),
                    set(os.sched_getaffinity(tid)),
                )
            except FileNotFoundError:
                continue

    def check(self) -> None:
        if start_ticks(self.path / "stat") != self.birth:
            raise ValueError("process identity changed; refuse affinity mutation")

    def apply(self, cpus: set[int] | None) -> dict[str, list[int]]:
        """Apply a bounded mask, or restore originals; verify all live threads."""
        self.check()
        if cpus is not None and (not cpus or not cpus <= self.leader_original):
            raise ValueError("CPU mask must be a nonempty subset of original affinity")
        for _ in range(4):
            for task in (self.path / "task").iterdir():
                try:
                    self.check()
                    tid, birth = int(task.name), start_ticks(task / "stat")
                    original = self.original.get(tid)
                    restore = (
                        original[1]
                        if original and original[0] == birth
                        else self.leader_original
                    )
                    os.sched_setaffinity(tid, restore if cpus is None else cpus)
                except ProcessLookupError:
                    continue
                except FileNotFoundError:
                    self.check()
            self.check()
            masks = {}
            for task in (self.path / "task").iterdir():
                try:
                    tid, birth = int(task.name), start_ticks(task / "stat")
                    original = self.original.get(tid)
                    expected = (
                        cpus
                        if cpus is not None
                        else (
                            original[1]
                            if original and original[0] == birth
                            else self.leader_original
                        )
                    )
                    actual = set(os.sched_getaffinity(tid))
                    masks[str(tid)] = sorted(actual)
                    if actual != expected:
                        break
                except (FileNotFoundError, ProcessLookupError):
                    continue
            else:
                return masks
        raise RuntimeError("thread affinity did not converge")
