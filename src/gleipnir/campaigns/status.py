"""Atomic, thread-safe status for campaigns with concurrent job lanes.

One runner owns each status file. This is an observation record, not a
checkpoint/resume contract or an inter-process scheduling lock.
"""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any


class CampaignStatus:
    """Keep legacy summary fields and per-job timing/failure details together."""

    def __init__(
        self,
        path: Path,
        jobs: list[dict[str, Any]],
        revision: str | None,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.path = path
        self.lock = threading.Lock()
        self.value: dict[str, Any] = {
            "state": "running",
            "phase": "initializing",
            "started_at_unix": time.time(),
            "revision": revision,
            "planned_jobs": [str(job["job_name"]) for job in jobs],
            "active_jobs": [],
            "completed_jobs": [],
            "failed_jobs": [],
            "job_status": {},
            "preflight": "pending",
        }
        if metadata and self.value.keys() & metadata.keys():
            raise ValueError("campaign metadata cannot replace lifecycle fields")
        self.value.update(metadata or {})
        self.update()

    def _write(self) -> None:
        self.value["updated_at_unix"] = time.time()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary.write_text(json.dumps(self.value, indent=2, sort_keys=True) + "\n")
        temporary.replace(self.path)

    def update(self, **values: Any) -> None:
        """Publish a phase or campaign-level transition."""
        with self.lock:
            self.value.update(values)
            self._write()

    def start(self, name: str, *, gpu: int | None = None) -> None:
        """Record a job start; repeated observations preserve its start time."""
        with self.lock:
            if name not in self.value["planned_jobs"]:
                raise ValueError(f"unknown campaign job: {name}")
            if (
                name in self.value["completed_jobs"]
                or name in self.value["failed_jobs"]
            ):
                raise ValueError(f"cannot restart terminal job: {name}")
            if name not in self.value["active_jobs"]:
                self.value["active_jobs"].append(name)
                self.value["job_status"][name] = {
                    "state": "running",
                    "started_at_unix": time.time(),
                    "gpu": gpu,
                }
            self._write()

    def finish(self, name: str) -> None:
        """Mark a successfully executed and validated job complete."""
        with self.lock:
            if name in self.value["completed_jobs"]:
                return
            if name not in self.value["active_jobs"]:
                raise ValueError(f"job is not active: {name}")
            self.value["active_jobs"].remove(name)
            self.value["completed_jobs"].append(name)
            self.value["job_status"][name].update(
                state="complete", completed_at_unix=time.time()
            )
            self._write()

    @contextmanager
    def job(self, name: str, *, gpu: int | None = None) -> Iterator[None]:
        """Publish failures immediately, even while another lane is still running."""
        self.start(name, gpu=gpu)
        try:
            yield
        except BaseException as error:
            with self.lock:
                self.value["active_jobs"].remove(name)
                self.value["failed_jobs"].append(name)
                failed_at = time.time()
                self.value["job_status"][name].update(
                    state="failed", error=repr(error), failed_at_unix=failed_at
                )
                self.value.update(
                    state="failed", error=repr(error), failed_at_unix=failed_at
                )
                self._write()
            raise
        else:
            self.finish(name)
