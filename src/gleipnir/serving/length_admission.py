"""Length-bucket admission with FIFO promotion for aged waiting requests."""

from __future__ import annotations

import math
from bisect import bisect_left
from dataclasses import dataclass
from typing import Protocol


class WaitingRequest(Protocol):
    request_id: str
    arrival_time: float
    num_prompt_tokens: int
    num_computed_tokens: int


@dataclass(frozen=True)
class AdmissionPolicy:
    """Admission order only; running work and resource eligibility stay upstream."""

    length_buckets: tuple[int, ...] = (1024, 4096, 16384, 32768)
    max_wait_seconds: float = 0.25
    mode: str = "length_aware"

    def __post_init__(self) -> None:
        if self.mode not in {"length_aware", "fcfs"}:
            raise ValueError("admission mode must be length_aware or fcfs")
        if not math.isfinite(self.max_wait_seconds) or self.max_wait_seconds <= 0:
            raise ValueError("max_wait_seconds must be finite and positive")
        if (
            not self.length_buckets
            or any(type(n) is not int or n <= 0 for n in self.length_buckets)
            or tuple(sorted(set(self.length_buckets))) != self.length_buckets
        ):
            raise ValueError("length_buckets must be strictly increasing positive ints")

    @classmethod
    def from_dict(cls, config: dict) -> AdmissionPolicy:
        """Parse the scheduling part of vLLM's additional configuration."""
        return cls(
            length_buckets=tuple(config["length_buckets"]),
            max_wait_seconds=config["max_wait_seconds"],
            mode=config.get("mode", "length_aware"),
        )

    def key(self, request: WaitingRequest, now: float) -> tuple[int, int, float]:
        """Aged requests precede fresh buckets, in original arrival order.

        Use vLLM's epoch arrival clock. Promotion applies at the next scheduler
        step; it cannot bound execution latency or override resource blocking.
        Stable sorting retains upstream order when arrivals are equal.
        """
        if self.mode == "fcfs" or now - request.arrival_time >= self.max_wait_seconds:
            return (0, 0, request.arrival_time)
        remaining = max(1, request.num_prompt_tokens - request.num_computed_tokens)
        return (1, bisect_left(self.length_buckets, remaining), request.arrival_time)


class LengthAdmissionMixin:
    """Queue-only extension composed with the installed vLLM Scheduler.

    Queues are sorted once per step using one timestamp, so peek/pop agree.
    Blocked requests remain in vLLM's skipped queue and pass its normal checks.
    """

    admission_policy: AdmissionPolicy
    admission_now: float

    def order_waiting(self, now: float) -> None:
        """Reorder both existing deques without moving or duplicating requests."""
        self.admission_now = now
        if self.admission_policy.mode == "fcfs":
            return
        for queue in (self.waiting, self.skipped_waiting):
            ordered = sorted(queue, key=lambda r: self.admission_policy.key(r, now))
            queue.clear()
            queue.extend(ordered)

    def _select_waiting_queue_for_scheduling(self):
        if self.admission_policy.mode == "fcfs":
            return super()._select_waiting_queue_for_scheduling()
        queues = [q for q in (self.skipped_waiting, self.waiting) if q]
        return min(
            queues,
            key=lambda q: self.admission_policy.key(
                q.peek_request(), self.admission_now
            ),
            default=None,
        )
