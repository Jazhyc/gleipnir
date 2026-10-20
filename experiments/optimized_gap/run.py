"""Score the 4B release candidate on OOD through its optimized serving stack."""

from __future__ import annotations

import fcntl

from experiments.optimized_gap.prepare import OPT_4B, OUT
from experiments.teacher_seed_variation.run import score

if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "queue.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        score(OPT_4B)  # Identical re-merge, optimized parity gate, OOD scoring.
