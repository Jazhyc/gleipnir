"""Opt-in prefix caching for the selected serving default (not for benchmarks).

Use it for synchronous monitoring of growing sessions: the replay measured
8-13x less time per session, but cached scores differ from uncached scores by
up to about 0.15. Keep it off for evaluation and benchmarking, whose repeated
passes would otherwise hit the cache; see README.md in this directory.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_WORKER = "experiments.b200_gdn_fast.worker.FastGdnWorker"
CACHED_WORKER = "experiments.b200_prefix_replay.worker.PrefixCachedFastGdnWorker"
WORKER_SOURCE = "experiments/b200_prefix_replay/worker.py"
# User decision 2026-10-16: accept cached-scoring drift for opt-in synchronous
# monitoring (docs/decisions/prefix_cached_serving.md). Same limits as the
# campaigns' new-adapter serving parity; the default stack keeps 0.005.
CANARY_LIMITS = {"score_mae": 0.02, "correlation": 0.99}


def prefix_cached_command(command: list[str]) -> list[str]:
    """Change only the prefix-cache flags and the worker subclass."""
    command = command.copy()
    if (
        "--no-enable-prefix-caching" not in command
        or "--enable-prefix-caching" in command
    ):
        raise ValueError("expected the selected no-prefix-caching recipe")
    worker = command.index("--worker-cls") + 1
    if command[worker] != DEFAULT_WORKER:
        raise ValueError("expected the selected default worker")
    command[command.index("--no-enable-prefix-caching")] = "--enable-prefix-caching"
    command[worker] = CACHED_WORKER
    index = command.index("--additional-config") + 1
    additional = json.loads(command[index])
    additional["gleipnir_frost_fp4"][WORKER_SOURCE] = hashlib.sha256(
        (ROOT / WORKER_SOURCE).read_bytes()
    ).hexdigest()
    command[index] = json.dumps(additional, sort_keys=True)
    return [*command, "--mamba-cache-mode", "align"]
