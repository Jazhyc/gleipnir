"""Experimental MXFP8 length bound up to 1M tokens; kernel arithmetic unchanged."""

from __future__ import annotations

import hashlib
from pathlib import Path

LIMIT = 1048576
CHUNK = 32768
HISTORIES = (65536, 131072, 262144, 524288, 1048576)
IRREGULAR = (17, CHUNK + 17)


def enable(module, limit: int = LIMIT) -> None:
    """Change only the checked descriptor/producer bound before compiling plans."""
    if module.MAX_BATCH != 128 or module.MAX_LENGTH not in {CHUNK, limit}:
        raise ValueError("original MXFP8 envelope changed")
    if not CHUNK <= limit <= LIMIT:
        raise ValueError("requested MXFP8 bound is outside the validated envelope")
    module.forward_plan.cache_clear()
    module.MAX_LENGTH = limit


def validate(receipt: dict, root: Path, limit: int) -> None:
    """Require full-chunk coverage through `limit` and exact executed sources."""
    if not receipt.get("passed") or receipt.get("context_limit") != LIMIT:
        raise ValueError("extended native envelope failed")
    if receipt.get("short_bitwise_cases") != 2:
        raise ValueError("short/batched envelope parity missing")
    coverage = {(c["query_tokens"], c["history_tokens"]) for c in receipt["checks"]}
    required = {(CHUNK, n) for n in HISTORIES} | {IRREGULAR}
    if coverage != required or any(
        not c["finite"] or c["quantized_relative_l2"] > 0.01 for c in receipt["checks"]
    ):
        raise ValueError("extended native chunk/history coverage failed")
    if limit > max(HISTORIES):
        raise ValueError("serving limit exceeds validated histories")
    for name, expected in receipt["sources"].items():
        path = Path(name)
        if not path.is_absolute():
            path = root / path
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"extended envelope source changed: {name}")
