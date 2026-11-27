"""Explicit experimental MXFP8 length bound, without changing kernel arithmetic."""

from __future__ import annotations

import hashlib
from pathlib import Path

LIMIT = 262144


def enable(module) -> None:
    """Change only the checked descriptor/producer bound before compiling plans."""
    if module.MAX_BATCH != 128 or module.MAX_LENGTH != 32768:
        raise ValueError("original MXFP8 envelope changed")
    module.forward_plan.cache_clear()
    module.MAX_LENGTH = LIMIT


def validate(receipt: dict, root: Path) -> None:
    """Require full chunk/history coverage and exact executed-source identities."""
    if not receipt.get("passed") or receipt.get("context_limit") != LIMIT:
        raise ValueError("extended native envelope failed")
    if receipt.get("short_bitwise_cases") != 2:
        raise ValueError("short/batched envelope parity missing")
    coverage = {(c["query_tokens"], c["history_tokens"]) for c in receipt["checks"]}
    required = {(32768, n) for n in (65536, 131072, 262144)} | {(17, 32785)}
    if coverage != required or any(
        not c["finite"] or c["quantized_relative_l2"] > 0.01 for c in receipt["checks"]
    ):
        raise ValueError("extended native chunk/history coverage failed")
    for name, expected in receipt["sources"].items():
        path = Path(name)
        if not path.is_absolute():
            path = root / path
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"extended envelope source changed: {name}")
