"""Durable artifacts and explicit dependencies, with no network or model loading."""

from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from gleipnir.adversarial import eval_sets

HERE = Path(__file__).resolve().parent
ROOT = eval_sets.ROOT


def config() -> dict:
    return json.loads((HERE / "config.json").read_text())


def sha(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def digest(value: Any) -> str:
    return sha(json.dumps(value, sort_keys=True, allow_nan=False))


def read(path: Path) -> list[dict]:
    if not path.exists():
        return []
    # A torn final line is an ambiguous receipt, not permission to resend.
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def unique(rows: list[dict]) -> dict[str, dict]:
    out = {}
    for row in rows:
        if row["id"] in out:
            raise ValueError("duplicate artifact id")
        out[row["id"]] = row
    return out


def append(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(value, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def fixed(path: Path, value: Any) -> None:
    """Write once, or verify equality on resume; never replace a different run."""
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise ValueError(f"artifact identity changed: {path.name}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as handle:
        handle.write(json.dumps(value, indent=2, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


@contextmanager
def lock(out: Path):
    out.mkdir(parents=True, exist_ok=True)
    with (out / ".lock").open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def probability(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("score must be numeric")
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError("invalid score")
    return float(value)


def source_identity() -> dict:
    paths = list(HERE.glob("*.py")) + [
        HERE / "config.json",
        HERE / "prompt.txt",
        HERE / "README.md",
    ]
    paths += list((ROOT / "src/gleipnir/adversarial").glob("*.py"))
    paths += [
        ROOT / name
        for name in (
            "experiments/adversarial_a2_cover_comments/run.py",
            "experiments/adversarial_a2_cover_comments/config.json",
            "experiments/adversarial_a4_feedback/config.json",
            "experiments/adversarial_code_or_comments/llm.py",
            "experiments/adversarial_code_or_comments/common.py",
            "experiments/adversarial_crossed_text/campaign.py",
            "experiments/adversarial_crossed_text/common.py",
            "src/gleipnir/__init__.py",
            "src/gleipnir/_compat.py",
            "src/gleipnir/_environment.py",
            "src/gleipnir/evaluation/__init__.py",
            "src/gleipnir/evaluation/metrics.py",
            "pyproject.toml",
            "uv.lock",
        )
    ]
    return {str(p.relative_to(ROOT)): eval_sets.file_sha256(p) for p in paths}
