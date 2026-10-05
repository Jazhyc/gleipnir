"""Small stdlib-only control protocol for resident training workers."""

from __future__ import annotations

import json
from pathlib import Path


def write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def validate_request(request: dict) -> None:
    if not isinstance(request.get("id"), str) or not request["id"].isalnum():
        raise ValueError("trial id must be alphanumeric")
    if request.get("variant", "baseline") not in {
        "baseline",
        "gemmprofile",
        "candidate",
    }:
        raise ValueError("unsupported resident variant")
    if request.get("variant") == "candidate" and (
        not isinstance(request.get("source_sha256"), str)
        or len(request["source_sha256"]) != 64
    ):
        raise ValueError("candidate source must be checksum-bound")
