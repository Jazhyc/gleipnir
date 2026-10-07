"""Resumable, provenance-checked direct binary teacher cache records."""

import json
import math
from pathlib import Path
from typing import Any


def binary_score(logprob_0: float, logprob_1: float) -> float:
    """Normalize only the two explicitly requested decision logprobs."""
    if not all(math.isfinite(x) for x in (logprob_0, logprob_1)):
        raise ValueError("nonfinite binary logprob")
    maximum = max(logprob_0, logprob_1)
    a, b = math.exp(logprob_0 - maximum), math.exp(logprob_1 - maximum)
    return b / (a + b)


def validate_resume(
    path: Path, references: dict[str, dict[str, Any]], contract_hash: str
) -> set[str]:
    """Reject unknown, duplicate, corrupt, or differently prompted cache rows."""
    completed: set[str] = set()
    if not path.exists():
        return completed
    with path.open() as stream:
        for line in stream:
            row = json.loads(line)
            key = row["id"]
            if key in completed or key not in references:
                raise ValueError("duplicate or unknown cached prefix")
            reference = references[key]
            if row["contract_sha256"] != contract_hash:
                raise ValueError("teacher cache contract drift")
            for field in ("prefix_sha256", "rendered_user_prompt_sha256"):
                if row[field] != reference[field]:
                    raise ValueError(f"cached prefix drift: {field}")
            expected = binary_score(row["logprob_0"], row["logprob_1"])
            if not math.isclose(row["score"], expected, abs_tol=1e-12):
                raise ValueError("cached probability disagrees with raw logprobs")
            completed.add(key)
    return completed
