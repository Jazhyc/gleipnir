"""Score-blind selection for an independent full-workload cache audit."""

import hashlib
import math
from collections import defaultdict
from collections.abc import Sequence
from typing import Any

from gleipnir.prefix_cache import binary_score


def select_cache_audit(references: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Select eight longest-character and eight hash-sampled parents per source."""
    groups = defaultdict(list)
    for row in references:
        groups[row["source"]].append(row)
    if set(groups) != {"stride", "bash_arena", "bash_bench", "gloom"}:
        raise ValueError("audit requires all four prefix sources")
    selected = []
    for source in sorted(groups):
        used = set()
        orders = [
            sorted(groups[source], key=lambda r: (-r["end_character"], r["id"])),
            sorted(
                groups[source],
                key=lambda r: hashlib.sha256(
                    ("prefix-audit-v1\0" + r["id"]).encode()
                ).digest(),
            ),
        ]
        for ordered in orders:
            chosen = []
            for row in ordered:
                if row["parent_prompt_id"] in used:
                    continue
                chosen.append(row)
                used.add(row["parent_prompt_id"])
                if len(chosen) == 8:
                    break
            if len(chosen) != 8:
                raise ValueError("audit requires 16 distinct parents per source")
            selected.extend(chosen)
    return selected


def validate_fresh_audit(
    audit: dict[str, Any],
    references: Sequence[dict[str, Any]],
    cached: dict[str, dict[str, Any]],
    contract_hash: str,
) -> None:
    """Recompute agreement from raw audit logits and exact frozen selection."""
    expected = {r["id"] for r in select_cache_audit(references)}
    rows = audit["rows"]
    if (
        audit["contract_sha256"] != contract_hash
        or audit.get("selection") != "prefix-audit-v1"
        or len(rows) != len(expected)
        or {r["id"] for r in rows} != expected
    ):
        raise ValueError("fresh audit identity or coverage drift")
    errors = []
    for row in rows:
        fresh = binary_score(row["logprob_0"], row["logprob_1"])
        original = cached[row["id"]]
        target = binary_score(original["logprob_0"], original["logprob_1"])
        if not (
            math.isclose(fresh, row["score"], abs_tol=1e-12, rel_tol=0)
            and math.isclose(target, row["cached_score"], abs_tol=1e-12, rel_tol=0)
        ):
            raise ValueError("fresh audit score provenance drift")
        errors.append(abs(fresh - target))
    if max(errors) > 0.05 or sum(errors) / len(errors) > 0.02:
        raise ValueError("fresh audit numerical agreement failed")
