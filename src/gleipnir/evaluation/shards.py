"""Deterministic batch-preserving evaluation shards and strict result merging."""

import math


def partition_pending(
    rows: list[dict],
    saved: list[dict],
    config_hash: str,
    count: int,
    index: int,
    batch_rows: int = 128,
) -> list[dict]:
    """Skip verified saved identities and assign whole original batches to workers."""
    if count < 1 or not 0 <= index < count or batch_rows < 1:
        raise ValueError("invalid shard parameters")
    allowed = {str(r["id"]) for r in rows}
    if len(allowed) != len(rows):
        raise ValueError("duplicate input identities")
    completed = set()
    for r in saved:
        identity = str(r["id"])
        if identity not in allowed or identity in completed:
            raise ValueError("saved identity drift")
        if r["config_sha256"] != config_hash or not math.isfinite(float(r["score"])):
            raise ValueError("saved provenance or score drift")
        completed.add(identity)
    pending = [r for r in rows if str(r["id"]) not in completed]
    return [r for i, r in enumerate(pending) if (i // batch_rows) % count == index]


def merge_predictions(
    rows: list[dict], parts: list[list[dict]], config_hash: str
) -> list[dict]:
    """Require exact coverage and no duplicate rows, even identical duplicates."""
    combined = [r for part in parts for r in part]
    remaining = partition_pending(rows, combined, config_hash, 1, 0)
    if remaining:
        raise ValueError("incomplete merged evaluation")
    by_id = {str(r["id"]): r for r in combined}
    return [by_id[str(r["id"])] for r in rows]
