"""Frozen activation-ranked exclusions and descriptive matched controls."""

from __future__ import annotations

import hashlib
import math
from collections import Counter, defaultdict


def activation_exclusions(rows: list[dict], fraction: float) -> set[str]:
    """Rank harmless originals by raw change, closing exact-content groups."""
    if not 0 < fraction < 1 or len({r["index"] for r in rows}) != len(rows):
        raise ValueError("invalid fraction or duplicate identities")
    eligible = [r for r in rows if r["label"] == 0]
    if not eligible or any(not math.isfinite(r["delta_z20"]) for r in eligible):
        raise ValueError("missing or nonfinite activation changes")
    selected = sorted(eligible, key=lambda r: (-r["delta_z20"], r["index"]))[
        : math.ceil(fraction * len(eligible))
    ]
    hashes = {r["trajectory_sha256"] for r in selected}
    closed = [r for r in rows if r["trajectory_sha256"] in hashes]
    if any(r["label"] != 0 for r in closed):
        raise ValueError("content closure would remove a harmful-labeled record")
    return {r["index"] for r in closed}


def matched_exclusions(
    rows: list[dict], selected: set[str], injected: set[str], seed: int
) -> tuple[set[str], dict]:
    """Randomize within source/label/augmentation/length-quartile strata."""
    buckets = defaultdict(list)
    for row in rows:
        buckets[(row["source"], row["label"])].append(row)
    strata = {}
    pools = defaultdict(list)
    for (source, label), group in buckets.items():
        ordered = sorted(group, key=lambda r: (r["prompt_tokens"], r["index"]))
        for position, row in enumerate(ordered):
            key = (source, label, row["index"] in injected, position * 4 // len(group))
            strata[row["index"]] = key
            pools[key].append(row["index"])
    if not selected <= strata.keys():
        raise ValueError("unknown exclusion identity")
    counts = Counter(strata[index] for index in selected)
    control = set()
    for key, count in sorted(counts.items()):
        candidates = sorted(
            pools[key],
            key=lambda index: hashlib.sha256(f"{seed}:{index}".encode()).hexdigest(),
        )
        control.update(candidates[:count])
    if len(control) != len(selected):
        raise ValueError("matched removal count drift")
    hashes = {r["trajectory_sha256"] for r in rows if r["index"] in control}
    closed = {r["index"] for r in rows if r["trajectory_sha256"] in hashes}
    if closed != control:
        raise ValueError("random control violates exact-content closure")
    return control, {
        "stratum_counts": {str(key): count for key, count in sorted(counts.items())},
        "activation_control_overlap": len(selected & control),
        "overlap_permitted": True,
        "selection": (
            "SHA256(seed:index) rank within fixed strata; "
            "no teacher/activation score use"
        ),
    }
