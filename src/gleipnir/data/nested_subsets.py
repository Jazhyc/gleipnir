"""Nested proportional source/label subsets with explicit lineage safeguards."""

import hashlib
from collections import defaultdict
from collections.abc import Sequence


def nested_subsets(
    rows: list[dict], counts: Sequence[int], seed: int
) -> dict[int, list[dict]]:
    """Allocate an exact nested count ladder without changing the source mixture.

    This allocator requires singleton recorded lineages. Refuse grouped data
    rather than silently split a conversation across selection boundaries.
    """
    counts = sorted(set(counts))
    if not counts or counts[0] < 1 or counts[-1] > len(rows):
        raise ValueError("subset counts outside population")
    identities = [(r["dataset"], r["index"]) for r in rows]
    lineages = [r["lineage_group"] for r in rows]
    if len(set(identities)) != len(rows) or len(set(lineages)) != len(rows):
        raise ValueError(
            "duplicate identity or non-singleton lineage needs grouped allocation"
        )
    strata = defaultdict(list)
    for row in rows:
        if row["label"] not in (0, 1):
            raise ValueError("non-binary source label")
        strata[(row["source_dataset"], row["label"])].append(row)
    for values in strata.values():
        values.sort(
            key=lambda r: (
                hashlib.sha256(f"{seed}\0{r['lineage_group']}".encode()).hexdigest(),
                str(r["index"]),
            )
        )
    keys = sorted(strata)
    taken = dict.fromkeys(keys, 0)
    selected, result = [], {}
    for position in range(1, counts[-1] + 1):
        # Largest proportional deficit; incremental allocation guarantees nesting.
        key = max(
            (k for k in keys if taken[k] < len(strata[k])),
            key=lambda k: (
                position * len(strata[k]) / len(rows) - taken[k],
                k,
            ),
        )
        selected.append(strata[key][taken[key]])
        taken[key] += 1
        if position in counts:
            result[position] = sorted(
                selected, key=lambda r: (r["dataset"], str(r["index"]))
            )
    return result
