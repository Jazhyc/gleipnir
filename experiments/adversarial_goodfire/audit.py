"""Prepare blinded coding packets and validate two-reader action-only labels.

No model is invoked. Human/independent-reader records belong in ignored
`data/goodfire/adjudications.jsonl`, never in the source tree.
"""

from __future__ import annotations

import argparse

from experiments.adversarial_goodfire.common import (
    DATA,
    read_jsonl,
    sha,
    write_json,
    write_jsonl,
)
from gleipnir.adversarial.contract import split_prompt

DECISIONS = {"visible_misconduct", "benign_or_ambiguous"}


def adjudicate(selection: list[dict], records: list[dict]) -> list[dict]:
    """Fail closed on missing/stale/disagreeing decisions; never create negatives."""
    candidates = {
        r["id"]: r
        for r in selection
        if r.get("primary_eligibility") == "pending_two_reader_adjudication"
    }
    by_id = {key: {} for key in candidates}
    for r in records:
        if r["id"] not in candidates:
            raise ValueError("unknown adjudication candidate")
        expected = candidates[r["id"]].get("trajectory_sha256")
        if not expected or r["trajectory_sha256"] != expected:
            raise ValueError("stale adjudication trajectory hash")
        reader = r.get("reader", "").strip()
        if (
            not reader
            or not r.get("reason", "").strip()
            or r.get("decision") not in DECISIONS
        ):
            raise ValueError("reader, reason and valid decision required")
        if reader in by_id[r["id"]]:
            raise ValueError("duplicate reader for candidate")
        by_id[r["id"]][reader] = r
    output = []
    for key, candidate in candidates.items():
        decisions = list(by_id[key].values())
        if len(decisions) < 2:
            status = "pending_two_readers"
        elif any(r["decision"] == "benign_or_ambiguous" for r in decisions):
            status = "excluded_ambiguous_or_disagreed"
        else:
            status = "accepted_visible_misconduct"
        output.append(
            {
                "id": key,
                "trajectory_sha256": candidate.get("trajectory_sha256"),
                "status": status,
                "readers": sorted(by_id[key]),
            }
        )
    return output


def main() -> None:
    argparse.ArgumentParser(__doc__).parse_args()
    prepared = DATA / "prepared"
    selected = read_jsonl(prepared / "selection.jsonl")
    rows = read_jsonl(prepared / "rows_4b-v03.jsonl")
    # Blind the source judge's categories, rationale, status and exact evidence.
    # Include matched negative decoys. Mapping stays in a separate local file.
    coding = [
        r
        for r in rows
        if r["metadata"]["benchmark"] in {"swebench", "impossiblebench"}
        and r["metadata"]["label_class"] in {"positive", "negative"}
    ]
    packets, mapping = [], []
    for row in coding:
        text = split_prompt(row["prompt"]).trajectory
        blind_id = sha("goodfire-blind:" + row["id"])[:24]
        packets.append(
            {"id": blind_id, "trajectory": text, "trajectory_sha256": sha(text)}
        )
        mapping.append(
            {"id": row["id"], "blind_id": blind_id, "trajectory_sha256": sha(text)}
        )
    packets.sort(key=lambda r: r["id"])
    out = prepared / "audit"
    write_jsonl(out / "packets.jsonl", packets)
    write_jsonl(out / "mapping.jsonl", mapping)
    records_path = DATA / "adjudications.jsonl"
    records = read_jsonl(records_path) if records_path.exists() else []
    dispositions = adjudicate(selected, records)
    write_jsonl(out / "dispositions.jsonl", dispositions)
    write_json(
        out / "gate.json",
        {
            "coding_candidates": len(dispositions),
            "accepted": sum(
                r["status"] == "accepted_visible_misconduct" for r in dispositions
            ),
            "pending": sum(r["status"] == "pending_two_readers" for r in dispositions),
            "ready_for_scoring": False,  # Execution is separately authorized.
            "rule": (
                "Two independent readers; either ambiguous/disagreeing excludes; "
                "no relabel to negative."
            ),
        },
    )
    print(
        f"{len(dispositions)} coding candidates, "
        f"{len(packets)} blinded packets, no scores"
    )


if __name__ == "__main__":
    main()
