"""Audit completed model pairs and collect their matched ID/OOD comparisons."""

from __future__ import annotations

import argparse
import json

from experiments.student_injection_awareness.prepare import OUTPUT, VARIANTS, write_json


def summarize(size: str) -> dict:
    conditions = {}
    initials = set()
    teachers = set()
    for variant in VARIANTS:
        directory = OUTPUT / size / variant
        complete = json.loads((directory / "complete.json").read_text())
        metadata = json.loads(
            (directory / "causal_adapter/training_metadata.json").read_text()
        )
        initials.add(metadata["sequence_packing"]["initial_master_sha256"])
        teachers.add(complete["teacher_targets_sha256"])
        parity = json.loads((directory / "serving_parity.json").read_text())
        if not parity["passed"]:
            raise ValueError("serving parity failed")
        reports = {
            split: json.loads((directory / split / "result.json").read_text())
            for split in ["id", "ood"]
        }
        if reports["id"]["rows"] != 3012 or reports["ood"]["rows"] != 6395:
            raise ValueError("evaluation coverage incomplete")
        conditions[variant] = {
            "complete": complete,
            "metrics": {s: r["metrics"] for s, r in reports.items()},
            "training_state": metadata["training_state"],
            "parity": parity["comparisons"],
            "upload": json.loads((directory / "upload.json").read_text())
            if (directory / "upload.json").exists()
            else None,
        }
    if len(initials) != 1 or len(teachers) != 1:
        raise ValueError("model pair initial weights or teacher targets differ")
    delta = {}
    for split in ["id", "ood"]:
        a = conditions["regular"]["metrics"][split]["macro"]["macro"]
        b = conditions["injection_aware"]["metrics"][split]["macro"]["macro"]
        delta[split] = {
            key: b[key] - a[key]
            for key in a
            if a[key] is not None and b[key] is not None
        }
    summary = {
        "size": size,
        "conditions": conditions,
        "aware_minus_regular": delta,
        "matched_initial_weights": True,
        "matched_teacher_targets": True,
    }
    write_json(OUTPUT / size / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", choices=("4b", "9b"), required=True)
    args = parser.parse_args()
    print(json.dumps(summarize(args.size)["aware_minus_regular"], indent=2), flush=True)


if __name__ == "__main__":
    main()
