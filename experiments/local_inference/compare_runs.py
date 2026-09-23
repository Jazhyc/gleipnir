"""Audit paired outputs and save a single-pass candidate comparison."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.local_inference.core import compare, write_json


def compare_runs(baseline: Path, candidate: Path) -> dict:
    """Require identical input/model identities before comparing measurements."""

    def read(root, name):
        return json.loads((root / name).read_text())

    a, b = [read(root, "result.json") for root in (baseline, candidate)]
    for key in (
        "subset_sha256",
        "merge_manifest_sha256",
        "reference_sha256",
        "rows",
        "prompt_tokens",
        "software",
        "runner_sha256",
    ):
        if a[key] != b[key]:
            raise ValueError(f"Unmatched run identity: {key}")
    if len(a["repeats"]) != 1 or len(b["repeats"]) != 1:
        raise ValueError("Expected one pass per condition")
    left, right = [read(root, "predictions_0.json") for root in (baseline, candidate)]
    for x, y in zip(left, right, strict=True):
        for key in ("id", "source", "label", "prompt_sha256", "tokens"):
            if x[key] != y[key]:
                raise ValueError(f"Unmatched prediction: {key}")
    paired = [
        {
            "id": x["id"],
            "source": x["source"],
            "baseline_score": x["score"],
            "candidate_score": y["score"],
            "score_delta": y["score"] - x["score"],
            "margin_delta": y["logit_margin"] - x["logit_margin"],
            "threshold_flip": (x["score"] >= 0.5) != (y["score"] >= 0.5),
        }
        for x, y in zip(left, right, strict=True)
    ]
    report = {
        "baseline": str(baseline),
        "candidate": str(candidate),
        "score_drift": compare([x["score"] for x in left], [x["score"] for x in right]),
        "scoring_speedup": a["median_seconds"] / b["median_seconds"],
        "scoring_seconds": [a["median_seconds"], b["median_seconds"]],
        "prompt_tokens_per_second": [
            a["median_prompt_tokens_per_second"],
            b["median_prompt_tokens_per_second"],
        ],
        "process_seconds": [
            read(root, "process_timing.json")["seconds"]
            for root in (baseline, candidate)
        ],
        "metrics": {"baseline": a["metrics"], "candidate": b["metrics"]},
        "paired": paired,
        "score_drift_by_source": {
            source: compare(
                [x["score"] for x in left if x["source"] == source],
                [x["score"] for x in right if x["source"] == source],
            )
            for source in sorted({x["source"] for x in left})
        },
        "score_drift_by_baseline_score": {},
        "unique_scores": [len({x["score"] for x in rows}) for rows in (left, right)],
        "note": "One pass each; no repeat-noise estimate or equivalence claim.",
    }
    for low, high in ((0, 0.1), (0.1, 0.4), (0.4, 0.6), (0.6, 0.9), (0.9, 1.01)):
        selected = [p for p in paired if low <= p["baseline_score"] < high]
        report["score_drift_by_baseline_score"][f"{low}:{high}"] = {
            "rows": len(selected),
            "max_absolute_delta": max(
                (abs(p["score_delta"]) for p in selected), default=None
            ),
        }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    args = parser.parse_args()
    report = compare_runs(args.baseline, args.candidate)
    write_json(args.candidate / "comparison.json", report)
    print(json.dumps({k: v for k, v in report.items() if k != "paired"}, indent=2))


if __name__ == "__main__":
    main()
