"""Backfill ranking comparisons from completed frozen inference predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.b200_inference_benchmark.run import DATA, ROOT, sha, write
from gleipnir.inference_benchmark import ranking_comparison


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument(
        "--baseline",
        type=Path,
        default=Path("results/b200_inference_benchmark/merged01"),
    )
    args = parser.parse_args()
    baseline, candidate = ROOT / args.baseline, ROOT / args.candidate
    summaries = [
        json.loads((p / "summary.json").read_text()) for p in (baseline, candidate)
    ]
    if (
        any(s["status"] != "complete" for s in summaries)
        or summaries[0]["manifest_sha256"] != summaries[1]["manifest_sha256"]
    ):
        raise ValueError("completed workload binding drift")
    manifest = json.loads((DATA / "manifest.json").read_text())
    if sha(DATA / "quick.json") != manifest["files"]["quick"]:
        raise ValueError("frozen label/prompt file drift")
    rows = json.loads((DATA / "quick.json").read_text())
    hashes, comparisons = {}, {}
    for c in sorted({t["concurrency"] for t in summaries[1]["trials"]}):
        runs = []
        for path, summary in zip((baseline, candidate), summaries, strict=True):
            files = [
                path / f"c{c}_repeat{t['repeat']}.json"
                for t in summary["trials"]
                if t["concurrency"] == c
            ]
            hashes.update({str(p.relative_to(ROOT)): sha(p) for p in files})
            runs.append([json.loads(p.read_text()) for p in files])
        comparisons[str(c)] = ranking_comparison(rows, *runs)
    report = {
        "baseline": str(baseline),
        "candidate": str(candidate),
        "inputs_sha256": {"labels_prompts": sha(DATA / "quick.json"), **hashes},
        "implementation_sha256": sha(ROOT / "src/gleipnir/inference_benchmark.py"),
        "concurrency": comparisons,
    }
    write(candidate / "ranking_comparison.json", report)
    print(json.dumps({c: v["auroc_delta"] for c, v in comparisons.items()}, indent=2))


if __name__ == "__main__":
    main()
