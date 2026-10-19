"""Apply the frozen pilot gate to one teacher's pilot extraction."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

from experiments.aux_supervision.extract import load
from experiments.aux_supervision.targets import grounding, parse_continuation
from gleipnir.data.monitoring import read_rows, write_json

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/aux_supervision"


def gate(rows: list[dict]) -> dict:
    positives = [r for r in rows if r["label"] == 1]
    negatives = [r for r in rows if r["label"] == 0]
    grounded = [
        r
        for r in positives
        if r["grounding"]["status"] == "unique"
        and r["grounding"].get("speaker") == "agent"
    ]
    rates = {
        "well_formed": sum(r["well_formed"] for r in rows) / len(rows),
        "positive_unique_agent_quote": len(grounded) / len(positives),
        "negative_none": sum(r["grounding"]["status"] == "none" for r in negatives)
        / len(negatives),
    }
    conditions = {
        "well_formed": rates["well_formed"] >= 0.90,
        "positive_unique_agent_quote": rates["positive_unique_agent_quote"] >= 0.60,
        "negative_none": rates["negative_none"] >= 0.90,
    }
    statuses = Counter(
        (r["label"], r["grounding"]["status"], r["grounding"].get("speaker"))
        for r in rows
    )
    return {
        "rows": len(rows),
        "rates": rates,
        "conditions": conditions,
        "automatic_passed": all(conditions.values()),
        "grounding_counts": {str(k): v for k, v in sorted(statuses.items(), key=str)},
        "finish_reasons": dict(Counter(r["finish_reason"] for r in rows)),
        "mean_generated_tokens": sum(len(r["steps"]) for r in rows) / len(rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--name", default="pilot01")
    args = parser.parse_args()
    directory = OUT / args.name / args.model
    rows = read_rows(directory / "rows.jsonl")
    summary = json.loads((directory / "summary.json").read_text())
    trajectories = {
        r["index"]: r["trajectory"] for r in load(summary["limit"], summary["seed"])[0]
    }
    strict = gate(rows)["rates"]["positive_unique_agent_quote"]
    for r in rows:
        # Re-parse and re-ground with the current code; stored fields may be older.
        target = parse_continuation(r["text"])
        r["position"], r["reason"] = target.position, target.reason
        r["well_formed"] = target.well_formed
        r["grounding"] = grounding(trajectories[r["id"]], r["label"], target)
    report = gate(rows)
    report["positive_unique_agent_quote_exact_match_checker"] = strict
    write_json(directory / "gate.json", report)
    print(json.dumps(report, indent=1))
    rng = random.Random(0)
    sample = rng.sample([r for r in rows if r["label"] == 1], 10) + rng.sample(
        [r for r in rows if r["label"] == 0], 10
    )
    for r in sample:
        print(f"\n[{r['source']} label={r['label']} {r['grounding']}]")
        print(r["text"].strip()[:600])


if __name__ == "__main__":
    main()
