"""Cross-GPU drift check: rescore rows already scored on another machine.

    python experiments/adversarial_lab_v03/drift.py score --ids ids.txt --out drift_h100
    python experiments/adversarial_lab_v03/drift.py compare --a ood --b drift_h100

Run 0 sets thresholds on the A100 while Runs 1-2 ran on the H100, so this
measures how far identical inputs move between the two machines.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.adversarial_lab_v03.run import CONFIG, OUTPUT, load, read


def score(args: argparse.Namespace) -> None:
    from gleipnir.adversarial.scorer import Scorer, score_rows

    wanted = [line.strip() for line in Path(args.ids).open() if line.strip()]
    rows = {str(r["id"]): r for r in load("ood")}
    directory = OUTPUT / args.out
    directory.mkdir(parents=True, exist_ok=True)
    scorer = Scorer(CONFIG["monitor"], device=args.device)
    (directory / "identity.json").write_text(
        json.dumps({"ids": len(wanted), "scorer": scorer.identity()}, indent=2) + "\n"
    )
    score_rows(scorer, [rows[i] for i in wanted], directory / "predictions.jsonl")


def compare(args: argparse.Namespace) -> None:
    a = {r["id"]: r for r in read(args.a)}
    b = {r["id"]: r for r in read(args.b)}
    common = sorted(set(a) & set(b))
    diffs = [abs(a[i]["score"] - b[i]["score"]) for i in common]
    margins = [
        abs((a[i]["logit_1"] - a[i]["logit_0"]) - (b[i]["logit_1"] - b[i]["logit_0"]))
        for i in common
    ]
    result = {
        "rows": len(common),
        "identical_scores": sum(d == 0 for d in diffs),
        "max_abs_score_diff": max(diffs),
        "mean_abs_score_diff": sum(diffs) / len(diffs),
        "max_abs_margin_diff": max(margins),
        "mean_abs_margin_diff": sum(margins) / len(margins),
    }
    print(json.dumps(result, indent=2))
    (OUTPUT / f"{args.b}_compare.json").write_text(json.dumps(result, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    s = commands.add_parser("score")
    s.add_argument("--ids", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--device", default="cuda")
    c = commands.add_parser("compare")
    c.add_argument("--a", required=True)
    c.add_argument("--b", required=True)
    args = parser.parse_args()
    {"score": score, "compare": compare}[args.command](args)


if __name__ == "__main__":
    main()
