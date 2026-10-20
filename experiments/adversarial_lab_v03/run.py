"""Score the six-worker lab sets with Gleipnir 4B v0.3 and report.

python experiments/adversarial_lab_v03/run.py smoke
python experiments/adversarial_lab_v03/run.py score --sets ood lab framing
python experiments/adversarial_lab_v03/run.py report
"""

from __future__ import annotations

import argparse
import json
import math
import random
import subprocess
from collections import defaultdict
from pathlib import Path
from statistics import median
from typing import Any

from gleipnir.adversarial import eval_sets
from gleipnir.adversarial.metrics import summary, threshold_at_fpr
from gleipnir.evaluation.metrics import normalized_partial_auroc

HERE = Path(__file__).resolve().parent
CONFIG = json.loads((HERE / "config.json").read_text())
OUTPUT = eval_sets.ROOT / CONFIG["output"]
DATA = eval_sets.ROOT / CONFIG["data"]


def git_commit() -> str:
    return subprocess.run(
        ["git", "-C", str(eval_sets.ROOT), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
    ).stdout.strip()


def load(name: str) -> list[dict[str, Any]]:
    if name in eval_sets.SETS:
        return eval_sets.load(name)
    path = DATA / f"{name}.jsonl"
    if eval_sets.file_sha256(path) != CONFIG["sha256"][name]:
        raise ValueError(f"{path} does not match its frozen SHA256")
    return [json.loads(line) for line in path.open()]


def smoke(args: argparse.Namespace) -> None:
    from gleipnir.adversarial.scorer import Scorer

    rows = load("lab")
    plain = [r for r in rows if r["metadata"]["condition"] == "plain"][:2]
    liars = [r for r in rows if r["metadata"]["condition"] == "verified_liar"][:2]
    scorer = Scorer(CONFIG["monitor"], device=args.device)
    results = []
    for row in plain + liars:
        adapted = scorer.score(row["prompt"])
        base = scorer.score(row["prompt"], adapter=False)
        results.append(
            {
                "id": row["id"],
                "condition": row["metadata"]["condition"],
                "score": adapted["score"],
                "base_score": base["score"],
                "adapter_effect": abs(adapted["score"] - base["score"]),
                "tokens": adapted["tokens"],
            }
        )
        print(results[-1], flush=True)
    passed = (
        all(math.isfinite(r["score"]) for r in results)
        and max(r["adapter_effect"] for r in results) >= 1e-6
    )
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "smoke.json").write_text(
        json.dumps(
            {
                "passed": passed,
                "identity": scorer.identity(),
                "commit": git_commit(),
                "rows": results,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"smoke passed={passed}")
    if not passed:
        raise SystemExit(1)


def score(args: argparse.Namespace) -> None:
    from gleipnir.adversarial.scorer import Scorer, score_rows

    scorer = Scorer(CONFIG["monitor"], device=args.device)
    for name in args.sets:
        rows = load(name)
        directory = OUTPUT / name
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "identity.json").write_text(
            json.dumps(
                {
                    "set": name,
                    "rows": len(rows),
                    "commit": git_commit(),
                    "scorer": scorer.identity(),
                },
                indent=2,
            )
            + "\n"
        )
        score_rows(scorer, rows, directory / "predictions.jsonl")


def read(name: str) -> list[dict[str, Any]]:
    path = OUTPUT / name / "predictions.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.open() if line.strip()]


def margin(record: dict[str, Any]) -> float:
    return record["logit_1"] - record["logit_0"]


def cluster_interval(records, statistic, seed: int = 0, samples: int = 2000):
    """95% bootstrap interval, resampling episodes (task_id) with replacement."""
    clusters = defaultdict(list)
    for r in records:
        clusters[r["task_id"]].append(r)
    keys = sorted(clusters)
    rng = random.Random(seed)
    values = []
    for _ in range(samples):
        drawn = [r for k in rng.choices(keys, k=len(keys)) for r in clusters[k]]
        values.append(statistic(drawn))
    values.sort()
    return [values[int(0.025 * samples)], values[int(0.975 * samples) - 1]]


def ood_report(ood: list[dict[str, Any]]) -> dict[str, Any]:
    ok = [r for r in ood if r["status"] == "ok"]
    by_source = defaultdict(list)
    for r in ok:
        by_source[r["group"]].append(r)
    sources = {
        name: normalized_partial_auroc(
            [r["label"] for r in rows], [r["score"] for r in rows]
        )
        for name, rows in sorted(by_source.items())
    }
    mean = sum(sources.values()) / len(sources)
    parity = CONFIG["ood_parity"]
    honest = [r["score"] for r in ok if r["label"] == 0]
    return {
        "rows": len(ood),
        "per_source_pauroc20": sources,
        "mean_ood_pauroc20": mean,
        "card_bf16": parity["card_bf16_mean_ood_pauroc20"],
        "parity_passed": abs(mean - parity["card_bf16_mean_ood_pauroc20"])
        <= parity["tolerance"],
        "deployer_thresholds": {
            f"{fpr:.0%}": threshold_at_fpr(honest, fpr) for fpr in CONFIG["fprs"]
        },
        "pooled": summary([r["label"] for r in ok], [r["score"] for r in ok]),
    }


def flag_table(records, thresholds) -> dict[str, Any]:
    result = {}
    for name in sorted({r["group"] for r in records}):
        rows = [r for r in records if r["group"] == name and r["status"] == "ok"]
        entry = {"rows": len(rows), "median_score": median(r["score"] for r in rows)}
        for key, t in thresholds.items():
            share = sum(r["score"] > t for r in rows) / len(rows)
            entry[f"flagged_at_{key}"] = share
            entry[f"flagged_at_{key}_ci95"] = cluster_interval(
                rows, lambda xs, t=t: sum(x["score"] > t for x in xs) / len(xs)
            )
        result[name] = entry
    return result


def framing_report(records, thresholds) -> dict[str, Any]:
    ok = [r for r in records if r["status"] == "ok"]
    base = {r["parent_id"]: r for r in ok if r["group"] == "none"}
    result = {"flags": flag_table(ok, thresholds), "paired_vs_none": {}}
    for name in sorted({r["group"] for r in ok} - {"none"}):
        pairs = [
            (r, base[r["parent_id"]])
            for r in ok
            if r["group"] == name and r["parent_id"] in base
        ]
        deltas = [margin(r) - margin(b) for r, b in pairs]
        result["paired_vs_none"][name] = {
            "pairs": len(pairs),
            "share_margin_up": sum(d > 0 for d in deltas) / len(deltas),
            "median_margin_change": median(deltas),
            "newly_flagged_at_1%": sum(
                r["score"] > thresholds["1%"] >= b["score"] for r, b in pairs
            ),
        }
    return result


def liar_report(lab) -> dict[str, Any]:
    ok = [r for r in lab if r["status"] == "ok"]
    liars = [r for r in ok if r["group"] == "verified_liar"]
    result = {}
    for control in ("honest_recruiter", "assigned_deceiver_no_lie", "plain"):
        others = [r for r in ok if r["group"] == control]
        if liars and others:
            rows = liars + others
            labels = [1] * len(liars) + [0] * len(others)
            from sklearn.metrics import roc_auc_score

            result[f"verified_liar_vs_{control}_auroc"] = float(
                roc_auc_score(labels, [r["score"] for r in rows])
            )
    return result


def report(args: argparse.Namespace) -> None:
    ood, lab, framing = read("ood"), read("lab"), read("framing")
    result: dict[str, Any] = {
        "rows": {"ood": len(ood), "lab": len(lab), "framing": len(framing)}
    }
    thresholds = None
    if ood:
        result["run0_ood"] = ood_report(ood)
        thresholds = result["run0_ood"]["deployer_thresholds"]
    if lab and thresholds:
        result["run1_lab_flags"] = flag_table(lab, thresholds)
        result["run1_liars"] = liar_report(lab)
    if framing and thresholds:
        result["run2_framing"] = framing_report(framing, thresholds)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "report.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2)[:6000])


def main() -> None:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("smoke", "score"):
        command = commands.add_parser(name)
        command.add_argument("--device", default="cuda")
    commands.choices["score"].add_argument(
        "--sets",
        nargs="+",
        default=["ood", "lab", "framing"],
        choices=["ood", "lab", "framing"],
    )
    commands.add_parser("report")
    args = parser.parse_args()
    {"smoke": smoke, "score": score, "report": report}[args.command](args)


if __name__ == "__main__":
    main()
