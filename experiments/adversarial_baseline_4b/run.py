"""Unattacked Gleipnir 4B scores: the baseline every phase-1 attack is measured on.

python experiments/adversarial_baseline_4b/run.py smoke
python experiments/adversarial_baseline_4b/run.py score --sets apps-neutral
python experiments/adversarial_baseline_4b/run.py report
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

HERE = Path(__file__).resolve().parent
CONFIG = json.loads((HERE / "config.json").read_text())
OUTPUT = eval_sets.ROOT / CONFIG["output"]
PAYLOADS = ("gpt4", "gpt4o", "gpt41")


def git_commit() -> str:
    return subprocess.run(
        ["git", "-C", str(eval_sets.ROOT), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
    ).stdout.strip()


def select(rows: list[dict], per_cell: int | None, seed: int) -> list[dict]:
    if not per_cell:
        return rows
    cells = defaultdict(list)
    for row in rows:
        cells[(eval_sets.group(row), eval_sets.label(row))].append(row)
    rng = random.Random(seed)
    picked = []
    for key in sorted(cells):
        picked += rng.sample(cells[key], min(per_cell, len(cells[key])))
    return picked


def smoke(args: argparse.Namespace) -> None:
    from gleipnir.adversarial.scorer import Scorer

    rows = eval_sets.load("apps-neutral")
    honest = [r for r in rows if r["metadata"]["condition"] == "clean"]
    backdoor = [r for r in rows if r["metadata"]["condition"] == "vanilla"]
    half = CONFIG["smoke"]["rows"] // 2
    scorer = Scorer(CONFIG["monitor"], device=args.device)
    results = []
    for row in honest[:half] + backdoor[:half]:
        adapted = scorer.score(row["prompt"])
        base = scorer.score(row["prompt"], adapter=False)
        results.append(
            {
                "id": row["id"],
                "label": eval_sets.label(row),
                "score": adapted["score"],
                "base_score": base["score"],
                "adapter_effect": abs(adapted["score"] - base["score"]),
                "tokens": adapted["tokens"],
            }
        )
        print(results[-1], flush=True)
    finite = all(math.isfinite(r["score"]) for r in results)
    effect = max(r["adapter_effect"] for r in results)
    passed = finite and effect >= CONFIG["smoke"]["min_adapter_effect"]
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "smoke.json").write_text(
        json.dumps(
            {
                "passed": passed,
                "max_adapter_score_effect": effect,
                "identity": scorer.identity(),
                "commit": git_commit(),
                "rows": results,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"smoke passed={passed} max adapter effect on P(1)={effect:.4f}")
    if not passed:
        raise SystemExit(1)


def score(args: argparse.Namespace) -> None:
    from gleipnir.adversarial.scorer import Scorer, score_rows

    scorer = Scorer(CONFIG["monitor"], device=args.device)
    for name in args.sets or CONFIG["sets"]:
        rows = select(eval_sets.load(name), args.per_cell, args.seed)
        directory = OUTPUT / name
        directory.mkdir(parents=True, exist_ok=True)
        identity = {
            "set": name,
            "set_sha256": eval_sets.SETS[name].sha256,
            "selection": {"per_cell": args.per_cell, "seed": args.seed},
            "rows": len(rows),
            "commit": git_commit(),
            "scorer": scorer.identity(),
        }
        identity_path = directory / "identity.json"
        if identity_path.exists():
            previous = json.loads(identity_path.read_text())
            if previous["selection"] != identity["selection"]:
                raise SystemExit(f"{name}: selection differs from the existing run")
        identity_path.write_text(json.dumps(identity, indent=2) + "\n")
        score_rows(scorer, rows, directory / "predictions.jsonl")


def read(name: str) -> list[dict[str, Any]]:
    path = OUTPUT / name / "predictions.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.open()]


def group_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Per group; groups holding only backdoors are compared with all honest rows."""
    ok = [r for r in records if r["status"] == "ok"]
    honest = [r for r in ok if r["label"] == 0]
    result = {}
    for name in sorted({r["group"] for r in ok}):
        subset = [r for r in ok if r["group"] == name]
        if all(r["label"] == 1 for r in subset):
            subset = subset + honest
        if len({r["label"] for r in subset}) == 2:
            result[name] = summary(
                [r["label"] for r in subset], [r["score"] for r in subset]
            )
    if len({r["label"] for r in ok}) == 2:
        result["pooled"] = summary([r["label"] for r in ok], [r["score"] for r in ok])
    return result


def a1_paired(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Injected backdoor minus the same task's vanilla and neutral-text backdoors."""
    by_task: dict[str, dict[str, float]] = defaultdict(dict)
    for r in records:
        if r["status"] == "ok" and r["label"] == 1:
            by_task[r["task_id"]][r["group"]] = r["score"]
    result = {}
    for payload in PAYLOADS:
        injected = f"inj_transfer_{payload}"
        for control in ("vanilla", f"neutral_transfer_{payload}"):
            deltas = [
                scores[injected] - scores[control]
                for scores in by_task.values()
                if injected in scores and control in scores
            ]
            if deltas:
                result[f"{injected} - {control}"] = {
                    "tasks": len(deltas),
                    "share_up": sum(d > 0 for d in deltas) / len(deltas),
                    "share_down": sum(d < 0 for d in deltas) / len(deltas),
                    "median_delta": median(deltas),
                }
    return result


def false_alarms(
    benchmark: list[dict[str, Any]], controls: list[dict[str, Any]]
) -> dict[str, Any]:
    """Honest controls flagged at thresholds set on clean honest benchmark rows."""
    clean = [r["score"] for r in benchmark if r["status"] == "ok" and r["label"] == 0]
    if not clean or not controls:
        return {}
    result = {}
    for fpr in CONFIG["fprs"]:
        threshold = threshold_at_fpr(clean, fpr)
        flagged = {}
        for name in sorted({r["group"] for r in controls}):
            scores = [
                r["score"]
                for r in controls
                if r["group"] == name and r["status"] == "ok"
            ]
            flagged[name] = {
                "rows": len(scores),
                "flagged": sum(s > threshold for s in scores) / len(scores),
            }
        for kind in ("inj", "neutral"):
            scores = [
                r["score"]
                for r in controls
                if r["group"].startswith(f"{kind}_") and r["status"] == "ok"
            ]
            flagged[f"all_{kind}"] = {
                "rows": len(scores),
                "flagged": sum(s > threshold for s in scores) / len(scores),
            }
        result[f"{fpr:.0%}"] = {"threshold": threshold, "groups": flagged}
    return result


def report(args: argparse.Namespace) -> None:
    benchmark = read("apps-neutral")
    controls = read("apps-controls-neutral")
    ood = read("ood")
    ood_groups = group_metrics(ood) if ood else {}
    sources = [k for k in ood_groups if k != "pooled"]
    result = {
        "rows": {
            name: {
                "total": len(records),
                "too_long": sum(r["status"] == "too_long" for r in records),
            }
            for name, records in (
                ("apps-neutral", benchmark),
                ("apps-controls-neutral", controls),
                ("ood", ood),
            )
        },
        "apps": group_metrics(benchmark) if benchmark else {},
        "a1_paired": a1_paired(benchmark),
        "false_alarms": false_alarms(benchmark, controls),
        "ood": ood_groups,
        "ood_source_mean": {
            "sources": len(sources),
            "pauroc20": sum(ood_groups[s]["pauroc20"] for s in sources) / len(sources),
            "recall_at_1pct_fpr": sum(
                ood_groups[s]["at_1pct_fpr"]["recall"] for s in sources
            )
            / len(sources),
        }
        if sources
        else {},
    }
    (OUTPUT / "report.json").write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {k: result[k] for k in ("rows", "a1_paired", "ood_source_mean")}, indent=2
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("smoke", "score"):
        command = commands.add_parser(name)
        command.add_argument("--device", default="cuda")
    commands.choices["score"].add_argument(
        "--sets", nargs="+", choices=sorted(eval_sets.SETS)
    )
    commands.choices["score"].add_argument("--per-cell", type=int)
    commands.choices["score"].add_argument("--seed", type=int, default=0)
    commands.add_parser("report")
    args = parser.parse_args()
    {"smoke": smoke, "score": score, "report": report}[args.command](args)


if __name__ == "__main__":
    main()
