"""Score the EvilGenie rows with one monitor (GPU box; not run locally).

    S=experiments/adversarial_evilgenie/score.py
    PYTHONPATH=src:. .venv/bin/python $S smoke --monitor 4b-v03
    PYTHONPATH=src:. .venv/bin/python $S score --monitor 4b-v03
    PYTHONPATH=src:. .venv/bin/python $S score --monitor 27b-merged

The natural-cheats scorer and canary rule (`experiments/adversarial_natural_
cheats/score.py`): canaries first (two calibration rows by rank with their
earlier scores, one positive and one negative row that fit), finite scores, a
repeat within `max_repeat_drift`, a nonzero adapter effect where there is an
adapter, and the gated reference drift. Then the rows are scored with the
shared scorer, resumably (ids already in predictions.jsonl are skipped); rows
over 29,696 tokens are recorded as `too_long`, never truncated. The canaries
are rescored at the end and must repeat.

Calibration (`--calibration`):
- `reuse` (default): the calibration rows are not rescored. analyze.py reads
  the natural-cheats calibration predictions named in config.json, so the
  calibration canaries must also match those predictions within
  `max_natural_cheats_drift`; a missing predictions file fails the canary.
- `score`: also score the calibration rows into this experiment's output.

`--order priority` (default) scores positives first, then negatives and clean
negatives shuffled together (seed 0), then heuristic and error rows, so a run
stopped early leaves a random subset of the metric rows; scores do not depend
on the order. `--order file` keeps the file order.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

from experiments.adversarial_natural_cheats import score as nc_score
from gleipnir.adversarial import eval_sets

HERE = Path(__file__).resolve().parent
CONFIG = json.loads((HERE / "config.json").read_text())
ROOT = eval_sets.ROOT
DATA = ROOT / CONFIG["data"]
OUTPUT = ROOT / CONFIG["output"]
SHARED_CANARY_KEYS = (
    "calibration_ranks",
    "rows_per_class",
    "max_repeat_drift",
    "min_adapter_effect",
    "max_reference_drift",
    "reference",
)


def same_canary_rule() -> None:
    """Our canary settings are the natural-cheats ones (its checks are reused)."""
    ours, theirs = CONFIG["canary"], nc_score.CONFIG["canary"]
    for key in SHARED_CANARY_KEYS:
        mine, other = ours[key], theirs[key]
        if key == "reference":
            # natural cheats also defines monitors this experiment never scores
            # (e.g. 4b-base, its Amendment 2); compare only the ones used here
            other = {m: other.get(m) for m in mine}
        if mine != other:
            raise ValueError(f"canary {key} differs from adversarial_natural_cheats")


def rows_monitor(monitor: str) -> str:
    """The monitor whose built rows this monitor scores (27b-merged: 27b)."""
    return CONFIG["monitors"][monitor].get("rows_from", monitor)


def load_rows(monitor: str) -> list[dict[str, Any]]:
    path = DATA / f"rows_{rows_monitor(monitor)}.jsonl"
    expected = CONFIG["sha256"].get(path.name)
    if expected is None or eval_sets.file_sha256(path) != expected:
        raise ValueError(f"{path} does not match its SHA-256 in config.json")
    return [json.loads(line) for line in path.open()]


def load_calibration(monitor: str) -> list[dict[str, Any]]:
    path = ROOT / CONFIG["calibration"]["rows"][monitor]
    expected = CONFIG["calibration"]["sha256"].get(path.name)
    if expected != nc_score.CONFIG["sha256"].get(path.name):
        raise ValueError(
            f"{path.name}: SHA-256 differs from adversarial_natural_cheats"
        )
    if expected is None or eval_sets.file_sha256(path) != expected:
        raise ValueError(f"{path} does not match its SHA-256 in config.json")
    return [json.loads(line) for line in path.open()]


def reused_predictions(monitor: str) -> Path:
    return ROOT / CONFIG["calibration"]["predictions"][monitor]


def ordered(rows: list[dict], order: str, seed: int = 0) -> list[dict]:
    if order == "file":
        return rows
    tiers: dict[int, list[dict]] = {0: [], 1: [], 2: []}
    for row in rows:
        name = row["metadata"]["label_class"]
        tier = (
            0
            if name == "positive"
            else 1
            if name in ("negative", "clean_negative")
            else 2
        )
        tiers[tier].append(row)
    random.Random(seed).shuffle(tiers[1])
    return tiers[0] + tiers[1] + tiers[2]


def canary_rows(
    monitor: str, sets: dict[str, list[dict]], reuse: bool
) -> list[dict[str, Any]]:
    """Calibration rows by rank (earlier scores attached) and a few rows."""
    spec = CONFIG["canary"]
    reference = nc_score.read_scores(ROOT / spec["reference"][monitor]["path"])
    reused = nc_score.read_scores(reused_predictions(monitor)) if reuse else {}
    chosen = []
    for row in sets["calibration"]:
        if row["metadata"]["calibration_rank"] in spec["calibration_ranks"]:
            chosen.append(
                {
                    "row": row,
                    "kind": "calibration",
                    "reference": reference.get(row["id"]),
                    "natural_cheats": reused.get(row["id"]) if reuse else None,
                }
            )
    too_long = rows_monitor(monitor)
    for name in spec["row_classes"]:
        picks = [
            r
            for r in sets["rows"]
            if r["metadata"]["label_class"] == name
            and not r["metadata"]["too_long"][too_long]
        ][: spec["rows_per_class"]]
        chosen += [
            {"row": r, "kind": name, "reference": None, "natural_cheats": None}
            for r in picks
        ]
    return chosen


def check_canaries(
    monitor: str, results: list[dict[str, Any]], has_adapter: bool, reuse: bool
) -> dict[str, Any]:
    """The natural-cheats rule, plus agreement with reused calibration scores."""
    out = nc_score.check_canaries(monitor, results, has_adapter)
    failures = list(out["failures"])
    calibration = [r for r in results if r["kind"] == "calibration"]
    drift = [
        abs(r["first"]["score"] - r["natural_cheats"])
        for r in calibration
        if r.get("natural_cheats") is not None
    ]
    if reuse:
        if len(drift) < len(calibration) or not calibration:
            failures.append("reused calibration predictions missing")
        elif max(drift) > CONFIG["canary"]["max_natural_cheats_drift"]:
            failures.append(f"drift from reused calibration {max(drift):.3g}")
    return {
        **out,
        "passed": not failures,
        "failures": failures,
        "calibration_mode": "reuse" if reuse else "score",
        "reused_predictions": str(reused_predictions(monitor)) if reuse else None,
        "natural_cheats_drift": drift,
    }


def run_canaries(
    scorer: Any, monitor: str, sets: dict[str, list[dict]], reuse: bool
) -> dict:
    results = []
    for item in canary_rows(monitor, sets, reuse):
        row = item["row"]
        first = scorer.score(row["prompt"])
        record = {
            "id": row["id"],
            "kind": item["kind"],
            "reference": item["reference"],
            "natural_cheats": item["natural_cheats"],
            "first": first,
            "repeat": scorer.score(row["prompt"]),
        }
        if scorer.spec.adapter is not None:
            base = scorer.score(row["prompt"], adapter=False)
            record["unadapted"] = base
            record["adapter_effect"] = abs(first["score"] - base["score"])
        results.append(record)
        print({k: v for k, v in record.items() if k != "repeat"}, flush=True)
    return check_canaries(monitor, results, scorer.spec.adapter is not None, reuse)


def code_sha256() -> dict[str, str]:
    """The experiment's code, since the box may run it uncommitted (rsync)."""
    return {p.name: eval_sets.file_sha256(p) for p in sorted(HERE.glob("*.py"))}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("smoke", "score"))
    parser.add_argument("--monitor", required=True, choices=sorted(CONFIG["monitors"]))
    parser.add_argument("--calibration", choices=("reuse", "score"), default="reuse")
    parser.add_argument("--order", choices=("priority", "file"), default="priority")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    same_canary_rule()
    reuse = args.calibration == "reuse"

    from gleipnir.adversarial.scorer import Scorer, score_rows

    sets = {
        "calibration": load_calibration(args.monitor),
        "rows": load_rows(args.monitor),
    }
    scorer = Scorer(args.monitor, device=args.device)
    directory = OUTPUT / args.monitor
    identity = {
        "commit": nc_score.git_commit(),
        "code_sha256": code_sha256(),
        "scorer": scorer.identity(),
    }
    canary = run_canaries(scorer, args.monitor, sets, reuse)
    nc_score.save(directory / "canary.json", {**identity, **canary})
    print(f"canaries passed={canary['passed']} {canary['failures']}", flush=True)
    if not canary["passed"]:
        raise SystemExit(1)
    if args.command == "smoke":
        return
    names = ["rows"] if reuse else ["rows", "calibration"]
    for name in names:
        rows = ordered(sets[name], args.order) if name == "rows" else sets[name]
        source = (
            f"rows_{rows_monitor(args.monitor)}.jsonl"
            if name == "rows"
            else Path(CONFIG["calibration"]["rows"][args.monitor]).name
        )
        digest = (
            CONFIG["sha256"][source]
            if name == "rows"
            else CONFIG["calibration"]["sha256"][source]
        )
        out = directory / name
        nc_score.save(
            out / "identity.json",
            {
                **identity,
                "monitor": args.monitor,
                "set": name,
                "rows": len(rows),
                "input": source,
                "input_sha256": digest,
                "order": args.order if name == "rows" else "file",
                "calibration_mode": args.calibration,
            },
        )
        score_rows(scorer, rows, out / "predictions.jsonl")
    prompts = {r["id"]: r["prompt"] for rows in sets.values() for r in rows}
    final = nc_score.final_canaries(scorer, canary, prompts)
    nc_score.save(directory / "canary_final.json", final)
    print(f"final canaries passed={final['passed']}", flush=True)
    if not final["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
