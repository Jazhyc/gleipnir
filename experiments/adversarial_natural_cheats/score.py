"""Score the natural-cheats rows with one monitor (GPU box; not run locally).

    S=experiments/adversarial_natural_cheats/score.py
    PYTHONPATH=src:. python $S smoke --monitor 4b-v03
    PYTHONPATH=src:. python $S score --monitor 4b-v03
    PYTHONPATH=src:. python $S score --monitor 27b

Both commands run the drift canaries first and stop if they fail: finite
scores, a repeat score within `max_repeat_drift`, a nonzero adapter effect, and
for the 4B, agreement with its earlier PEFT A100 scores within
`max_reference_drift` (the 27B has no earlier PEFT scores on these rows; its
comparison against the merged 27B is recorded only). `score` then scores the
1,000 strict-OOD calibration rows and the experiment rows with the shared
scorer, resumably (ids already in predictions.jsonl are skipped). Rows over
29,696 tokens are recorded as `too_long`, never truncated. The canaries are
rescored at the end and must repeat.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
from pathlib import Path
from typing import Any

from gleipnir.adversarial import eval_sets

HERE = Path(__file__).resolve().parent
CONFIG = json.loads((HERE / "config.json").read_text())
ROOT = eval_sets.ROOT
DATA = ROOT / CONFIG["data"]
OUTPUT = ROOT / CONFIG["output"]
SETS = ("calibration", "rows")


def git_commit() -> str:
    return subprocess.run(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True
    ).stdout.strip()


def save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def load_set(name: str, monitor: str) -> list[dict[str, Any]]:
    path = DATA / f"{name}_{monitor}.jsonl"
    expected = CONFIG["sha256"].get(path.name)
    if expected is None or eval_sets.file_sha256(path) != expected:
        raise ValueError(f"{path} does not match its SHA256 in config.json")
    return [json.loads(line) for line in path.open()]


def read_scores(path: Path) -> dict[str, float]:
    if not path.exists():
        return {}
    scores = {}
    for line in path.open():
        record = json.loads(line)
        if record.get("status") == "ok":
            scores[record["id"]] = record["score"]
    return scores


def canary_rows(monitor: str, sets: dict[str, list[dict]]) -> list[dict[str, Any]]:
    """Calibration rows by rank (with reference scores) and a few experiment rows."""
    spec = CONFIG["canary"]
    reference = read_scores(ROOT / spec["reference"][monitor]["path"])
    chosen = []
    for row in sets["calibration"]:
        if row["metadata"]["calibration_rank"] in spec["calibration_ranks"]:
            chosen.append(
                {
                    "row": row,
                    "kind": "calibration",
                    "reference": reference.get(row["id"]),
                }
            )
    for name in spec["row_classes"]:
        picks = [
            r
            for r in sets["rows"]
            if r["metadata"]["detection_point"] == spec["row_point"]
            and r["metadata"]["label_class"] == name
            and not r["metadata"]["too_long"][monitor]
        ][: spec["rows_per_class"]]
        chosen += [{"row": r, "kind": name, "reference": None} for r in picks]
    return chosen


def finite(value: dict[str, Any]) -> bool:
    return all(math.isfinite(value[k]) for k in ("score", "logit_0", "logit_1"))


def check_canaries(
    monitor: str, results: list[dict[str, Any]], has_adapter: bool
) -> dict[str, Any]:
    """Apply the canary rules to scored canaries (pure; tested with mocks)."""
    spec = CONFIG["canary"]
    gate = spec["reference"][monitor]["gate"]
    failures = []
    if not all(finite(r["first"]) and finite(r["repeat"]) for r in results):
        failures.append("non-finite score or logit")
    repeat = [abs(r["first"]["score"] - r["repeat"]["score"]) for r in results]
    if max(repeat) > spec["max_repeat_drift"]:
        failures.append(f"repeat drift {max(repeat):.3g}")
    effects = [
        r["adapter_effect"] for r in results if r.get("adapter_effect") is not None
    ]
    if has_adapter and (not effects or max(effects) < spec["min_adapter_effect"]):
        failures.append("no adapter effect")
    reference = [
        abs(r["first"]["score"] - r["reference"])
        for r in results
        if r["reference"] is not None
    ]
    expected = sum(1 for r in results if r["kind"] == "calibration")
    if gate and len(reference) < expected:
        failures.append("reference scores missing")
    if gate and reference and max(reference) > spec["max_reference_drift"]:
        failures.append(f"reference drift {max(reference):.3g}")
    return {
        "passed": not failures,
        "failures": failures,
        "max_repeat_drift": max(repeat),
        "adapter_effects": effects,
        "reference_drift": reference,
        "reference": spec["reference"][monitor],
        "rows": results,
    }


def run_canaries(scorer: Any, monitor: str, sets: dict[str, list[dict]]) -> dict:
    results = []
    for item in canary_rows(monitor, sets):
        row = item["row"]
        first = scorer.score(row["prompt"])
        record = {
            "id": row["id"],
            "kind": item["kind"],
            "reference": item["reference"],
            "first": first,
            "repeat": scorer.score(row["prompt"]),
        }
        if scorer.spec.adapter is not None:
            base = scorer.score(row["prompt"], adapter=False)
            record["unadapted"] = base
            record["adapter_effect"] = abs(first["score"] - base["score"])
        results.append(record)
        print({k: v for k, v in record.items() if k != "repeat"}, flush=True)
    return check_canaries(monitor, results, scorer.spec.adapter is not None)


def final_canaries(
    scorer: Any, canary: dict[str, Any], prompts: dict[str, str]
) -> dict:
    drift = {
        r["id"]: abs(scorer.score(prompts[r["id"]])["score"] - r["first"]["score"])
        for r in canary["rows"]
    }
    worst = max(drift.values())
    return {
        "passed": worst <= CONFIG["canary"]["max_repeat_drift"],
        "drift": drift,
        "max_drift": worst,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("smoke", "score"))
    parser.add_argument("--monitor", required=True, choices=sorted(CONFIG["monitors"]))
    parser.add_argument("--sets", nargs="+", choices=SETS, default=list(SETS))
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    from gleipnir.adversarial.scorer import Scorer, score_rows

    sets = {name: load_set(name, args.monitor) for name in SETS}
    scorer = Scorer(args.monitor, device=args.device)
    directory = OUTPUT / args.monitor
    identity = {"commit": git_commit(), "scorer": scorer.identity()}
    canary = run_canaries(scorer, args.monitor, sets)
    save(directory / "canary.json", {**identity, **canary})
    print(f"canaries passed={canary['passed']} {canary['failures']}", flush=True)
    if not canary["passed"]:
        raise SystemExit(1)
    if args.command == "smoke":
        return
    for name in args.sets:
        rows = sets[name]
        out = directory / name
        save(
            out / "identity.json",
            {
                **identity,
                "monitor": args.monitor,
                "set": name,
                "rows": len(rows),
                "input_sha256": CONFIG["sha256"][f"{name}_{args.monitor}.jsonl"],
            },
        )
        score_rows(scorer, rows, out / "predictions.jsonl")
    prompts = {r["id"]: r["prompt"] for rows in sets.values() for r in rows}
    final = final_canaries(scorer, canary, prompts)
    save(directory / "canary_final.json", final)
    print(f"final canaries passed={final['passed']}", flush=True)
    if not final["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
