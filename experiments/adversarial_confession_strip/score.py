"""Score the stripped rows on the GPU, one monitor at a time (resumable;
canaries before and after; no activations). Adapted from monitor-aware's
`score.py`:

    $PY experiments/adversarial_confession_strip/score.py canary --monitor 4b
    $PY experiments/adversarial_confession_strip/score.py score --monitor 4b
    $PY experiments/adversarial_confession_strip/score.py score --monitor 27b

Writes `results/adversarial_confession_strip/scoring/<monitor>/`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import time
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CONFIG = json.loads((HERE / "config.json").read_text())
DATA = ROOT / CONFIG["data"]
OUTPUT = ROOT / CONFIG["output"] / "scoring"


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.open()] if path.exists() else []


def git_commit() -> str:
    return subprocess.run(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True
    ).stdout.strip()


def load(key: str) -> list[dict[str, Any]]:
    path = DATA / f"{key}.jsonl"
    if file_sha256(path) != CONFIG["sha256"][key]:
        raise ValueError(f"{path} does not match its frozen SHA256")
    return read_jsonl(path)


def scorable(rows: list[dict[str, Any]], monitor: str) -> list[dict[str, Any]]:
    """Monitor-aware's scorable set: its too_long flags are kept unchanged."""
    return [r for r in rows if not r["metadata"]["too_long"][monitor]]


def record_for(row: dict[str, Any]) -> dict[str, Any]:
    m = row["metadata"]
    return {
        "id": row["id"],
        "group": m["condition"],
        "label": m["ground_truth"],
        "setup": m["setup"],
        "task_id": m["task_id"],
        "parent_id": m["parent_id"],
    }


def canary_check(results: list[dict], tolerance: float, has_adapter: bool) -> bool:
    """Pure: finite, within tolerance of the known score, adapter has an effect."""
    return bool(results) and all(
        r["finite"]
        and r["abs_diff"] <= tolerance
        and (not has_adapter or r["adapter_effect"] > 0)
        for r in results
    )


def load_scorer(key: str, device: str):
    from gleipnir.adversarial.scorer import Scorer

    return Scorer(CONFIG["gpu_monitors"][key]["monitor"], device=device)


def run_canaries(scorer, key: str) -> dict[str, Any]:
    tolerance = CONFIG["canary"]["max_abs_score"]
    has_adapter = scorer.spec.adapter is not None
    results = []
    for row in (r for r in load("canaries") if r["monitor"] == key):
        plain = scorer.score(row["prompt"])
        base = scorer.score(row["prompt"], adapter=False) if has_adapter else plain
        results.append(
            {
                "id": row["id"],
                "score": plain["score"],
                "expected": row["expected"],
                "abs_diff": abs(plain["score"] - row["expected"]),
                "adapter_effect": abs(plain["score"] - base["score"]),
                "finite": all(
                    math.isfinite(plain[k]) for k in ("score", "logit_0", "logit_1")
                ),
            }
        )
        print(results[-1], flush=True)
    return {
        "passed": canary_check(results, tolerance, has_adapter),
        "tolerance": tolerance,
        "rows": results,
    }


def canary(args: argparse.Namespace) -> None:
    scorer = load_scorer(args.monitor, args.device)
    receipt = {
        **run_canaries(scorer, args.monitor),
        "identity": scorer.identity(),
        "commit": git_commit(),
    }
    directory = OUTPUT / args.monitor
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "canary.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(f"canary passed={receipt['passed']}")
    if not receipt["passed"]:
        raise SystemExit(1)


def log_canaries(base: Path, check: dict[str, Any], when: str) -> None:
    with (base / "canary_runs.jsonl").open("a") as h:
        h.write(
            json.dumps(
                {**check, "time": time.time(), "commit": git_commit(), "when": when}
            )
            + "\n"
        )


def score(args: argparse.Namespace) -> None:
    from gleipnir.adversarial.scorer import TooLong

    scorer = load_scorer(args.monitor, args.device)
    prompt = CONFIG["gpu_monitors"][args.monitor]["prompt"]
    base = OUTPUT / args.monitor
    base.mkdir(parents=True, exist_ok=True)
    check = run_canaries(scorer, args.monitor)
    log_canaries(base, check, "before")
    if not check["passed"]:
        raise SystemExit("canaries failed: not scoring")
    rows = scorable(load(f"rows_{prompt}"), args.monitor)
    directory = base / "rows"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "identity.json").write_text(
        json.dumps(
            {
                "monitor": args.monitor,
                "rows": len(rows),
                "input_sha256": CONFIG["sha256"][f"rows_{prompt}"],
                "commit": git_commit(),
                "scorer": scorer.identity(),
            },
            indent=2,
        )
        + "\n"
    )
    path = directory / "predictions.jsonl"
    done = {r["id"] for r in read_jsonl(path)}
    todo = [r for r in rows if r["id"] not in done]
    started, tokens = time.time(), 0
    with path.open("a") as handle:
        for n, row in enumerate(todo, 1):
            record = record_for(row)
            try:
                record.update(scorer.score(row["prompt"]), status="ok")
                tokens += record["tokens"]
            except TooLong as error:
                record.update(status="too_long", error=str(error))
            handle.write(json.dumps(record) + "\n")
            handle.flush()
            if n % 25 == 0 or n == len(todo):
                rate = tokens / max(time.time() - started, 1e-9)
                print(f"{args.monitor}: {n}/{len(todo)}, {rate:,.0f} tok/s", flush=True)
    final = run_canaries(scorer, args.monitor)
    log_canaries(base, final, "after")
    print(f"closing canary passed={final['passed']}")
    if not final["passed"]:
        raise SystemExit(1)


def main() -> None:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("canary", "score"):
        c = commands.add_parser(name)
        c.add_argument(
            "--monitor", required=True, choices=sorted(CONFIG["gpu_monitors"])
        )
        c.add_argument("--device", default="cuda")
    args = parser.parse_args()
    {"canary": canary, "score": score}[args.command](args)


if __name__ == "__main__":
    main()
