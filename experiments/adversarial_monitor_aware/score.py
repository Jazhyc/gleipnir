"""Score the monitor-aware rows (Amendment 4).

GPU, one monitor at a time (on the box; resumable, canaries before and after,
decision-token residuals at the probe layers), modelled on adversarial_lab_liars:

    $PY experiments/adversarial_monitor_aware/score.py canary --monitor 4b
    $PY experiments/adversarial_monitor_aware/score.py score --monitor 4b
    $PY experiments/adversarial_monitor_aware/score.py score --monitor 27b

Kimi K3 through OpenRouter (on the Mac; resumable, capped, frozen order).
Prepared, not run:

    PYTHONPATH=src:. python experiments/adversarial_monitor_aware/score.py kimi
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CONFIG = json.loads((HERE / "config.json").read_text())
SCORING = CONFIG["scoring"]
DATA = ROOT / SCORING["data"]
OUTPUT = ROOT / SCORING["output"]


def sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.open()] if path.exists() else []


def git_commit() -> str:
    return subprocess.run(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True
    ).stdout.strip()


def load_rows(prompt: str) -> list[dict[str, Any]]:
    key = f"rows_{prompt}"
    path = DATA / f"{key}.jsonl"
    if file_sha256(path) != SCORING["sha256"][key]:
        raise ValueError(f"{path} does not match its frozen SHA256")
    return read_jsonl(path)


def scorable(rows: list[dict[str, Any]], monitor: str) -> list[dict[str, Any]]:
    """Rows within the token limit for `monitor`'s row set (Kimi: the 27B's)."""
    limit_of = "27b" if monitor == "kimi" else monitor
    return [r for r in rows if not r["metadata"]["too_long"][limit_of]]


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


# ---------------------------------------------------------------- GPU


def load_scorer(key: str, device: str):
    from gleipnir.adversarial.scorer import Scorer

    spec = SCORING["gpu_monitors"][key]
    return Scorer(spec["monitor"], device=device), spec


def canary_check(results: list[dict], tolerance: float, has_adapter: bool) -> bool:
    """Pure: finite, within tolerance of the known score, hooks change nothing,
    and the adapter changes the score."""
    return bool(results) and all(
        r["finite"]
        and r["hooks_identical"]
        and r["abs_diff"] <= tolerance
        and (not has_adapter or r["adapter_effect"] > 0)
        for r in results
    )


def run_canaries(scorer, spec: dict, key: str) -> dict[str, Any]:
    from experiments.adversarial_lab_liars.run import Capture

    tolerance = SCORING["canary"]["max_abs_score"]
    path = DATA / "canaries.jsonl"
    if file_sha256(path) != SCORING["sha256"]["canaries"]:
        raise ValueError("canaries.jsonl does not match its frozen SHA256")
    has_adapter = scorer.spec.adapter is not None
    results = []
    for row in (r for r in read_jsonl(path) if r["monitor"] == key):
        plain = scorer.score(row["prompt"])
        with Capture(scorer.model, spec["layers"], spec["hidden_size"]) as capture:
            hooked = scorer.score(row["prompt"])
        capture.array()
        base = scorer.score(row["prompt"], adapter=False) if has_adapter else plain
        results.append(
            {
                "id": row["id"],
                "score": plain["score"],
                "expected": row["expected"],
                "abs_diff": abs(plain["score"] - row["expected"]),
                "hooks_identical": (plain["logit_0"], plain["logit_1"])
                == (hooked["logit_0"], hooked["logit_1"]),
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
    scorer, spec = load_scorer(args.monitor, args.device)
    receipt = {
        **run_canaries(scorer, spec, args.monitor),
        "identity": scorer.identity(),
        "commit": git_commit(),
    }
    directory = OUTPUT / args.monitor
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "canary.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(f"canary passed={receipt['passed']}")
    if not receipt["passed"]:
        raise SystemExit(1)


def score(args: argparse.Namespace) -> None:
    import numpy as np

    from experiments.adversarial_lab_liars.run import Capture
    from gleipnir.adversarial.scorer import TooLong

    scorer, spec = load_scorer(args.monitor, args.device)
    base = OUTPUT / args.monitor
    base.mkdir(parents=True, exist_ok=True)
    check = run_canaries(scorer, spec, args.monitor)
    with (base / "canary_runs.jsonl").open("a") as h:
        h.write(
            json.dumps({**check, "time": time.time(), "commit": git_commit()}) + "\n"
        )
    if not check["passed"]:
        raise SystemExit("canaries failed: not scoring")
    rows = scorable(load_rows(spec["prompt"]), args.monitor)
    directory = base / "rows"
    acts = directory / "activations"
    acts.mkdir(parents=True, exist_ok=True)
    (directory / "identity.json").write_text(
        json.dumps(
            {
                "monitor": args.monitor,
                "rows": len(rows),
                "input_sha256": SCORING["sha256"][f"rows_{spec['prompt']}"],
                "commit": git_commit(),
                "layers": spec["layers"],
                "activation": "decision-token residual, block output, float32",
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
                with Capture(
                    scorer.model, spec["layers"], spec["hidden_size"]
                ) as capture:
                    record.update(scorer.score(row["prompt"]), status="ok")
                file = acts / f"{sha(row['id'])[:24]}.npy"
                np.save(file, capture.array())
                record["activation_file"] = file.name
                tokens += record["tokens"]
            except TooLong as error:
                record.update(status="too_long", error=str(error))
            handle.write(json.dumps(record) + "\n")
            handle.flush()
            if n % 25 == 0 or n == len(todo):
                rate = tokens / max(time.time() - started, 1e-9)
                print(f"{args.monitor}: {n}/{len(todo)}, {rate:,.0f} tok/s", flush=True)
    final = run_canaries(scorer, spec, args.monitor)
    with (base / "canary_runs.jsonl").open("a") as h:
        h.write(
            json.dumps(
                {**final, "time": time.time(), "commit": git_commit(), "when": "after"}
            )
            + "\n"
        )
    print(f"closing canary passed={final['passed']}")
    if not final["passed"]:
        raise SystemExit(1)


# ---------------------------------------------------------------- Kimi


def kimi_config():
    from gleipnir.teachers.openrouter import OpenRouterConfig

    spec = SCORING["kimi"]
    return OpenRouterConfig(
        model=spec["model"],
        max_tokens=spec["max_tokens"],
        top_logprobs=spec["top_logprobs"],
        provider_sort=None,
        provider_only=spec["provider_only"],
        provider_max_prompt_price=spec["max_prompt_price"],
        provider_max_completion_price=spec["max_completion_price"],
        allow_fallbacks=False,
    )


def kimi_todo(order: list[dict], done: set[str], rows: dict[str, dict]) -> list[dict]:
    """Rows still to score, in the frozen order; rows over the limit are skipped."""
    return [o for o in order if o["id"] not in done and o["id"] in rows]


def kimi(args: argparse.Namespace) -> None:
    # is_it_4b's request and parsing code, so Kimi settings match lab_liars.
    from experiments.adversarial_is_it_4b.run import kimi_record
    from experiments.adversarial_lab_liars.common import (
        openrouter_balance,
        openrouter_key,
    )
    from gleipnir.teachers.openrouter import request_settings

    spec = SCORING["kimi"]
    if file_sha256(DATA / "kimi_order.json") != SCORING["sha256"]["kimi_order"]:
        raise ValueError("kimi_order.json does not match its frozen SHA256")
    order = json.loads((DATA / "kimi_order.json").read_text())
    rows = {r["id"]: r for r in scorable(load_rows("teacher"), "kimi")}
    key = openrouter_key()
    balance = openrouter_balance(key)
    print(f"OpenRouter balance before: ${balance:.2f}", flush=True)
    if balance < spec["min_balance_usd"]:
        raise SystemExit(f"balance under ${spec['min_balance_usd']}: not starting")
    config = kimi_config()
    directory = OUTPUT / "kimi" / "rows"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "predictions.jsonl"
    done, spent = set(), [0.0]
    for r in read_jsonl(path):
        spent[0] += r.get("cost") or 0.0
        if r["status"] == "ok":
            done.add(r["id"])
    (directory / "identity.json").write_text(
        json.dumps(
            {
                "monitor": "kimi",
                "commit": git_commit(),
                "request_settings": request_settings(config),
            },
            indent=2,
        )
        + "\n"
    )
    todo = kimi_todo(order, done, rows)
    if args.limit:
        todo = todo[: args.limit]
    print(f"{len(todo)} to score, ${spent[0]:.2f} spent so far", flush=True)
    lock = threading.Lock()

    def work(item):
        if spent[0] >= spec["max_usd"]:
            return None
        row = rows[item["id"]]
        record = kimi_record(row, config, key)
        record.update({k: v for k, v in record_for(row).items() if k not in record})
        record["tier"] = item["tier"]
        with lock:
            spent[0] += record.get("cost") or 0.0
            with path.open("a") as h:
                h.write(json.dumps(record) + "\n")
        return record

    statuses: dict[str, int] = {}
    with ThreadPoolExecutor(spec["workers"]) as pool:
        for n, record in enumerate(pool.map(work, todo), 1):
            status = record["status"] if record else "skipped_cap"
            statuses[status] = statuses.get(status, 0) + 1
            if n % 50 == 0 or n == len(todo):
                print(f"  {n}/{len(todo)}, ${spent[0]:.2f}, {statuses}", flush=True)
    summary = {
        "spent": round(spent[0], 4),
        "statuses": statuses,
        "stopped_at_cap": spent[0] >= spec["max_usd"],
        "balance_before": round(balance, 2),
        "balance_after": round(openrouter_balance(key), 2),
    }
    with (OUTPUT / "kimi" / "runs.jsonl").open("a") as h:
        h.write(
            json.dumps({**summary, "time": time.time(), "commit": git_commit()}) + "\n"
        )
    print(json.dumps(summary))


def main() -> None:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    k = commands.add_parser("kimi")
    k.add_argument("--limit", type=int)
    for name in ("canary", "score"):
        c = commands.add_parser(name)
        c.add_argument(
            "--monitor", required=True, choices=sorted(SCORING["gpu_monitors"])
        )
        c.add_argument("--device", default="cuda")
    args = parser.parse_args()
    {"kimi": kimi, "canary": canary, "score": score}[args.command](args)


if __name__ == "__main__":
    main()
