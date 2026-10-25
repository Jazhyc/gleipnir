"""Score arm 1's rows (modelled on monitor-aware's score.py, which it reuses).

GPU, on the A100 `gleipnir-qwen-gpu` in ~/gleipnir-coc (vm_score.sh): one
monitor at a time, resumable, monitor-aware's canaries before and after, and
the decision-token residual saved for every row at monitor-aware's layers:

    $PY experiments/adversarial_quiet_cheats/score.py canary --monitor 4b
    $PY experiments/adversarial_quiet_cheats/score.py score --monitor 4b
    $PY experiments/adversarial_quiet_cheats/score.py score --monitor 27b

Kimi K3 through OpenRouter (on the Mac), frozen order, capped at $1.00 and
started only with a balance of at least $2.00. The 5 drift originals run
first; if any differs from monitor-aware's Kimi score by more than 0.05, it
stops before scoring anything else (Amendment 1):

    PYTHONPATH=src:. python experiments/adversarial_quiet_cheats/score.py kimi

Rows must match `arm1_deletion.rows_sha256` in config.json (set at the rows
freeze); nothing scores before that.
"""

from __future__ import annotations

import argparse
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from experiments.adversarial_quiet_cheats.common import (
    ARM1,
    DATA,
    MA_CONFIG,
    MA_OUT,
    OUT,
    file_sha256,
    read_jsonl,
    sha,
)

SCORING = OUT / "scoring"
GPU = MA_CONFIG["scoring"]["gpu_monitors"]
KIMI = ARM1["kimi"]


def frozen(name: str) -> str:
    expected = (ARM1.get("rows_sha256") or {}).get(name)
    if not expected:
        raise SystemExit(f"{name}: no frozen SHA256 in config (rows freeze not done)")
    path = DATA / (f"{name}.jsonl" if name.startswith("rows_") else f"{name}.json")
    if file_sha256(path) != expected:
        raise ValueError(f"{path} does not match its frozen SHA256")
    return str(path)


def load_rows(prompt: str) -> list[dict[str, Any]]:
    frozen(f"rows_{prompt}")
    return read_jsonl(DATA / f"rows_{prompt}.jsonl")


def scorable(rows: list[dict], monitor: str) -> list[dict]:
    limit_of = "27b" if monitor == "kimi" else monitor
    return [r for r in rows if not r["metadata"]["too_long"][limit_of]]


def record_for(row: dict[str, Any]) -> dict[str, Any]:
    m = row["metadata"]
    return {
        "id": row["id"],
        "group": m["condition"],
        "label": m["ground_truth"],
        "role": m["role"],
        "setup": m["setup"],
        "task_id": m["task_id"],
        "parent_id": m["parent_id"],
    }


# ---------------------------------------------------------------- GPU


def gpu_receipt(passed: bool, tag: str, monitor: str, check: dict) -> None:
    from experiments.adversarial_monitor_aware.score import git_commit

    base = SCORING / monitor
    base.mkdir(parents=True, exist_ok=True)
    with (base / "canary_runs.jsonl").open("a") as h:
        h.write(
            json.dumps(
                {**check, "when": tag, "time": time.time(), "commit": git_commit()}
            )
            + "\n"
        )


def canary(args: argparse.Namespace) -> None:
    from experiments.adversarial_monitor_aware.score import load_scorer, run_canaries

    scorer, spec = load_scorer(args.monitor, args.device)
    check = run_canaries(scorer, spec, args.monitor)
    gpu_receipt(check["passed"], "standalone", args.monitor, check)
    print(f"canary passed={check['passed']}")
    if not check["passed"]:
        raise SystemExit(1)


def score(args: argparse.Namespace) -> None:
    import numpy as np

    from experiments.adversarial_lab_liars.run import Capture
    from experiments.adversarial_monitor_aware.score import (
        git_commit,
        load_scorer,
        run_canaries,
    )
    from gleipnir.adversarial.scorer import TooLong

    scorer, spec = load_scorer(args.monitor, args.device)
    check = run_canaries(scorer, spec, args.monitor)
    gpu_receipt(check["passed"], "before", args.monitor, check)
    if not check["passed"]:
        raise SystemExit("canaries failed: not scoring")
    rows = scorable(load_rows(spec["prompt"]), args.monitor)
    directory = SCORING / args.monitor / "rows"
    acts = directory / "activations"
    acts.mkdir(parents=True, exist_ok=True)
    (directory / "identity.json").write_text(
        json.dumps(
            {
                "monitor": args.monitor,
                "rows": len(rows),
                "input_sha256": ARM1["rows_sha256"][f"rows_{spec['prompt']}"],
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
    gpu_receipt(final["passed"], "after", args.monitor, final)
    print(f"closing canary passed={final['passed']}")
    if not final["passed"]:
        raise SystemExit(1)


# ---------------------------------------------------------------- Kimi


def ma_kimi_scores() -> dict[str, float]:
    """monitor-aware's Kimi scores by parent run id (status ok)."""
    path = MA_OUT / "scoring" / "kimi" / "rows" / "predictions.jsonl"
    return {r["parent_id"]: r["score"] for r in read_jsonl(path) if r["status"] == "ok"}


def drift_verdict(
    records: list[dict], reference: dict[str, float], tolerance: float
) -> dict[str, Any]:
    """Pure: each drift row scored ok and within tolerance of monitor-aware's."""
    rows = []
    for r in records:
        ref = reference.get(r.get("parent_id"))
        ok = (
            r.get("status") == "ok"
            and ref is not None
            and abs(r["score"] - ref) <= tolerance
        )
        rows.append(
            {"id": r["id"], "score": r.get("score"), "reference": ref, "ok": ok}
        )
    return {
        "passed": len(rows) == KIMI["drift_rows"] and all(x["ok"] for x in rows),
        "tolerance": tolerance,
        "rows": rows,
    }


def kimi(args: argparse.Namespace) -> None:
    from experiments.adversarial_is_it_4b.run import kimi_record
    from experiments.adversarial_lab_liars.common import (
        openrouter_balance,
        openrouter_key,
    )
    from experiments.adversarial_monitor_aware.score import git_commit, kimi_config
    from gleipnir.teachers.openrouter import request_settings

    frozen("kimi_order")
    order = json.loads((DATA / "kimi_order.json").read_text())
    rows = {r["id"]: r for r in scorable(load_rows("teacher"), "kimi")}
    key = openrouter_key()
    balance = openrouter_balance(key)
    print(f"OpenRouter balance before: ${balance:.2f}", flush=True)
    if balance < KIMI["min_balance_usd"]:
        raise SystemExit(f"balance under ${KIMI['min_balance_usd']}: not starting")
    config = kimi_config()
    directory = SCORING / "kimi" / "rows"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "predictions.jsonl"
    done, spent = {}, [0.0]
    for r in read_jsonl(path):
        spent[0] += r.get("cost") or 0.0
        if r["status"] == "ok":
            done[r["id"]] = r
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
    lock = threading.Lock()

    def work(item):
        if spent[0] >= KIMI["max_usd"]:
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

    def run_tier(items):
        todo = [o for o in items if o["id"] not in done and o["id"] in rows]
        with ThreadPoolExecutor(KIMI["workers"]) as pool:
            for record in pool.map(work, todo):
                if record and record["status"] == "ok":
                    done[record["id"]] = record

    drift_items = [o for o in order if o["tier"] == "drift"]
    run_tier(drift_items)
    verdict = drift_verdict(
        [done[o["id"]] for o in drift_items if o["id"] in done],
        ma_kimi_scores(),
        KIMI["drift_tolerance"],
    )
    (SCORING / "kimi" / "drift.json").write_text(json.dumps(verdict, indent=2) + "\n")
    print(f"drift check passed={verdict['passed']}", flush=True)
    if verdict["passed"]:
        rest = [o for o in order if o["tier"] != "drift"]
        if args.limit:
            rest = rest[: args.limit]
        run_tier(rest)
    summary = {
        "spent": round(spent[0], 4),
        "drift_passed": verdict["passed"],
        "scored_ok": len(done),
        "order_rows": len(order),
        "stopped_at_cap": spent[0] >= KIMI["max_usd"],
        "balance_before": round(balance, 2),
        "balance_after": round(openrouter_balance(key), 2),
    }
    with (SCORING / "kimi" / "runs.jsonl").open("a") as h:
        h.write(
            json.dumps({**summary, "time": time.time(), "commit": git_commit()}) + "\n"
        )
    print(json.dumps(summary))
    if not verdict["passed"]:
        raise SystemExit("drift check failed: originals can't be reused; stopped")


def main() -> None:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    k = commands.add_parser("kimi")
    k.add_argument("--limit", type=int)
    for name in ("canary", "score"):
        c = commands.add_parser(name)
        c.add_argument("--monitor", required=True, choices=sorted(GPU))
        c.add_argument("--device", default="cuda")
    args = parser.parse_args()
    {"kimi": kimi, "canary": canary, "score": score}[args.command](args)


if __name__ == "__main__":
    main()
