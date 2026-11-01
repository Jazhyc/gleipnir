"""Post-hoc: score the lab-liar rows with untrained Qwen3.8-27B (27B base model).

Same teacher-prompt rows, frozen order and request code as the Kimi run; tiers 0-2 of
kimi_order.json only (every recruiter and swap row). See README "Post-hoc: untrained
Qwen3.8-27B". Writes results/adversarial_lab_liars/qwen_base/<set>/predictions.jsonl.

    PYTHONPATH=src:. python experiments/adversarial_lab_liars/qwen_base.py [--limit N]
"""

from __future__ import annotations

import argparse
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor

from experiments.adversarial_lab_liars.common import (
    DATA,
    OUTPUT,
    openrouter_balance,
    openrouter_key,
    read_jsonl,
)
from experiments.adversarial_lab_liars.run import (
    SETS,
    git_commit,
    load_set,
    record_for,
    scorable,
)

SPEC = {
    "model": "qwen/qwen3.8-27b",
    "provider_only": "alibaba",
    "max_tokens": 64,
    "top_logprobs": 5,
    "max_prompt_price": 0.5,
    "max_completion_price": 3.0,
    "workers": 8,
    "max_tier": 2,
    "max_usd": 3.0,
    "min_balance_usd": 5.0,
}


def config():
    from gleipnir.teachers.openrouter import OpenRouterConfig

    return OpenRouterConfig(
        model=SPEC["model"],
        max_tokens=SPEC["max_tokens"],
        top_logprobs=SPEC["top_logprobs"],
        provider_sort=None,
        provider_only=SPEC["provider_only"],
        provider_max_prompt_price=SPEC["max_prompt_price"],
        provider_max_completion_price=SPEC["max_completion_price"],
        allow_fallbacks=False,
    )


def main() -> None:
    from experiments.adversarial_is_it_4b.run import kimi_record
    from gleipnir.teachers.openrouter import request_settings

    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    key = openrouter_key()  # never printed
    balance = openrouter_balance(key)
    print(f"OpenRouter balance before: ${balance:.2f}", flush=True)
    if balance < SPEC["min_balance_usd"]:
        raise SystemExit("balance too low: not starting")
    order = [
        o
        for o in json.loads((DATA / "kimi_order.json").read_text())
        if o["tier"] <= SPEC["max_tier"]
    ]
    rows = {(n, r["id"]): r for n in SETS for r in scorable(load_set(n, "teacher"))}
    cfg = config()
    done, spent = set(), [0.0]
    for name in SETS:
        directory = OUTPUT / "qwen_base" / name
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "predictions.jsonl"
        path.touch()
        for r in read_jsonl(path):
            spent[0] += r.get("cost") or 0.0
            if r["status"] == "ok":
                done.add((name, r["id"]))
        identity = {
            "monitor": "qwen_base",
            "set": name,
            "commit": git_commit(),
            "request_settings": request_settings(cfg),
        }
        (directory / "identity.json").write_text(json.dumps(identity, indent=2) + "\n")
    todo = [o for o in order if (o["set"], o["id"]) not in done]
    if args.limit:
        todo = todo[: args.limit]
    print(f"{len(todo)} to score, ${spent[0]:.2f} spent so far", flush=True)
    lock = threading.Lock()

    def work(item):
        if spent[0] >= SPEC["max_usd"]:
            return None
        row = rows[(item["set"], item["id"])]
        record = kimi_record(row, cfg, key)
        record.update({k: v for k, v in record_for(row).items() if k not in record})
        record["tier"] = item["tier"]
        with lock:
            spent[0] += record.get("cost") or 0.0
            out = OUTPUT / "qwen_base" / item["set"] / "predictions.jsonl"
            with out.open("a") as h:
                h.write(json.dumps(record) + "\n")
        return record

    statuses: dict[str, int] = {}
    with ThreadPoolExecutor(SPEC["workers"]) as pool:
        for n, record in enumerate(pool.map(work, todo), 1):
            status = record["status"] if record else "skipped_cap"
            statuses[status] = statuses.get(status, 0) + 1
            if n % 50 == 0 or n == len(todo):
                print(f"  {n}/{len(todo)}, ${spent[0]:.2f}, {statuses}", flush=True)
    summary = {
        "spent": round(spent[0], 4),
        "statuses": statuses,
        "stopped_at_cap": spent[0] >= SPEC["max_usd"],
        "balance_before": round(balance, 2),
        "balance_after": round(openrouter_balance(key), 2),
        "time": time.time(),
        "commit": git_commit(),
    }
    with (OUTPUT / "qwen_base" / "runs.jsonl").open("a") as h:
        h.write(json.dumps(summary) + "\n")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
