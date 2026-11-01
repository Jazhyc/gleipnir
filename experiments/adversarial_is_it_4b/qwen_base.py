"""Post-hoc: untrained Qwen3.8-27B (Gleipnir 27B's base model) line and framing.

Scores the rows that set a 1% line and the accusing-post framing test, so the
base model can sit next to Gleipnir and Kimi in the lab figure. Rows are the Kimi
arm's teacher-prompt rows: the first 400 honest calibration rows (the 27B arm's
size) and the framing subset's `none` and `accuse_w1` conditions (100 readers
each). Kimi's request code (`kimi_record`), OpenRouter pinned to Alibaba. See
README "Post-hoc: untrained Qwen3.8-27B line and framing".

    PYTHONPATH=src:. python experiments/adversarial_is_it_4b/qwen_base.py
"""

from __future__ import annotations

import argparse
import json
import threading
from concurrent.futures import ThreadPoolExecutor

from experiments.adversarial_is_it_4b.run import (
    OUTPUT,
    arm_rows,
    git_commit,
    kimi_record,
)
from experiments.adversarial_lab_liars.common import (
    openrouter_balance,
    openrouter_key,
)
from experiments.adversarial_lab_liars.qwen_base import config
from gleipnir.adversarial import eval_sets

HONEST_ROWS = 400
CONDITIONS = ("none", "accuse_w1")
MAX_USD = 3.0


def rows_for(name: str) -> list[dict]:
    rows = arm_rows("kimi-teacher", name)
    if name == "calibration":
        return [
            r
            for r in rows
            if eval_sets.label(r) == 0
            and r["metadata"]["calibration_rank"] < HONEST_ROWS
        ]
    return [r for r in rows if r["metadata"]["condition"] in CONDITIONS]


def main() -> None:
    from gleipnir.teachers.openrouter import request_settings

    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    key = openrouter_key()  # never printed
    balance = openrouter_balance(key)
    print(f"OpenRouter balance before: ${balance:.2f}", flush=True)
    if balance < 4.0:
        raise SystemExit("balance too low: not starting")
    cfg = config()
    lock = threading.Lock()
    spent = [0.0]
    for name in ("calibration", "framing"):
        rows = rows_for(name)
        directory = OUTPUT / "qwen-base" / name
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "predictions.jsonl"
        done = set()
        if path.exists():
            for line in path.open():
                record = json.loads(line)
                spent[0] += record.get("cost") or 0.0
                if record["status"] == "ok":
                    done.add(record["id"])
        identity = {
            "arm": "qwen-base",
            "set": name,
            "rows": len(rows),
            "commit": git_commit(),
            "request_settings": request_settings(cfg),
        }
        (directory / "identity.json").write_text(json.dumps(identity, indent=2) + "\n")
        todo = [r for r in rows if r["id"] not in done][: args.limit or None]
        print(f"{name}: {len(todo)} to score, ${spent[0]:.2f} spent", flush=True)

        def work(row, path=path):
            if spent[0] >= MAX_USD:
                return None
            record = kimi_record(row, cfg, key)
            with lock:
                spent[0] += record.get("cost") or 0.0
                with path.open("a") as handle:
                    handle.write(json.dumps(record) + "\n")
            return record

        with ThreadPoolExecutor(8) as pool:
            statuses: dict[str, int] = {}
            for record in pool.map(work, todo):
                s = record["status"] if record else "skipped_cap"
                statuses[s] = statuses.get(s, 0) + 1
        print(f"  {name}: {statuses}, ${spent[0]:.2f}", flush=True)
    print(f"balance after: ${openrouter_balance(key):.2f}")


if __name__ == "__main__":
    main()
