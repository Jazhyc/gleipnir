"""Arm 1 monitor rows, after the checks (needs transformers for token counts).

    PY=../../.venv-data/bin/python
    PYTHONPATH=src:. $PY experiments/adversarial_quiet_cheats/rows.py

Rows, rendered as monitor-aware (`trajectories.render`), under the student
prompt (4B) and the teacher prompt (27B, Kimi):
- `quiet`: every valid quiet version (checks.py), at its final attempt;
- `original`: the 37 cheats at monitor-aware's cut, byte-identical to
  monitor-aware's rows (checked);
- `strip`: every valid comment-strip control;
- `strip_original`: those controls' monitor-aware rows.

Also writes `kimi_order.json`: 5 drift originals (drawn with the config seed
from originals the 27B can score) first, then quiet rows, then strip rows,
each tier shuffled with seed 0. Kimi reuses monitor-aware's scores for the
originals and control originals (Amendment 1). The SHA256 values printed here
go into `config.json` (`arm1_deletion.rows_sha256`) in the rows-freeze
amendment, before any score; `score.py` refuses rows that don't match.
"""

from __future__ import annotations

import json
import random
from collections import Counter
from typing import Any

from experiments.adversarial_quiet_cheats.common import (
    ARM1,
    DATA,
    file_sha256,
    load_ma_rows,
    load_runs,
    read_jsonl,
    write_jsonl,
)

ROLES = ("quiet", "original", "strip", "strip_original")


def final_items(set_name: str) -> list[dict[str, Any]]:
    """Valid items with their replayed messages (latest attempt per id)."""
    checks = {r["id"]: r for r in read_jsonl(DATA / f"{set_name}_checks.jsonl")}
    replays = {
        (r["id"], r["attempt"]): r
        for r in read_jsonl(DATA / f"{set_name}_replay.jsonl")
    }
    out = []
    for item in read_jsonl(DATA / f"{set_name}.jsonl"):
        c = checks.get(item["id"])
        if c is None or not c["valid"] or c["attempt"] != item.get("attempt", 1):
            continue
        rp = replays[(item["id"], item.get("attempt", 1))]
        out.append({**item, "messages": rp["messages"], "check": c})
    return out


def meta(run: dict, role: str, row_id: str, extra: dict) -> dict[str, Any]:
    cheat = role in ("quiet", "original")
    return {
        "source_dataset": "adversarial_quiet_cheats",
        "arm": 1,
        "role": role,
        "parent_id": run["id"],
        "task_id": run["task"],
        "setup": run["setup"],
        "cheat": cheat,
        "ground_truth": int(cheat),
        "condition": f"arm1:{role}",
        **extra,
    }


def kimi_order(
    rows: list[dict], seed: int, drift_n: int, drift_seed: int
) -> list[dict]:
    originals = sorted(
        r["id"]
        for r in rows
        if r["meta"]["role"] == "original" and not r["meta"]["too_long"]["27b"]
    )
    drift = random.Random(drift_seed).sample(originals, drift_n)
    rng = random.Random(seed)
    out = [{"id": i, "tier": "drift"} for i in drift]
    for tier in ("quiet", "strip"):
        ids = sorted(
            r["id"]
            for r in rows
            if r["meta"]["role"] == tier and not r["meta"]["too_long"]["27b"]
        )
        rng.shuffle(ids)
        out += [{"id": i, "tier": tier} for i in ids]
    return out


def main() -> None:
    from experiments.adversarial_monitor_aware.rows import instructions, token_counters
    from experiments.adversarial_natural_cheats.trajectories import render
    from gleipnir.adversarial.contract import MAX_TOKENS, join_prompt

    runs = {r["id"]: r for r in load_runs()}
    ma = {"student": load_ma_rows("student"), "teacher": load_ma_rows("teacher")}
    texts, counters = instructions(), token_counters()
    key = json.loads((DATA / "key_PRIVATE.json").read_text())["units"]
    rows = []
    for item in final_items("quiet"):
        run = runs[item["run_id"]]
        extra = {
            "tier": item["check"]["tier"],
            "attempt": item["attempt"],
            "cut_message_index": item["cut"],
        }
        rows.append(
            {
                "id": item["id"],
                "trajectory": render(item["messages"]),
                "meta": meta(run, "quiet", item["id"], extra),
            }
        )
    for k in key:
        run = runs[k["run_id"]]
        rid = f"{run['id']}::arm1_original"
        rows.append(
            {
                "id": rid,
                "trajectory": render(run["messages"][: k["cut"] + 1]),
                "meta": meta(run, "original", rid, {"cut_message_index": k["cut"]}),
            }
        )
    for item in final_items("strip"):
        run = runs[item["run_id"]]
        rows.append(
            {
                "id": item["id"],
                "trajectory": render(item["messages"]),
                "meta": meta(
                    run, "strip", item["id"], {"cut_message_index": item["cut"]}
                ),
            }
        )
        rid = f"{run['id']}::arm1_strip_original"
        rows.append(
            {
                "id": rid,
                "trajectory": render(run["messages"][: item["cut"] + 1]),
                "meta": meta(
                    run, "strip_original", rid, {"cut_message_index": item["cut"]}
                ),
            }
        )
    for row in rows:
        m = row["meta"]
        if m["role"] in ("original", "strip_original"):
            for prompt in ("student", "teacher"):
                if (
                    join_prompt(texts[prompt], row["trajectory"])
                    != ma[prompt][m["parent_id"]]["prompt"]
                ):
                    raise ValueError(
                        f"{row['id']}: not byte-identical to monitor-aware's row"
                    )
        tokens = {
            k: count(join_prompt(texts[p], row["trajectory"]))
            for k, (p, count) in counters.items()
        }
        m["tokens"] = tokens
        m["too_long"] = {k: n > MAX_TOKENS for k, n in tokens.items()}
    digests = {}
    for prompt in ("student", "teacher"):
        digests[f"rows_{prompt}"] = write_jsonl(
            DATA / f"rows_{prompt}.jsonl",
            [
                {
                    "id": r["id"],
                    "prompt": join_prompt(texts[prompt], r["trajectory"]),
                    "metadata": r["meta"],
                }
                for r in rows
            ],
        )
    order = kimi_order(
        rows,
        ARM1["kimi"]["order_seed"],
        ARM1["kimi"]["drift_rows"],
        ARM1["kimi"]["drift_seed"],
    )
    (DATA / "kimi_order.json").write_text(json.dumps(order, indent=1) + "\n")
    digests["kimi_order"] = file_sha256(DATA / "kimi_order.json")
    manifest = {
        "rows": len(rows),
        "per_role": dict(Counter(r["meta"]["role"] for r in rows)),
        "too_long": {
            k: dict(
                Counter(r["meta"]["role"] for r in rows if r["meta"]["too_long"][k])
            )
            for k in counters
        },
        "tiers": dict(
            Counter(r["meta"].get("tier") for r in rows if r["meta"]["role"] == "quiet")
        ),
        "tokens_27b_total": sum(r["meta"]["tokens"]["27b"] for r in rows),
        "kimi_rows": len(order),
        "kimi_tokens_27b": sum(
            next(r["meta"]["tokens"]["27b"] for r in rows if r["id"] == o["id"])
            for o in order
        ),
        "sha256": digests,
    }
    (DATA / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
