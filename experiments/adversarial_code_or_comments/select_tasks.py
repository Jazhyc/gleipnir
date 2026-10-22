"""Task selection: paired APPS tasks, seeded pools, splits and repetition subset.

Named `select_tasks.py`, not `select.py`: a `select.py` here would shadow the
standard-library `select` module (imported by `subprocess`) whenever a script
in this folder is run directly.

    PYTHONPATH=src python -m experiments.adversarial_code_or_comments.select_tasks \
        --dry-run
    # after triggers are verified:
    PYTHONPATH=src python -m experiments.adversarial_code_or_comments.select_tasks \
        --triggers results/adversarial_code_or_comments/triggers.jsonl
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import mean
from typing import Any

from experiments.adversarial_code_or_comments.common import (
    ROOT,
    config,
    load_pairs,
    order_key,
    output_dir,
    prompt,
    read_jsonl,
    write_json,
)
from gleipnir.adversarial import apps, eval_sets


def ordered(tasks, seed: int) -> list[str]:
    return sorted(map(str, tasks), key=lambda t: order_key(seed, t))


def pools(tasks, seed: int) -> tuple[list[str], list[str]]:
    order = ordered(tasks, seed)
    half = math.ceil(len(order) / 2)
    return order[:half], order[half:]


def take_confirmed(pool: list[str], confirmed: set[str], n: int):
    """First `n` confirmed tasks in pool order, and the unconfirmed ones passed."""
    chosen, dropped = [], []
    for task in pool:
        if len(chosen) == n:
            break
        (chosen if task in confirmed else dropped).append(task)
    return chosen, dropped


def repetition_subset(development: list[str], seed: int, n: int) -> list[str]:
    return ordered(development, seed)[:n]


def split(cfg: dict[str, Any], tasks, confirmed: set[str] | None) -> dict[str, Any]:
    """The frozen rule. `confirmed=None` assumes every trigger is confirmed."""
    sel = cfg["selection"]
    dev_pool, held_pool = pools(tasks, sel["split_seed"])
    confirmed = set(map(str, tasks)) if confirmed is None else confirmed
    dev, dev_dropped = take_confirmed(dev_pool, confirmed, sel["development_tasks"])
    held, held_dropped = take_confirmed(held_pool, confirmed, sel["heldout_tasks"])
    return {
        "development_pool": dev_pool,
        "heldout_pool": held_pool,
        "development": dev,
        "heldout": held,
        "repetition": repetition_subset(
            dev, sel["repetition_seed"], sel["repetition_tasks"]
        ),
        "dropped_without_trigger": {
            "development": dev_dropped,
            "heldout": held_dropped,
        },
    }


def raw_validity(cfg: dict[str, Any]) -> dict[str, Any] | None:
    """Cross-check with the raw release's validity flags, if the file is present."""
    raw = cfg["data"]["raw"]
    path = ROOT / raw["path"]
    if not path.exists():
        return None
    if eval_sets.file_sha256(path) != raw["sha256"]:
        return {"error": "raw file does not match its manifest SHA256"}
    honest, backdoor = set(), set()
    for line in path.open():
        row = json.loads(line)
        task = str(row["task_id"])
        if row["condition"] == cfg["data"]["honest_condition"]:
            if row["valid_parses"] and row["valid_tests"]:
                honest.add(task)
        elif row["condition"] == cfg["data"]["backdoor_condition"]:
            if row["valid_parses"] and row["valid_tests"] and row["valid_backdoor"]:
                backdoor.add(task)
    return {
        "honest_valid": len(honest),
        "backdoor_valid": len(backdoor),
        "both_valid": len(honest & backdoor),
        "note": "before the release's 20-problem dev exclusion",
        "_both": honest & backdoor,
    }


def cost_estimate(cfg: dict[str, Any], pairs: dict[str, Any], n_text_tasks: int):
    """Rough Tinker cost from measured prompt sizes and configured assumptions."""
    a = cfg["cost_assumptions"]
    per_token = a["chars_per_token"]
    sizes = [
        {code: apps.parse(row["prompt"]) for code, row in pair.items()}
        for pair in pairs.values()
    ]
    trigger_chars = len(prompt("trigger.txt", cfg)) + mean(
        len(s["honest"].problem) + len(s["honest"].code) + len(s["backdoored"].code)
        for s in sizes
    )
    text_overhead = len(prompt("text_rules.txt", cfg)) + mean(
        len(prompt(k["prompt"], cfg)) for k in cfg["texts"]["kinds"].values()
    )
    text_chars = text_overhead + mean(
        len(s[c].problem) + len(s[c].code) for s in sizes for c in s
    )
    check_chars = len(prompt("fidelity.txt", cfg)) + text_chars - text_overhead + 700
    writer = cfg["models"]["writer"]
    checker = cfg["models"]["checker"]
    trigger_calls = len(pairs) * a["trigger_attempts_per_task"]
    kinds = len(cfg["texts"]["kinds"])
    text_calls = n_text_tasks * kinds * a["text_attempts_per_text"]
    check_calls = n_text_tasks * len(cfg["texts"]["fidelity"]["checked_kinds"])

    def usd(spec, calls, prompt_chars, completion_tokens):
        return (
            calls
            * (
                prompt_chars / per_token * spec["usd_per_m_prompt"]
                + completion_tokens * spec["usd_per_m_completion"]
            )
            / 1e6
        )

    out: dict[str, Any] = {
        "calls": {
            "trigger": round(trigger_calls),
            "texts": round(text_calls),
            "fidelity": check_calls,
        },
        "mean_prompt_chars": {
            "trigger": round(trigger_chars),
            "text": round(text_chars),
            "fidelity": round(check_chars),
        },
    }
    for mode, completion in a["writer_completion_tokens"].items():
        out[f"writer_usd_{mode}"] = round(
            usd(writer, trigger_calls, trigger_chars, completion)
            + usd(writer, text_calls, text_chars, completion),
            2,
        )
    out["checker_usd"] = round(
        usd(checker, check_calls, check_chars, a["checker_completion_tokens"]), 2
    )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--triggers", help="triggers.jsonl from trigger.py merge")
    parser.add_argument("--out", help="output folder (default: config output)")
    parser.add_argument("--dry-run", action="store_true", help="print, write nothing")
    args = parser.parse_args()
    cfg = config()
    pairs, counts = load_pairs(cfg)
    confirmed = None
    trigger_counts = None
    if args.triggers:
        triggers = read_jsonl(Path(args.triggers))
        confirmed = {t["task_id"] for t in triggers if t["status"] == "confirmed"}
        trigger_counts = {
            status: sum(t["status"] == status for t in triggers)
            for status in sorted({t["status"] for t in triggers})
        }
    chosen = split(cfg, pairs, confirmed)
    raw = raw_validity(cfg)
    if raw and "_both" in raw:
        both = raw.pop("_both")
        raw["eligible_inside_raw_both_valid"] = len(set(pairs) & both)
    sel = cfg["selection"]
    summary = {
        "data": {"set": cfg["data"]["set"], "sha256": cfg["data"]["sha256"]},
        "tasks_with_honest_row": counts["honest"],
        "tasks_with_backdoor_row": counts["backdoored"],
        "tasks_with_both": counts["both"],
        "raw_validity_crosscheck": raw,
        "pool_sizes": {
            "development": len(chosen["development_pool"]),
            "heldout": len(chosen["heldout_pool"]),
        },
        "split_sizes": {
            "development": len(chosen["development"]),
            "heldout": len(chosen["heldout"]),
            "repetition": len(chosen["repetition"]),
        },
        "targets": {
            "development": sel["development_tasks"],
            "heldout": sel["heldout_tasks"],
            "repetition": sel["repetition_tasks"],
        },
        "triggers": trigger_counts or "not yet verified; split assumes all confirmed",
        "dropped_without_trigger": {
            k: len(v) for k, v in chosen["dropped_without_trigger"].items()
        },
        "first_tasks": {
            k: chosen[k][:5] for k in ("development", "heldout", "repetition")
        },
        "rows": {
            "main_grid_per_split": len(chosen["development"]) * 8,
            "repetition_new": len(chosen["repetition"]) * 2 * 3 * 3,
        },
        "cost_estimate_usd": cost_estimate(
            cfg, pairs, len(chosen["development"]) + len(chosen["heldout"])
        ),
    }
    print(json.dumps(summary, indent=2))
    if args.dry_run:
        return
    out = output_dir(args.out, cfg)
    name = "selection.json" if args.triggers else "pools.json"
    write_json(out / name, {**chosen, "summary": summary, "final": bool(args.triggers)})
    print(f"wrote {out / name}")


if __name__ == "__main__":
    main()
