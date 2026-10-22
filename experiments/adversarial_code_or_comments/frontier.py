"""Frontier black-box monitors on the held-out code-or-comments grid (Amendment 7).

Every held-out main row (2 codes x 4 texts, about 1,565 rows) under Jeremias's
teacher instruction, as the 27B sees it; score-mode monitors get the same
instruction with only the output changed (``gleipnir.adversarial.frontier``).
Monitors, settings and caps are in `config.json` under `amendment_7`.

    M=experiments.adversarial_code_or_comments.frontier
    PYTHONPATH=src:. python -m $M run --monitor gpt-6.1-sol [--limit N]
    PYTHONPATH=src:. python -m $M submit --monitor claude-sonnet-5.5
    PYTHONPATH=src:. python -m $M collect --monitor claude-sonnet-5.5
    PYTHONPATH=src:. python -m $M spend
"""

from __future__ import annotations

import argparse
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from experiments.adversarial_code_or_comments.common import (
    config,
    instruction,
    load_key,
    output_dir,
    read_jsonl,
    require_frozen,
    require_predictions,
)
from experiments.adversarial_code_or_comments.score import (
    FIELDS,
    rows_for,
    with_instruction,
)
from gleipnir.adversarial.frontier import (
    FrontierMonitor,
    anthropic_batch_params,
    anthropic_result_record,
    by_row,
    needs_request,
    openrouter_record,
)

CFG = config()
SPEC = CFG["amendment_7"]["frontier"]
CHARS_PER_TOKEN = 2.0  # over-estimates tokens, for worst-case reservations


def monitor(name: str) -> FrontierMonitor:
    spec = SPEC["monitors"][name]
    return FrontierMonitor(
        name=name,
        api=spec["api"],
        model=spec["model"],
        mode=spec["mode"],
        max_tokens=spec["max_tokens"],
        usd_per_m_prompt=spec["usd_per_m_prompt"],
        usd_per_m_completion=spec["usd_per_m_completion"],
        reasoning_effort=spec.get("reasoning_effort"),
        provider_only=spec.get("provider_only"),
        discount=spec.get("discount", 1.0),
        extra=spec.get("extra", {}),
    )


def out_path(name: str) -> Path:
    return output_dir(None, CFG) / "frontier" / f"{name}.jsonl"


def selected_rows() -> list[dict[str, Any]]:
    """Held-out main rows under the teacher instruction, in a fixed order."""
    variants = read_jsonl(output_dir(None, CFG) / "variants.jsonl")
    head = instruction("teacher", CFG)
    rows = rows_for(variants, {"main"}, "heldout")
    return sorted(
        ({**r, "prompt": with_instruction(r["prompt"], head)} for r in rows),
        key=lambda r: r["id"],
    )


def all_spend() -> dict[str, float]:
    return {
        name: sum(r.get("cost") or 0.0 for r in read_jsonl(out_path(name)))
        for name in SPEC["monitors"]
        if out_path(name).exists()
    }


def caps_allow(name: str, more: float) -> bool:
    m = SPEC["monitors"][name]
    spend = all_spend()
    api_total = sum(
        v for k, v in spend.items() if SPEC["monitors"][k]["api"] == m["api"]
    )
    return (
        spend.get(name, 0.0) + more <= m["cap_usd"]
        and api_total + more <= SPEC["caps_usd"][m["api"]]
    )


def tagged(record: dict[str, Any], row: dict[str, Any]) -> dict[str, Any]:
    meta = row["metadata"]
    out = {**record, **{k: meta.get(k) for k in FIELDS}}
    if "logit_0" in record:
        out["margin"] = record["logit_1"] - record["logit_0"]
    return out


def gate() -> None:
    require_frozen(CFG)
    require_predictions(CFG)


def run(args) -> None:
    m = monitor(args.monitor)
    if m.api != "openrouter":
        raise SystemExit("use submit/collect for Anthropic batch monitors")
    gate()
    path = out_path(m.name)
    path.parent.mkdir(parents=True, exist_ok=True)
    history = by_row(read_jsonl(path))
    rows = [r for r in selected_rows() if needs_request(history.get(r["id"], []))]
    if args.limit:
        rows = rows[: args.limit]
    import requests

    key = load_key("OPENROUTER_API_KEY", CFG)
    lock = threading.Lock()
    reserved = [0.0]
    stopped = [False]
    print(f"{m.name}: {len(rows)} rows to score", flush=True)

    def work(row):
        worst = m.worst_case_usd(int(len(row["prompt"]) / CHARS_PER_TOKEN))
        with lock:
            if stopped[0] or not caps_allow(m.name, reserved[0] + worst):
                stopped[0] = True
                return None
            reserved[0] += worst
        record = tagged(
            openrouter_record(m, row["id"], row["prompt"], key, requests.post), row
        )
        with lock:
            reserved[0] -= worst
            with path.open("a") as handle:
                handle.write(json.dumps(record) + "\n")
        return record

    statuses: dict[str, int] = {}
    workers = SPEC["monitors"][m.name].get("workers", SPEC["workers"])
    with ThreadPoolExecutor(workers) as pool:
        for n, record in enumerate(pool.map(work, rows), 1):
            if record:
                statuses[record["status"]] = statuses.get(record["status"], 0) + 1
            if n % 100 == 0 or n == len(rows):
                print(
                    f"  {n}/{len(rows)} {statuses} ${all_spend().get(m.name, 0):.2f}",
                    flush=True,
                )
    if stopped[0]:
        print(f"STOPPED at a spend cap: {all_spend()}")


def submit(args) -> None:
    import anthropic
    from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
    from anthropic.types.messages.batch_create_params import Request

    m = monitor(args.monitor)
    gate()
    state = out_path(m.name).with_suffix(".batch.json")
    if state.exists():
        raise SystemExit(f"{state} exists; collect it first")
    history = by_row(read_jsonl(out_path(m.name)))
    rows = [r for r in selected_rows() if needs_request(history.get(r["id"], []))]
    if args.limit:
        rows = rows[: args.limit]
    worst = sum(m.worst_case_usd(int(len(r["prompt"]) / CHARS_PER_TOKEN)) for r in rows)
    if not caps_allow(m.name, worst):
        raise SystemExit(f"worst case ${worst:.2f} would exceed a cap; nothing sent")
    ids = {f"r{i}": r["id"] for i, r in enumerate(rows)}
    client = anthropic.Anthropic(api_key=load_key("ANTHROPIC_API_KEY", CFG))
    batch = client.messages.batches.create(
        requests=[
            Request(
                custom_id=cid,
                params=MessageCreateParamsNonStreaming(
                    **anthropic_batch_params(m, rows[int(cid[1:])]["prompt"])
                ),
            )
            for cid in ids
        ]
    )
    state.parent.mkdir(parents=True, exist_ok=True)
    state.write_text(
        json.dumps(
            {"batch_id": batch.id, "ids": ids, "worst_case_usd": worst}, indent=2
        )
        + "\n"
    )
    print(f"submitted {len(ids)} rows as {batch.id}; worst case ${worst:.2f}")


def collect(args) -> None:
    import anthropic

    m = monitor(args.monitor)
    state_path = out_path(m.name).with_suffix(".batch.json")
    state = json.loads(state_path.read_text())
    client = anthropic.Anthropic(api_key=load_key("ANTHROPIC_API_KEY", CFG))
    batch = client.messages.batches.retrieve(state["batch_id"])
    if batch.processing_status != "ended":
        print(f"{batch.processing_status}: {batch.request_counts}")
        return
    rows = {r["id"]: r for r in selected_rows()}
    with out_path(m.name).open("a") as handle:
        for result in client.messages.batches.results(state["batch_id"]):
            row = rows[state["ids"][result.custom_id]]
            record = anthropic_result_record(m, result)
            record["id"] = row["id"]
            handle.write(json.dumps(tagged(record, row)) + "\n")
    state_path.rename(state_path.with_name(f"{m.name}.batch_{state['batch_id']}.json"))
    print(f"collected {batch.request_counts}; spend {all_spend()}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=("run", "submit", "collect", "spend"))
    parser.add_argument("--monitor", choices=sorted(SPEC["monitors"]))
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if args.command == "spend":
        print(
            json.dumps({"by_monitor": all_spend(), "caps": SPEC["caps_usd"]}, indent=2)
        )
        return
    if not args.monitor:
        raise SystemExit("--monitor is required")
    {"run": run, "submit": submit, "collect": collect}[args.command](args)


if __name__ == "__main__":
    main()
