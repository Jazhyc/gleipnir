"""Frontier black-box monitors on the natural-cheat rows (Amendment 2).

Each monitor scores the 27B's teacher-prompt rows at the first-cheat and
comments-stripped cuts, the same rows the 27B scored (rows too long for the
27B are skipped for every monitor). Settings, caps and the row order are in
`config.json` under `amendment_2.frontier`. Runs only from a committed tree.

    M=experiments.adversarial_natural_cheats.frontier
    PYTHONPATH=src:. python -m $M run --monitor kimi-k3 [--limit N] [--dry-run]
    PYTHONPATH=src:. python -m $M submit --monitor claude-sonnet-5.5   # Anthropic batch
    PYTHONPATH=src:. python -m $M collect --monitor claude-sonnet-5.5
    PYTHONPATH=src:. python -m $M spend
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from experiments.adversarial_natural_cheats.analyze import load_rows, read_predictions
from gleipnir.adversarial import eval_sets
from gleipnir.adversarial.frontier import (
    FrontierMonitor,
    anthropic_batch_params,
    anthropic_result_record,
    openrouter_record,
)

HERE = Path(__file__).resolve().parent
CONFIG = json.loads((HERE / "config.json").read_text())
ROOT = eval_sets.ROOT
OUTPUT = ROOT / CONFIG["output"]
SPEC = CONFIG["amendment_2"]["frontier"]
CHARS_PER_TOKEN = 2.0  # below the measured 2.47, so reservations over-estimate


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


def predictions_path(name: str) -> Path:
    return OUTPUT / name / "rows" / "predictions.jsonl"


def selected_rows() -> list[dict[str, Any]]:
    """Teacher rows at the frontier cuts that the 27B scored, in fixed order."""
    scored = {
        r["id"]
        for r in read_predictions(predictions_path(SPEC["row_set_scored_by"]))
        if r.get("status") == "ok"
    }
    order = {cut: i for i, cut in enumerate(SPEC["cuts"])}
    rows = [
        r
        for r in load_rows("rows", SPEC["row_set_from"])
        if r["metadata"]["detection_point"] in order and r["id"] in scored
    ]
    return sorted(
        rows, key=lambda r: (order[r["metadata"]["detection_point"]], r["id"])
    )


def spent(records: list[dict[str, Any]]) -> float:
    return sum(r.get("cost") or 0.0 for r in records)


def all_spend() -> dict[str, float]:
    out = {}
    for name in SPEC["monitors"]:
        path = predictions_path(name)
        if path.exists():
            out[name] = spent([json.loads(x) for x in path.open() if x.strip()])
    return out


def api_spend(api: str) -> float:
    return sum(v for k, v in all_spend().items() if SPEC["monitors"][k]["api"] == api)


def load_key(name: str) -> str:
    """From the environment or the wrapper's git-ignored .env; never printed."""
    if os.environ.get(name):
        return os.environ[name]
    env = (ROOT / SPEC["env_file"]).resolve()
    for line in env.read_text().splitlines():
        if line.startswith(name + "="):
            return line.split("=", 1)[1].strip().strip('"').strip("'")
    raise SystemExit(f"{name} missing from the environment and {env.name}")


def require_committed() -> None:
    out = subprocess.run(
        [
            "git",
            "status",
            "--porcelain",
            "--",
            "experiments/adversarial_natural_cheats",
            "src/gleipnir/adversarial",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    if out.stdout.strip():
        raise SystemExit("uncommitted protocol changes; commit them before a paid run")


def caps_allow(name: str, more: float) -> bool:
    m = SPEC["monitors"][name]
    cap = SPEC["caps_usd"]
    own = all_spend().get(name, 0.0)
    return own + more <= m["cap_usd"] and api_spend(m["api"]) + more <= cap[m["api"]]


class DryResponse:
    status_code = 200
    text = ""

    def __init__(self, body: dict[str, Any]):
        self.body = body

    def json(self):
        return self.body


def dry_post(url, headers=None, json=None, timeout=None):  # noqa: A002
    """Network-free stand-in: a fixed score, or fixed logprobs."""
    if json.get("logprobs"):
        tokens = [
            {
                "token": t,
                "logprob": -0.1,
                "top_logprobs": [
                    {"token": "1", "logprob": -0.7},
                    {"token": "0", "logprob": -0.7},
                ],
            }
            for t in ("Prediction", ":", "1")
        ]
        message = {"content": "Prediction:1"}
        logprobs = {"content": tokens}
    else:
        message, logprobs = {"content": "Score:50"}, None
    return DryResponse(
        {
            "choices": [
                {"message": message, "finish_reason": "stop", "logprobs": logprobs}
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 3, "cost": 0.0},
            "model": "dry-run",
        }
    )


def run(args) -> None:
    m = monitor(args.monitor)
    if m.api != "openrouter":
        raise SystemExit("use submit/collect for Anthropic batch monitors")
    if m.name in SPEC.get("deferred", []) and not args.force:
        raise SystemExit(
            f"{m.name} is deferred (Amendment 2); run it only after a decision"
        )
    if args.dry_run and not args.out:
        raise SystemExit("dry runs need --out, so they never write into results/")
    if not args.dry_run:
        require_committed()
    path = (
        predictions_path(m.name)
        if not args.out
        else Path(args.out) / "predictions.jsonl"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    done = {r["id"] for r in read_predictions(path) if r.get("status") == "ok"}
    rows = [r for r in selected_rows() if r["id"] not in done]
    if args.limit:
        rows = rows[: args.limit]
    key = "dry-run" if args.dry_run else load_key("OPENROUTER_API_KEY")
    if args.dry_run:
        post = dry_post
    else:
        import requests

        post = requests.post
    lock = threading.Lock()
    reserved = [0.0]
    stopped = [False]
    print(f"{m.name}: {len(rows)} rows to score", flush=True)

    def work(row):
        worst = m.worst_case_usd(int(len(row["prompt"]) / CHARS_PER_TOKEN))
        with lock:
            if stopped[0] or not (
                args.dry_run or caps_allow(m.name, reserved[0] + worst)
            ):
                stopped[0] = True
                return None
            reserved[0] += worst
        record = openrouter_record(m, row["id"], row["prompt"], key, post)
        record["detection_point"] = row["metadata"]["detection_point"]
        with lock:
            reserved[0] -= worst
            with path.open("a") as handle:
                handle.write(json.dumps(record) + "\n")
        return record

    statuses: dict[str, int] = {}
    with ThreadPoolExecutor(SPEC["workers"]) as pool:
        for n, record in enumerate(pool.map(work, rows), 1):
            if record:
                statuses[record["status"]] = statuses.get(record["status"], 0) + 1
            if n % 50 == 0 or n == len(rows):
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
    require_committed()
    state = predictions_path(m.name).parent / "batch.json"
    if state.exists() and not args.force:
        raise SystemExit(f"{state} exists; collect it first or pass --force")
    done = {
        r["id"]
        for r in read_predictions(predictions_path(m.name))
        if r.get("status") == "ok"
    }
    rows = [r for r in selected_rows() if r["id"] not in done]
    if args.limit:
        rows = rows[: args.limit]
    worst = sum(m.worst_case_usd(int(len(r["prompt"]) / CHARS_PER_TOKEN)) for r in rows)
    if not caps_allow(m.name, worst):
        raise SystemExit(f"worst case ${worst:.2f} would exceed a cap; nothing sent")
    # custom_id allows only [a-zA-Z0-9_-]{1,64}; map ids to positions.
    ids = {f"r{i}": r["id"] for i, r in enumerate(rows)}
    client = anthropic.Anthropic(api_key=load_key("ANTHROPIC_API_KEY"))
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
    state_path = predictions_path(m.name).parent / "batch.json"
    state = json.loads(state_path.read_text())
    client = anthropic.Anthropic(api_key=load_key("ANTHROPIC_API_KEY"))
    batch = client.messages.batches.retrieve(state["batch_id"])
    if batch.processing_status != "ended":
        print(f"{batch.processing_status}: {batch.request_counts}")
        return
    meta = {r["id"]: r["metadata"]["detection_point"] for r in selected_rows()}
    with predictions_path(m.name).open("a") as handle:
        for result in client.messages.batches.results(state["batch_id"]):
            record = anthropic_result_record(m, result)
            record["id"] = state["ids"][result.custom_id]
            record["detection_point"] = meta.get(record["id"])
            handle.write(json.dumps(record) + "\n")
    state_path.rename(state_path.with_name(f"batch_{state['batch_id']}.json"))
    print(f"collected {batch.request_counts}; spend {all_spend()}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=("run", "submit", "collect", "spend"))
    parser.add_argument("--monitor", choices=sorted(SPEC["monitors"]))
    parser.add_argument("--limit", type=int)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--out", help="dry runs: write here instead of results/")
    parser.add_argument("--force", action="store_true")
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
