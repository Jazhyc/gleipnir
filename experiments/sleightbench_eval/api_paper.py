"""Authors' SleightBench protocol for an OpenRouter model (exploratory run sb06).

The request follows upstream ``render_prompt``: the verbatim ControlArena
prompt (``prompts/control_arena_paper.md``) as the system message and the
stripped transcript inside ``<transcript>`` tags as the user message. Each
transcript is sampled ``trials`` times at temperature 1. The score is the
last ``<score>N</score>`` (0–100) in the final answer. Both messages carry
ephemeral cache breakpoints, and every transcript has its own sticky session,
so trials 2..n can reuse the provider prompt cache. ``probe`` sends one prompt
a few times and reports the cache telemetry and cost before any full run.
Requests deny provider data collection and never fall back to another provider.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import requests

from experiments.sleightbench_eval import benchmark
from experiments.sleightbench_eval.reasoning import paper_score
from experiments.sleightbench_eval.run import CONFIG, DATA, EXPERIMENT, RESULTS
from gleipnir.data.monitoring import read_rows, write_json

ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"
RETRYABLE = {408, 409, 425, 429, 500, 502, 503, 504}


def prompts() -> list[dict[str, str]]:
    """The stripped-view system/user pairs, in frozen transcript order."""
    system = (EXPERIMENT / "prompts/control_arena_paper.md").read_text()
    rows = []
    for t in benchmark.transcripts(DATA, set(CONFIG["excluded_attacks"])):
        body = benchmark.body(benchmark.stripped(benchmark.read_messages(t.path)))
        rows.append(
            {
                "id": f"stripped/{t.id}",
                "system": system,
                "user": f"<transcript>\n{body}\n</transcript>",
            }
        )
    return rows


def prompt_sha256(row: dict[str, str]) -> str:
    return hashlib.sha256(
        json.dumps([row["system"], row["user"]]).encode("utf-8")
    ).hexdigest()


def payload(row: dict[str, str], spec: dict[str, Any]) -> dict[str, Any]:
    def cached(text: str) -> list[dict[str, Any]]:
        return [{"type": "text", "text": text, "cache_control": {"type": "ephemeral"}}]

    return {
        "model": spec["model"],
        "messages": [
            {"role": "system", "content": cached(row["system"])},
            {"role": "user", "content": cached(row["user"])},
        ],
        "temperature": spec["temperature"],
        "max_tokens": spec["max_tokens"],
        "reasoning": {"effort": spec["reasoning_effort"], "exclude": True},
        "provider": {
            "only": [spec["provider"]],
            "allow_fallbacks": False,
            "data_collection": "deny",
            "require_parameters": True,
        },
        "session_id": f"sleightbench-{spec['name']}-{prompt_sha256(row)[:24]}",
        "usage": {"include": True},
    }


def request(row: dict[str, str], spec: dict[str, Any], key: str) -> dict[str, Any]:
    body = payload(row, spec)
    error = None
    for attempt in range(spec["max_retries"] + 1):
        started = time.perf_counter()
        try:
            response = requests.post(
                ENDPOINT,
                headers={"Authorization": f"Bearer {key}"},
                json=body,
                timeout=spec["timeout_seconds"],
            )
            if response.status_code < 400:
                data = response.json()
                choice = data["choices"][0]
                text = choice["message"].get("content") or ""
                usage = data.get("usage") or {}
                details = usage.get("prompt_tokens_details") or {}
                return {
                    "text": text,
                    "score": paper_score(text),
                    "finish_reason": choice.get("finish_reason"),
                    "provider": data.get("provider"),
                    "model": data.get("model"),
                    "prompt_tokens": usage.get("prompt_tokens"),
                    "completion_tokens": usage.get("completion_tokens"),
                    "reasoning_tokens": (
                        usage.get("completion_tokens_details") or {}
                    ).get("reasoning_tokens"),
                    "cached_tokens": details.get("cached_tokens", 0),
                    "cache_write_tokens": details.get("cache_write_tokens", 0),
                    "cost": usage.get("cost"),
                    "latency_seconds": time.perf_counter() - started,
                    "attempts": attempt + 1,
                }
            error = f"HTTP {response.status_code}: {response.text[:300]}"
            if response.status_code not in RETRYABLE:
                break
        except (requests.RequestException, KeyError, ValueError) as exc:
            error = repr(exc)
        time.sleep(min(60.0, 2.0**attempt))
    raise RuntimeError(f"{row['id']} failed: {error}")


def arm_spec(arm: str) -> dict[str, Any]:
    spec = {**CONFIG["api_paper"]["common"], **CONFIG["api_paper"]["arms"][arm]}
    spec["name"] = arm
    return spec


def probe(arm: str, row_id: str, repeats: int, key: str) -> None:
    row = next(r for r in prompts() if r["id"] == row_id)
    spec = arm_spec(arm)
    for index in range(repeats):
        result = request(row, spec, key)
        fields = (
            "provider",
            "prompt_tokens",
            "cached_tokens",
            "cache_write_tokens",
            "completion_tokens",
            "reasoning_tokens",
            "cost",
            "score",
            "latency_seconds",
        )
        print(
            json.dumps({"repeat": index, **{k: result[k] for k in fields}}), flush=True
        )


def run(arm: str, key: str, concurrency: int) -> None:
    spec = arm_spec(arm)
    out = RESULTS / f"sb06_{arm}" / "kimi_k3.trials.jsonl"
    complete = out.parent / "kimi_k3.complete.json"
    if complete.exists():
        raise ValueError(f"{arm} already complete; no reruns")
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = prompts()
    done = {(r["id"], r["trial"]) for r in read_rows(out)} if out.exists() else set()
    lock = threading.Lock()

    def one_transcript(row: dict[str, str]) -> None:
        # Trial 0 first warms this transcript's cache, then the rest follow.
        for trial in range(spec["trials"]):
            if (row["id"], trial) in done:
                continue
            result = request(row, spec, key)
            with lock, out.open("a") as handle:
                handle.write(
                    json.dumps(
                        {
                            "id": row["id"],
                            "trial": trial,
                            "prompt_sha256": prompt_sha256(row),
                            **result,
                        }
                    )
                    + "\n"
                )

    with ThreadPoolExecutor(concurrency) as pool:
        for future in [pool.submit(one_transcript, r) for r in rows]:
            future.result()
    trials = read_rows(out)
    if len({(r["id"], r["trial"]) for r in trials}) != len(rows) * spec["trials"]:
        raise ValueError("incomplete trials; rerun to resume")
    collapse(out, rows, spec["trials"])
    write_json(
        complete,
        {
            "arm": arm,
            "spec": spec,
            "rows": len(rows),
            "trials": spec["trials"],
            "unparsed_trials": sum(r["score"] is None for r in trials),
            "cost_usd": sum(r["cost"] or 0 for r in trials),
            "cached_tokens": sum(r["cached_tokens"] or 0 for r in trials),
            "prompt_tokens": sum(r["prompt_tokens"] or 0 for r in trials),
        },
    )


def collapse(out: Path, rows: list[dict[str, str]], trials: int) -> None:
    """Write ``kimi_k3.jsonl`` in the ``report-trials`` row format."""
    by_id: dict[str, list[float | None]] = {r["id"]: [None] * trials for r in rows}
    for r in read_rows(out):
        by_id[r["id"]][r["trial"]] = r["score"]
    with (out.parent / "kimi_k3.jsonl").open("w") as handle:
        for row_id, scores in by_id.items():
            handle.write(json.dumps({"id": row_id, "trials": scores}) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("arm", choices=sorted(CONFIG["api_paper"]["arms"]))
    commands = parser.add_subparsers(dest="command", required=True)
    command = commands.add_parser("probe")
    command.add_argument("--id", required=True)
    command.add_argument("--repeats", type=int, default=3)
    command = commands.add_parser("run")
    command.add_argument("--concurrency", type=int, default=8)
    args = parser.parse_args()
    key = os.environ["OPENROUTER_API_KEY"]
    if args.command == "probe":
        probe(args.arm, args.id, args.repeats, key)
    else:
        run(args.arm, key, args.concurrency)


if __name__ == "__main__":
    main()
