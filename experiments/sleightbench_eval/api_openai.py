"""OpenAI logit-interface SleightBench run (exploratory run sb07: GPT-6 Sol).

Scores the frozen ``api_control_arena.jsonl`` rows (the sb05 Kimi K3 prompts)
through ``gleipnir.teachers.openai.AuditedClient`` with the GPT-6 Luna request
contract: Responses API, reasoning ``none``, temperature 1, ``top_p=1``, top-20
logprobs, ``store=false``. Only the model and pricing change. The client
requires an exact ``Prediction:0|1`` completion with both literal terminal
logprobs, logs every attempt and stops at a conservative budget bound.
``--limit`` runs a canary on the first rows.
"""

from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor

from experiments.sleightbench_eval.run import CONFIG, DATA, EXPERIMENT, RESULTS
from gleipnir.data.monitoring import read_rows, write_json
from gleipnir.teachers.openai import AuditedClient


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", choices=sorted(CONFIG["openai_models"]))
    parser.add_argument("--limit", type=int)
    parser.add_argument("--concurrency", type=int, default=8)
    args = parser.parse_args()
    spec = CONFIG["openai_models"][args.model]
    frozen = json.loads((EXPERIMENT / "manifest_api_control_arena.json").read_text())
    kimi = {
        r["id"]: r["usage"]["prompt_tokens"]
        for r in read_rows(RESULTS / "sb05" / "kimi_k3.jsonl")
    }
    rows = []
    for row in read_rows(DATA / "api_control_arena.jsonl"):
        row["metadata"] = {"kimi_raw_prompt_tokens": kimi[row["id"]]}
        rows.append(row)
    if [r["id"] for r in rows] != list(frozen):
        raise ValueError("API prompts differ from the frozen manifest")
    out = RESULTS / spec["run"] / f"{args.model}.jsonl"
    root = out.parent / f"{args.model}_attempts"
    root.mkdir(parents=True, exist_ok=True)
    done = {r["id"] for r in read_rows(out)} if out.exists() else set()
    todo = [r for r in rows if r["id"] not in done][: args.limit]
    execution = spec["execution"]
    client = AuditedClient(
        os.environ["OPENAI_API_KEY"],
        spec["request"],
        root,
        budget_usd=execution["budget_usd"],
        max_attempts=execution["max_attempts"],
        timeout_seconds=execution["timeout_seconds"],
        pricing=spec["pricing"],
        tokens_per_minute=execution["tokens_per_minute"],
        coverage_repeats=execution["coverage_repeats"],
    )

    def one(row: dict) -> dict:
        return {"id": row["id"], **client.score(row)}

    failures = []
    with ThreadPoolExecutor(args.concurrency) as pool, out.open("a") as handle:
        for row, future in [(r, pool.submit(one, r)) for r in todo]:
            try:
                result = future.result()
            except Exception as error:  # preserve valid siblings
                failures.append(f"{row['id']}: {error}")
                continue
            handle.write(json.dumps(result) + "\n")
            handle.flush()
    for failure in failures:
        print("FAILED", failure, flush=True)
    scored = read_rows(out)
    print(f"scored {len(scored)}/{len(rows)} spent_bound ${client.spent_bound:.4f}")
    if len(scored) == len(rows) and not failures:
        write_json(
            out.with_suffix(".complete.json"),
            {
                "model": spec["request"]["model"],
                "rows": len(scored),
                "conservative_cost_usd": client.spent_bound,
            },
        )


if __name__ == "__main__":
    main()
