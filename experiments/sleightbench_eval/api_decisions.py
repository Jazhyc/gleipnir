"""Native decision-model SleightBench runs (sb10: Jev 1.13; sb11: pplx-decider v1.2).

Uses the interface of the decision models' OOD points: the raw trajectory as
Decisions state, the frozen Jev Noul question, and the returned probability
as the score (``gleipnir.evaluation.openrouter_decisions``). Each state is the
trajectory inside the frozen ``api_control_arena.jsonl`` prompt for the same
row, so the model sees exactly what the logit-interface API monitors saw, without
their instruction. Requests are pinned to the provider without fallbacks,
with data collection denied. Any failure or input rejection stops the run;
``--limit`` runs a canary on the first rows.
"""

from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor

from dotenv import load_dotenv

from experiments.sleightbench_eval import benchmark
from experiments.sleightbench_eval.run import CONFIG, DATA, EXPERIMENT, RESULTS, ROOT
from gleipnir.data.monitoring import read_rows, write_json
from gleipnir.evaluation.openrouter_decisions import (
    DecisionRecord,
    DecisionsConfig,
    score_decision,
    stable_sha256,
)

OPEN, CLOSE = "<agent_trajectory>\n", "\n</agent_trajectory>"


def state(prompt: str) -> str:
    """The trajectory body between the frozen prompt's envelope tags."""
    if prompt.count(OPEN) != 1 or prompt.count(CLOSE) != 1:
        raise ValueError("prompt must contain one trajectory envelope")
    return prompt.split(OPEN)[1].split(CLOSE)[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", choices=sorted(CONFIG["decision_models"]))
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    spec = CONFIG["decision_models"][args.model]
    contract = json.loads((ROOT / spec["question"]["path"]).read_text())
    if stable_sha256(contract["question"]) != spec["question"]["question_sha256"]:
        raise ValueError("Noul question differs from the frozen hash")
    frozen = json.loads((EXPERIMENT / "manifest_api_control_arena.json").read_text())
    rows = read_rows(DATA / "api_control_arena.jsonl")
    if {r["id"]: benchmark.sha256(r["prompt"].encode()) for r in rows} != frozen:
        raise ValueError("API prompts differ from the frozen manifest")
    execution = spec["execution"]
    config = DecisionsConfig(
        model=spec["openrouter_id"],
        provider=spec["provider"],
        question_id=contract["question_id"],
        question=contract["question"],
        request_timeout=execution["request_timeout_seconds"],
        max_retries=execution["max_retries"],
        app_title="Gleipnir SleightBench evaluation",
    )
    out = RESULTS / spec["run"] / f"{args.model}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    done = {r["id"] for r in read_rows(out)} if out.exists() else set()
    todo = [r for r in rows if r["id"] not in done][: args.limit]
    load_dotenv()
    key = os.environ["OPENROUTER_API_KEY"]
    rate = spec["pricing"]["input_usd_per_million"] / 1e6

    def one(row: dict) -> dict:
        record = DecisionRecord(row["id"], state(row["prompt"]), {})
        result = score_decision(record, config, key)
        if result["model"] != spec["provider_revision"]:
            raise ValueError(f"unexpected revision {result['model']!r}")
        return result

    previous = read_rows(out) if done else []
    spent = sum(r["usage"]["input_tokens"] for r in previous) * rate
    with ThreadPoolExecutor(execution["concurrency"]) as pool, out.open("a") as handle:
        for future in [pool.submit(one, r) for r in todo]:
            result = future.result()  # any failure stops the run
            spent += result["usage"]["input_tokens"] * rate
            if spent > execution["budget_usd"]:
                raise RuntimeError(f"budget exceeded: ${spent:.4f}")
            handle.write(json.dumps(result) + "\n")
            handle.flush()
    scored = read_rows(out)
    print(f"scored {len(scored)}/{len(rows)} list-price cost ${spent:.4f}")
    if len(scored) == len(rows):
        write_json(
            out.with_suffix(".complete.json"),
            {
                "model": spec["provider_revision"],
                "rows": len(scored),
                "input_tokens": sum(r["usage"]["input_tokens"] for r in scored),
                "list_price_cost_usd": spent,
            },
        )


if __name__ == "__main__":
    main()
