"""Offline budget choice from measured smoke usage and full reservation bounds."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from experiments.adversarial_crossed_text.campaign import Campaign

from .common import ROOT, config, digest, fixed, read
from .dependency import arm_config, instructed_pairs
from .pipeline import generation_estimate, messages
from .selection import benchmark_pairs, select
from .smoke import development_tasks, verify_evidence


def budget_check(
    cfg: dict, dependencies: Path, gate: dict, pairs: dict, ledger: list[dict]
) -> dict:
    if gate["status"] != "passed":
        raise ValueError("attacker smoke did not pass")
    verify_evidence(cfg, gate, pairs)
    writer = gate["selected_writer"]
    measured = gate["summaries"][-1]
    if not measured["usage_complete"]:
        raise ValueError("complete smoke usage required for cost check")
    latest = {r["id"]: r for r in ledger}
    scope = sum(r["charge"] for r in latest.values() if r["scope"] == "a4-writer")
    total = sum(r["charge"] for r in latest.values() if r["api"] == "tinker")
    remaining = min(
        140 - scope,
        cfg["budgets_usd"]["tinker"] - scope,
        cfg["campaign"]["caps_usd"]["tinker"] - total,
    )
    tasks, population = {}, {}
    for arm in cfg["monitor_contracts"]["search_arms"]:
        ac = arm_config(cfg, arm)
        dep = json.loads((dependencies / f"{arm}.json").read_text())
        tasks[arm] = select(dep, instructed_pairs(pairs, ac), ac)["task_ids"]
        population[arm] = {
            "dependency_sha256": digest(dep),
            "tasks_sha256": digest(tasks[arm]),
            "requests_sha256": digest(
                {t: messages(pairs[t]["backdoored"]) for t in tasks[arm]}
            ),
        }
    one_cfg = {**cfg, "writer": writer, "attempts": 1}
    per_attempt = sum(
        generation_estimate(ts, pairs, one_cfg)["worst_case_usd"]
        for ts in tasks.values()
    )
    attempts = min(10, max(0, math.floor((remaining - 1e-9) / per_attempt)))
    if not attempts:
        raise ValueError("not enough budget for one complete attempt in both arms")
    smoke_bytes = (
        sum(
            sum(len(m["content"].encode()) for m in messages(pairs[t]["backdoored"]))
            for t in development_tasks(cfg)
        )
        / 5
    )
    bytes_per_round = sum(
        sum(
            sum(len(m["content"].encode()) for m in messages(pairs[t]["backdoored"]))
            for t in ts
        )
        for ts in tasks.values()
    )
    expected_prompt = (
        attempts * bytes_per_round * measured["mean_prompt_tokens"] / smoke_bytes
    )
    expected_output = (
        attempts * sum(map(len, tasks.values())) * measured["mean_completion_tokens"]
    )
    return {
        "smoke_gate_sha256": digest(gate),
        "writer": writer,
        "cohort_sizes": {a: len(ts) for a, ts in tasks.items()},
        "population": population,
        "per_arm_one_attempt_worst_usd": {
            a: generation_estimate(ts, pairs, one_cfg)["worst_case_usd"]
            for a, ts in tasks.items()
        },
        "attempts": attempts,
        "planned_calls": attempts * sum(map(len, tasks.values())),
        "scope_already_charged_usd": scope,
        "campaign_tinker_already_charged_usd": total,
        "remaining_usd": remaining,
        "worst_case_generation_usd": attempts * per_attempt,
        "expected_generation_usd": (
            expected_prompt * writer["usd_per_m_prompt"]
            + expected_output * writer["usd_per_m_completion"]
        )
        / 1e6,
        "mean_smoke_prompt_tokens": measured["mean_prompt_tokens"],
        "mean_smoke_completion_tokens": measured["mean_completion_tokens"],
        "ledger_sha256": digest(ledger),
        "method": (
            "expected input scaled by smoke token/UTF-8 byte ratio; "
            "mean all-call output; worst-case bytes + 2048 framing "
            "and full output cap"
        ),
    }


def check_gate(cfg: dict) -> dict:
    spec = cfg["smoke_gate"]
    gate = json.loads((ROOT / spec["path"]).read_text())
    if (
        digest(gate) != spec["sha256"]
        or gate["status"] != "passed"
        or gate["selected_writer"] != cfg["writer"]
    ):
        raise ValueError("smoke gate/writer mismatch")
    verify_evidence(cfg, gate)
    for summary in gate["summaries"]:
        rows = read(
            ROOT
            / cfg["smoke"]["output"]
            / str(summary["candidate"])
            / "generations.jsonl"
        )
        if digest(rows) != summary["receipts_sha256"]:
            raise ValueError("smoke receipt changed")
    receipt = json.loads((ROOT / cfg["cost_check"]["path"]).read_text())
    if (
        digest(receipt) != cfg["cost_check"]["sha256"]
        or receipt["smoke_gate_sha256"] != digest(gate)
        or receipt["writer"] != cfg["writer"]
        or receipt["attempts"] != cfg["attempts"]
    ):
        raise ValueError("cost receipt/settings mismatch")
    return receipt


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--dependencies",
        type=Path,
        default=ROOT / "results/adversarial_a4_scale_dependencies_20261023",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=ROOT / "results/adversarial_a4_scale_cost_bound_20261023.json",
    )
    args = p.parse_args()
    cfg = config()
    gate = json.loads((ROOT / cfg["smoke"]["output"] / "gate.json").read_text())
    result = budget_check(
        cfg,
        args.dependencies,
        gate,
        benchmark_pairs(cfg),
        read(Campaign(cfg, args.out.parent).ledger),
    )
    fixed(args.out, result)
    print(json.dumps({k: v for k, v in result.items() if k != "writer"}, indent=2))


def verify_population(
    receipt: dict, arm: str, selected: dict, pairs: dict, cfg: dict
) -> None:
    expected = receipt["population"][arm]
    tasks = selected["task_ids"]
    if expected != {
        "dependency_sha256": selected["dependency_sha256"],
        "tasks_sha256": digest(tasks),
        "requests_sha256": digest({t: messages(pairs[t]["backdoored"]) for t in tasks}),
    }:
        raise ValueError("cost approval population changed")
    one = generation_estimate(tasks, pairs, {**cfg, "attempts": 1})["worst_case_usd"]
    if one != receipt["per_arm_one_attempt_worst_usd"][arm]:
        raise ValueError("cost approval request bound changed")


if __name__ == "__main__":
    main()
