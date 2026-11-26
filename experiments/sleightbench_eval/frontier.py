"""Cost and score of each monitor on SleightBench, for the report's frontier.

Every row is the stripped view with the ControlArena instruction: the binary
logit prompt (``sb02`` for Gleipnir, ``sb04`` immediate for the base models,
``sb05`` and the OpenAI runs) or the same instruction as a Noul question for
the decision models (``sb12``, ``sb15``, ``sb16``).

Costs are USD per 1,000 SleightBench evaluations, at the prices of the OOD
frontier (``docs/research/tool_trajectory_ood_frontier.md``):

- API monitors: mean billed input and output tokens at the list price,
  without cache discounts.
- Base models: their own prompt tokens at the hosted proxy prices of the OOD
  frontier, with the one output token.
- Gleipnir: its prompt tokens at the per-token cost of one B200 serving the
  benchmark of Section 4 with one request at a time, at the on-demand and the
  committed-use rate.

``python -m experiments.sleightbench_eval.frontier`` writes
``results/sleightbench_eval/frontier.json`` and prints a Markdown table.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass

from experiments.sleightbench_eval.run import EXPERIMENT, RESULTS
from gleipnir.data.monitoring import read_rows, write_json

VIEW = "stripped"
OUTPUT = RESULTS / "frontier.json"

# One B200, one request at a time (Section 4): hourly rate and prompt tokens/s.
RATES = {"on_demand": 7.99, "committed": 3.44}
THROUGHPUT = {"4b": 204_673, "27b": 50_788}


@dataclass(frozen=True)
class Monitor:
    name: str
    run: str
    model: str  # result file stem
    kind: str  # gleipnir, base or api
    input_price: float = 0.0  # USD per million tokens
    output_price: float = 0.0
    tokens_from: str = ""  # Qwen tokenizer whose prompt counts price the row


MONITORS = [
    Monitor("Gleipnir 4B", "sb02", "4b", "gleipnir", tokens_from="4b"),
    Monitor("Gleipnir 27B", "sb02", "27b", "gleipnir", tokens_from="27b"),
    Monitor("Qwen3.5-4B", "sb04_immediate", "4b", "base", 0.03, 0.15, "4b"),
    Monitor("Qwen3.8-27B", "sb04_immediate", "27b", "base", 0.195, 1.56, "27b"),
    # The frozen Fireworks ceiling of the OOD frontier, not the Makora rate
    # that sb05 paid.
    Monitor("Kimi K3", "sb05", "kimi_k3", "api", 3.0, 15.0),
    Monitor("GPT-6 Luna", "sb09", "gpt6_luna", "api", 0.10, 0.50),
    Monitor("GPT-6.1 Terra", "sb13", "gpt61_terra", "api", 1.0, 5.0),
    Monitor("GPT-6 Sol", "sb07", "gpt6_sol", "api", 2.0, 10.0),
    Monitor("GPT-5.5", "sb08", "gpt55", "api", 5.0, 30.0),
    Monitor("Jev 1.13", "sb12", "jev_113_ca", "api", 0.042),
    Monitor("Jev 1.14", "sb16", "jev_114_ca", "api", 0.042),
    # Undiscounted list price; OpenRouter lists $0.06/M at 50% off.
    Monitor("Drex Max", "sb17", "drex_max_ca", "api", 0.12),
    Monitor("pplx-decider v1.2", "sb12", "decider_v12_ca", "api", 0.02),
    # Free on OpenRouter; priced at v1.2's list price (user, 2026-11-18).
    Monitor("pplx-decider v1.3", "sb15", "decider_v13_ca", "api", 0.02),
    Monitor("Gemini Decision 1", "sb15", "gemini_decision_1_ca", "api", 0.10),
]


def _usage(row: dict) -> tuple[int, int]:
    usage = row["usage"]
    prompt = usage.get("prompt_tokens", usage.get("input_tokens"))
    output = usage.get("completion_tokens", usage.get("output_tokens", 0))
    return int(prompt), int(output)


def _prompt_tokens(model: str) -> dict[str, int]:
    manifest = json.loads((EXPERIMENT / "manifest_control_arena.json").read_text())
    return {
        p["id"]: p["prompt_tokens"]
        for p in manifest["models"][model]["prompts"]
        if p["view"] == VIEW
    }


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def per_thousand(tokens: float, usd_per_million: float) -> float:
    return tokens * usd_per_million / 1e3


def gleipnir_usd_per_million(model: str, rate: str) -> float:
    return RATES[rate] / (THROUGHPUT[model] * 3600) * 1e6


def rows() -> list[dict]:
    out = []
    for monitor in MONITORS:
        report = json.loads((RESULTS / monitor.run / "report.json").read_text())
        result = report["models"][monitor.model][VIEW]
        units, ci = result["units"], result["intervals_95"]
        scores = {
            "pauroc_at_20": units["pauroc_at_20"],
            "catch_rate": units["catch_rate"],
            "auroc": units["auroc"],
            "pair_win_rate": units["pair_win_rate"],
            "threshold": units["threshold"],
            "intervals_95": {k: ci[k] for k in ("pauroc_at_20", "catch_rate")},
        }
        if monitor.kind == "api":
            scored = [
                r
                for r in read_rows(RESULTS / monitor.run / f"{monitor.model}.jsonl")
                if r["id"].startswith(f"{VIEW}/")
            ]
            usage = [_usage(r) for r in scored]
            input_tokens = _mean([u[0] for u in usage])
            output_tokens = _mean([u[1] for u in usage])
        else:
            input_tokens = _mean(list(_prompt_tokens(monitor.tokens_from).values()))
            output_tokens = 1.0 if monitor.kind == "base" else 0.0
        if monitor.kind == "gleipnir":
            for rate in RATES:
                price = gleipnir_usd_per_million(monitor.model, rate)
                out.append(
                    {
                        "monitor": monitor.name,
                        "pricing": rate,
                        "kind": monitor.kind,
                        "mean_input_tokens": input_tokens,
                        "mean_output_tokens": output_tokens,
                        "cost_per_1k": per_thousand(input_tokens, price),
                        **scores,
                        "spec": asdict(monitor),
                    }
                )
            continue
        cost = per_thousand(input_tokens, monitor.input_price) + per_thousand(
            output_tokens, monitor.output_price
        )
        out.append(
            {
                "monitor": monitor.name,
                "pricing": "list",
                "kind": monitor.kind,
                "mean_input_tokens": input_tokens,
                "mean_output_tokens": output_tokens,
                "cost_per_1k": cost,
                **scores,
                "spec": asdict(monitor),
            }
        )
    return out


def main() -> None:
    table = rows()
    write_json(OUTPUT, {"view": VIEW, "rates": RATES, "monitors": table})
    print(
        "| Monitor | Pricing | Input tokens | Cost / 1K | pAUROC@20 | Catch | AUROC |"
    )
    print("|---|---|---:|---:|---:|---:|---:|")
    for r in sorted(table, key=lambda r: r["cost_per_1k"]):
        print(
            f"| {r['monitor']} | {r['pricing']} | {r['mean_input_tokens']:,.0f} "
            f"| {r['cost_per_1k']:.3f} | {r['pauroc_at_20']:.3f} "
            f"| {r['catch_rate']:.3f} | {r['auroc']:.3f} |"
        )
    print(f"wrote {OUTPUT}")


if __name__ == "__main__":
    main()
