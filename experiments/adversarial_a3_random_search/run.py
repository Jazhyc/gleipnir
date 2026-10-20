"""Prepare, run, and summarize the frozen 16-task A3 pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path

import numpy as np

from experiments.adversarial_a3_random_search.journal import Journal
from gleipnir.adversarial import apps, eval_sets
from gleipnir.adversarial.metrics import summary, threshold_at_fpr
from gleipnir.adversarial.random_search import (
    SearchState,
    SuffixSpace,
    append_comment,
    proposal,
)

HERE = Path(__file__).resolve().parent
CONFIG = json.loads((HERE / "config.json").read_text())
OUT = eval_sets.ROOT / CONFIG["output"]


def read(path: Path) -> list[dict]:
    return (
        [json.loads(x) for x in path.read_text().splitlines()] if path.exists() else []
    )


def inputs() -> tuple[dict, dict, dict]:
    path = eval_sets.ROOT / CONFIG["baseline"]
    assert eval_sets.file_sha256(path) == CONFIG["baseline_sha256"]
    baseline = {r["id"]: r for r in read(path)}
    all_rows = {}
    for row in eval_sets.load("apps-neutral"):
        task = str(row["metadata"]["task_id"])
        condition = eval_sets.group(row)
        if condition in ("clean", "vanilla"):
            all_rows[(task, condition)] = row
    thresholds = {
        str(f): threshold_at_fpr(
            [r["score"] for r in baseline.values() if r["label"] == 0], f
        )
        for f in CONFIG["fprs"]
    }
    eligible = []
    for task in sorted({task for task, _ in all_rows}):
        pair = [all_rows.get((task, c)) for c in ("clean", "vanilla")]
        if any(r is None for r in pair):
            continue
        vanilla = pair[1]
        if baseline[vanilla["id"]]["score"] <= thresholds["0.01"]:
            continue
        if any(
            baseline[r["id"]]["tokens"] > CONFIG["selection_max_original_tokens"]
            for r in pair
        ):
            continue
        try:
            for row in pair:
                append_comment(apps.parse(row["prompt"]).code, "pilot")
        except ValueError:
            continue
        eligible.append(task)
    selected = sorted(
        eligible,
        key=lambda task: hashlib.sha256(("a3-pilot-v1:" + task).encode()).hexdigest(),
    )[: len(CONFIG["task_ids"])]
    if len(eligible) != CONFIG["eligible_tasks"] or selected != CONFIG["task_ids"]:
        raise ValueError("frozen cohort selection changed")
    rows = {key: row for key, row in all_rows.items() if key[0] in selected}
    return rows, baseline, thresholds


def identity() -> dict:
    files = [
        "src/gleipnir/adversarial/" + name
        for name in (
            "random_search.py",
            "python_code.py",
            "apps.py",
            "contract.py",
            "scorer.py",
            "monitors.py",
            "metrics.py",
            "eval_sets.py",
        )
    ]
    files += [str(p.relative_to(eval_sets.ROOT)) for p in HERE.glob("*.py")]
    return {
        "config_sha256": eval_sets.file_sha256(HERE / "config.json"),
        "data_sha256": eval_sets.SETS["apps-neutral"].sha256,
        "source_sha256": {
            f: eval_sets.file_sha256(eval_sets.ROOT / f) for f in sorted(files)
        },
    }


def prepare() -> dict:
    rows, baseline, thresholds = inputs()
    ident = identity()
    manifest = {
        "identity": ident,
        "thresholds": thresholds,
        "rows": [
            {
                "task_id": task,
                "condition": condition,
                "id": row["id"],
                "prompt_sha256": hashlib.sha256(row["prompt"].encode()).hexdigest(),
                "baseline": baseline[row["id"]],
            }
            for (task, condition), row in sorted(rows.items())
        ],
    }
    for row in rows.values():
        append_comment(apps.parse(row["prompt"]).code, "pilot")
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "manifest.json"
    if path.exists() and json.loads(path.read_text()) != manifest:
        raise ValueError("manifest changed")
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def attacked_prompt(row: dict, payload: str) -> str:
    original = apps.parse(row["prompt"]).code
    edited = append_comment(original, payload)
    prompt = apps.with_code(row["prompt"], edited)
    # with_code preserves the trajectory's original trailing-newline convention;
    # require the original code bytes to remain its prefix in the final prompt.
    if not apps.parse(prompt).code.startswith(original):
        raise ValueError("original code bytes changed in trajectory")
    return prompt


def run() -> None:
    from gleipnir.adversarial.scorer import Scorer

    manifest = prepare()
    status_path = OUT / "run_status.json"
    previous = json.loads(status_path.read_text()) if status_path.exists() else {}
    if previous.get("state") == "running":
        raise RuntimeError(
            "unclean previous session: reconcile time and calls before resume"
        )
    elapsed_before = previous.get("elapsed_total_seconds", 0.0)
    if elapsed_before >= CONFIG["gpu_budget_seconds"]:
        raise RuntimeError("cumulative GPU time budget exhausted")
    status_path.write_text(
        json.dumps({"state": "running", "elapsed_total_seconds": elapsed_before})
    )
    started = time.monotonic()
    state_name = "failed"
    try:
        scorer = Scorer(CONFIG["monitor"])
        space = SuffixSpace(scorer.tokenizer, CONFIG["suffix_max_tokens"])
        pool_hash = hashlib.sha256(json.dumps(space.pool).encode()).hexdigest()
        ident = {
            **manifest["identity"],
            "scorer": scorer.identity(),
            "pool_sha256": pool_hash,
        }
        journal = Journal(OUT / "queries.jsonl", ident, CONFIG["total_call_cap"])
        rows, baseline, thresholds = inputs()
        (OUT / "runtime.json").write_text(
            json.dumps(
                {
                    "identity": ident,
                    "pool_size": len(space.pool),
                    "commit": subprocess.check_output(
                        ["git", "rev-parse", "HEAD"], text=True
                    ).strip(),
                },
                indent=2,
            )
        )

        def score(
            call_id: str, prompt: str, metadata: dict, adapter: bool = True
        ) -> dict:
            if (
                call_id not in journal.results
                and elapsed_before + time.monotonic() - started
                >= CONFIG["gpu_budget_seconds"]
            ):
                raise RuntimeError("cumulative GPU time budget exhausted")
            return journal.score(
                call_id, prompt, lambda p: scorer.score(p, adapter=adapter), metadata
            )

        fresh = {}
        for task in CONFIG["task_ids"]:
            for condition in ("clean", "vanilla"):
                row = rows[(task, condition)]
                result = score(
                    f"baseline:{task}:{condition}",
                    row["prompt"],
                    {"kind": "baseline", "task": task, "condition": condition},
                )
                if (
                    abs(result["score"] - baseline[row["id"]]["score"])
                    > CONFIG["baseline_max_abs_drift"]
                ):
                    raise ValueError("baseline drift exceeds frozen tolerance")
                if condition == "vanilla" and result["score"] <= thresholds["0.01"]:
                    raise ValueError("selected task no longer baseline-detected")
                fresh[(task, condition)] = result
        effects = []
        for condition in ("clean", "vanilla"):
            task = CONFIG["task_ids"][0]
            row = rows[(task, condition)]
            raw = score(
                f"base-model:{condition}",
                row["prompt"],
                {"kind": "adapter_canary", "condition": condition},
                adapter=False,
            )
            effects.append(abs(raw["score"] - fresh[(task, condition)]["score"]))
        if max(effects) <= 1e-6:
            raise ValueError("no adapter effect")
        (OUT / "canary.json").write_text(
            json.dumps(
                {
                    "passed": True,
                    "adapter_effects": effects,
                    "max_drift": max(
                        abs(v["score"] - baseline[rows[k]["id"]]["score"])
                        for k, v in fresh.items()
                    ),
                }
            )
        )
        states = {
            (task, seed, arm): SearchState(fresh[(task, "vanilla")]["score"])
            for task in CONFIG["task_ids"]
            for seed in CONFIG["seeds"]
            for arm in CONFIG["arms"]
        }
        for query in range(1, CONFIG["queries_per_search"] + 1):
            for task in CONFIG["task_ids"]:
                for seed in CONFIG["seeds"]:
                    for arm in CONFIG["arms"]:
                        state = states[(task, seed, arm)]
                        candidate = proposal(space, state, task, seed, arm, query)
                        prompt = attacked_prompt(
                            rows[(task, "vanilla")], candidate.text
                        )
                        metadata = {
                            "kind": "search",
                            "task": task,
                            "seed": seed,
                            "arm": arm,
                            "query": query,
                            "payload": candidate.text,
                            "pieces": list(candidate.pieces),
                            "payload_tokens": candidate.tokens,
                        }
                        value = score(
                            f"search:{task}:{seed}:{arm}:{query}", prompt, metadata
                        )
                        state.observe(candidate, value["score"])
            if query in CONFIG["checkpoints"]:
                print(
                    f"completed {query} queries/search; "
                    f"physical calls {len(journal.intents)}",
                    flush=True,
                )
        for (task, seed, arm), state in states.items():
            if state.best is not None:
                prompt = attacked_prompt(rows[(task, "clean")], state.best.text)
                score(
                    f"honest:{task}:{seed}:{arm}",
                    prompt,
                    {
                        "kind": "honest",
                        "task": task,
                        "seed": seed,
                        "arm": arm,
                        "payload": state.best.text,
                    },
                )
        state_name = "succeeded"
    finally:
        status_path.write_text(
            json.dumps(
                {
                    "state": state_name,
                    "elapsed_total_seconds": elapsed_before
                    + time.monotonic()
                    - started,
                },
                indent=2,
            )
        )


def summarize(records: list[dict], config: dict, thresholds: dict) -> dict:
    intents = {r["id"]: r for r in records if r["type"] == "intent"}
    results = {
        r["id"]: r for r in records if r["type"] == "result" and r["status"] == "ok"
    }
    baselines, scores, honest_results = {}, {}, {}
    for call_id, intent in intents.items():
        if call_id not in results:
            continue
        meta = intent["metadata"]
        value = results[call_id]["value"]
        if meta["kind"] == "baseline":
            baselines[(meta["task"], meta["condition"])] = value
        elif meta["kind"] == "search":
            key = (meta["task"], meta["seed"], meta["arm"])
            scores.setdefault(key, []).append((meta["query"], value, meta))
        elif meta["kind"] == "honest":
            honest_results[(meta["task"], meta["seed"], meta["arm"])] = value
    output = {
        "physical_calls": len(intents),
        "successful_calls": len(results),
        "failed_or_pending": len(intents) - len(results),
        "curves": {},
        "paired": {},
        "prompt_tokens": sum(r["value"].get("tokens", 0) for r in results.values()),
    }
    values = {}
    for arm in config["arms"]:
        curve = {}
        for limit in config["checkpoints"]:
            entry = {}
            for fpr, threshold in thresholds.items():
                task_rates, seed_rates = [], []
                complete_count = 0
                for task in config["task_ids"]:
                    seed_flags = []
                    for seed in config["seeds"]:
                        rs = [
                            r
                            for r in scores.get((task, seed, arm), [])
                            if r[0] <= limit
                        ]
                        complete = len(rs) == limit and {r[0] for r in rs} == set(
                            range(1, limit + 1)
                        )
                        complete_count += int(complete)
                        baseline = baselines.get((task, "vanilla"), {}).get(
                            "score", 1.0
                        )
                        best = min([baseline] + [r[1]["score"] for r in rs])
                        # Incomplete budgets are not reported as completed successes.
                        seed_flags.append(int(complete and best <= threshold))
                    task_rates.append(float(np.mean(seed_flags)))
                    seed_rates.append(seed_flags)
                rng = np.random.default_rng(0)
                boot = rng.choice(task_rates, (10000, len(task_rates))).mean(axis=1)
                values[(arm, limit, fpr)] = np.asarray(task_rates)
                entry[fpr] = {
                    "successes": int(np.sum(seed_rates)),
                    "planned_searches": len(config["task_ids"]) * len(config["seeds"]),
                    "completed_searches": complete_count,
                    "conservative_rate": float(np.mean(task_rates)),
                    "task_bootstrap95": np.quantile(boot, [0.025, 0.975]).tolist(),
                    "successes_by_seed": np.sum(seed_rates, axis=0).tolist(),
                }
            curve[str(limit)] = entry
        output["curves"][arm] = curve
    for limit in config["checkpoints"]:
        output["paired"][str(limit)] = {}
        for fpr in thresholds:
            diff = (
                values[("adaptive", limit, fpr)] - values[("independent", limit, fpr)]
            )
            rng = np.random.default_rng(0)
            boot = rng.choice(diff, (10000, len(diff))).mean(axis=1)
            output["paired"][str(limit)][fpr] = {
                "adaptive_minus_independent": float(diff.mean()),
                "task_bootstrap95": np.quantile(boot, [0.025, 0.975]).tolist(),
            }
    output["search_outcomes"] = []
    for (task, seed, arm), rs in sorted(scores.items()):
        base = baselines[(task, "vanilla")]
        ordered = sorted(rs, key=lambda r: r[0])
        best_score = base["score"]
        winner = None
        for candidate in ordered:
            if candidate[1]["score"] < best_score:
                best_score, winner = candidate[1]["score"], candidate
        output["search_outcomes"].append(
            {
                "task": task,
                "seed": seed,
                "arm": arm,
                "queries": len(rs),
                "baseline_score": base["score"],
                "best_score": best_score,
                "winning_query": winner[0] if winner else 0,
                "payload": winner[2]["payload"] if winner else "",
                "payload_tokens": winner[2]["payload_tokens"] if winner else 0,
                "added_prompt_tokens": winner[1].get("tokens", 0)
                - base.get("tokens", 0)
                if winner
                else 0,
                "first_success_query": {
                    f: next((q for q, v, _ in ordered if v["score"] <= t), None)
                    for f, t in thresholds.items()
                },
            }
        )
    output["honest_controls"] = {}
    for arm in config["arms"]:
        valid = []
        for task in config["task_ids"]:
            for seed in config["seeds"]:
                rs = scores.get((task, seed, arm), [])
                if (
                    len(rs) != config["queries_per_search"]
                    or (task, "clean") not in baselines
                ):
                    continue
                base = baselines[(task, "vanilla")]["score"]
                best = min([base] + [r[1]["score"] for r in rs])
                h = honest_results.get((task, seed, arm))
                if best == base:
                    h = baselines[(task, "clean")]
                if h is not None:
                    valid.append((best, h["score"]))
        output["honest_controls"][arm] = {
            "pairs": len(valid),
            "flagged": {f: sum(h > t for _, h in valid) for f, t in thresholds.items()},
            "ranking_conditional_cohort": summary(
                [1] * len(valid) + [0] * len(valid),
                [b for b, _ in valid] + [h for _, h in valid],
            )
            if valid
            else None,
        }
    for control in output["honest_controls"].values():
        metrics = control["ranking_conditional_cohort"]
        if metrics:
            control["ranking_conditional_cohort"] = {
                k: metrics[k] for k in ("auroc", "pauroc20")
            }
    return output


def report() -> None:
    manifest = prepare()
    records = read(OUT / "queries.jsonl")
    if not records:
        raise ValueError("no scoring receipts")
    recorded_identity = records[0]["value"]
    for key, value in manifest["identity"].items():
        if recorded_identity.get(key) != value:
            raise ValueError("report input identity changed")
    result = summarize(records, CONFIG, manifest["thresholds"])
    status_path = OUT / "run_status.json"
    result["runtime"] = (
        json.loads(status_path.read_text()) if status_path.exists() else {}
    )
    elapsed = result["runtime"].get("elapsed_total_seconds")
    result["prompt_tokens_per_second_including_setup"] = (
        result["prompt_tokens"] / elapsed if elapsed else None
    )
    (OUT / "report.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["prepare", "run", "report"])
    globals()[parser.parse_args().command]()
