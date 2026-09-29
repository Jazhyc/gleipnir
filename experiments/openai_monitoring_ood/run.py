"""Run resumable Standard OpenAI monitoring with a frozen full teacher prompt."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from dotenv import dotenv_values

from experiments.tool_trajectory_monitoring.prompting import load_prompt_set
from experiments.tool_trajectory_monitoring.teacher_canary import (
    atomic_write_json,
    load_jsonl,
    summarize_scored_rows,
)
from gleipnir.openai_monitor import AuditedClient, digest


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def cached_scores(
    path: Path,
    rows: list[dict[str, Any]],
    settings: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    """Reject corrupt, duplicate, or mismatched scores on resumption."""
    expected = {r["id"]: hashlib.sha256(r["prompt"].encode()).hexdigest() for r in rows}
    result = {}
    if path.exists():
        for r in load_jsonl(path):
            if r["id"] in result or r["id"] not in expected:
                raise ValueError("duplicate or unexpected cached row")
            if r["prompt_sha256"] != expected[r["id"]] or r[
                "request_settings_sha256"
            ] != digest(settings):
                raise ValueError("cached score identity differs")
            result[r["id"]] = r
    return result


def score_rows(
    client: AuditedClient,
    rows: list[dict[str, Any]],
    path: Path,
    workers: int,
) -> list[dict[str, Any]]:
    """Keep at most workers requests outstanding and flush every valid sibling."""
    scores = cached_scores(path, rows, client.settings)
    remaining = iter(r for r in rows if r["id"] not in scores)
    started = time.perf_counter()
    initial = len(scores)
    failure = None
    with path.open("a") as output, ThreadPoolExecutor(max_workers=workers) as pool:
        pending = {}

        def enqueue() -> None:
            row = next(remaining, None)
            if row is not None:
                pending[pool.submit(client.score, row)] = row["id"]

        for _ in range(workers):
            enqueue()
        while pending:
            done, _ = wait(pending, timeout=30, return_when=FIRST_COMPLETED)
            for future in done:
                record_id = pending.pop(future)
                try:
                    score = future.result()
                    output.write(json.dumps(score) + "\n")
                    output.flush()
                    scores[record_id] = score
                except Exception as error:
                    failure = error
                    print(f"STOP {error}", flush=True)
            if failure is None:
                for _ in done:
                    enqueue()
            if not done or len(scores) % 100 < max(1, len(done)) or not pending:
                elapsed = time.perf_counter() - started
                rate = (len(scores) - initial) / elapsed if elapsed else 0
                eta = (len(rows) - len(scores)) / rate if rate else None
                print(
                    json.dumps(
                        {
                            "phase": path.stem,
                            "saved": len(scores),
                            "total": len(rows),
                            "in_flight": len(pending),
                            "rows_per_second": rate,
                            "eta_seconds": eta,
                            "conservative_cost_bound_usd": client.spent_bound,
                        }
                    ),
                    flush=True,
                )
    if failure is not None:
        raise RuntimeError(f"{path.stem} stopped; valid siblings saved") from failure
    return [scores[r["id"]] for r in rows]


def usage_cost(attempts: list[dict[str, Any]]) -> dict[str, Any]:
    """Include usage from failed parses and retries in operational accounting."""
    usages = [(r.get("raw_response") or {}).get("usage") or {} for r in attempts]
    inputs = sum(u.get("input_tokens", 0) for u in usages)
    outputs = sum(u.get("output_tokens", 0) for u in usages)
    cached = sum(
        u.get("input_tokens_details", {}).get("cached_tokens", 0) for u in usages
    )
    reasoning = sum(
        u.get("output_tokens_details", {}).get("reasoning_tokens", 0) for u in usages
    )
    writes_known = all(
        "cache_write_tokens" in u.get("input_tokens_details", {}) for u in usages if u
    )
    writes = sum(
        u.get("input_tokens_details", {}).get("cache_write_tokens", 0) for u in usages
    )
    return {
        "http_attempts": len(attempts),
        "input_tokens": inputs,
        "output_tokens": outputs,
        "cached_input_tokens": cached,
        "cache_write_tokens": writes if writes_known else None,
        "usage_priced_standard_usd": (
            (
                (inputs - cached - writes) * 0.10
                + cached * 0.01
                + writes * 0.125
                + outputs * 0.50
            )
            / 1e6
        )
        if writes_known
        else None,
        "reasoning_tokens": reasoning,
        "uncached_standard_usd": (inputs * 0.10 + outputs * 0.50) / 1e6,
        "cache_adjusted_standard_usd_without_write_premium": (
            (inputs - cached) * 0.10 + cached * 0.01 + outputs * 0.50
        )
        / 1e6,
        "all_uncached_tokens_as_writes_usd": (
            (inputs - cached) * 0.125 + cached * 0.01 + outputs * 0.50
        )
        / 1e6,
        "conservative_budget_bound_usd": sum(
            r["conservative_cost_usd"] for r in attempts
        ),
        "billing_note": (
            "Usage-derived estimates, not a billing receipt. Unknown transport "
            "billing is included only in the budget bound. Write-adjusted pricing "
            "requires cache_write_tokens on every usage-bearing attempt."
        ),
        "parse_failures": sum("parse_failure" in r for r in attempts),
        "transport_failures": sum("transport_error" in r for r in attempts),
        "http_errors": sum(r.get("http_status", 0) >= 400 for r in attempts),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("experiments/openai_monitoring_ood/config.json"),
    )
    parser.add_argument("--stop-after-canary", action="store_true")
    parser.add_argument(
        "--tokens-per-minute",
        type=int,
        help="Operational request pacing; preserves model settings and 40 workers.",
    )
    parser.add_argument(
        "--coverage-repeats",
        type=int,
        default=0,
        help="Bound identical repeats only when a literal label is absent from top20.",
    )
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    template = load_prompt_set().teacher
    if template.template_sha256 != config["prompt_template_sha256"]:
        raise ValueError("teacher template drift")
    for key in ("input", "canary_input"):
        if file_hash(Path(config[key])) != config[key + "_sha256"]:
            raise ValueError(f"frozen {key} checksum differs")
    rows = load_jsonl(Path(config["input"]))
    if len(rows) != config["rows"] or len({r["id"] for r in rows}) != len(rows):
        raise ValueError("frozen row coverage differs")
    for row in rows:
        metadata = row["metadata"]
        if (
            metadata["prompt_template_sha256"] != template.template_sha256
            or metadata["rendered_prompt_sha256"]
            != hashlib.sha256(row["prompt"].encode()).hexdigest()
            or not row["prompt"].startswith(template.cache_prefix)
        ):
            raise ValueError("rendered teacher prompt differs")
    canary = sorted(
        load_jsonl(Path(config["canary_input"])),
        key=lambda r: hashlib.sha256(r["id"].encode()).hexdigest(),
    )[: config["canary_rows"]]
    longest = max(rows, key=lambda r: len(r["prompt"]))
    root = Path(config["output"])
    root.mkdir(parents=True, exist_ok=True)
    frozen = root / "manifest.json"
    identity = {
        "config": config,
        "config_sha256": digest(config),
        "input_manifest_sha256": file_hash(Path(config["input_manifest"])),
        "canary_ids": [r["id"] for r in canary],
        "longest_ood_id": longest["id"],
        "request_settings_sha256": digest(config["request"]),
    }
    if frozen.exists():
        if json.loads(frozen.read_text())["identity"] != identity:
            raise ValueError("run manifest drift")
    else:
        atomic_write_json(
            frozen,
            {"created_at_utc": datetime.now(UTC).isoformat(), "identity": identity},
        )
    key = dotenv_values(".env").get("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY missing from .env")
    execution = config["execution"]
    client = AuditedClient(
        key,
        config["request"],
        root,
        budget_usd=execution["budget_usd"],
        max_attempts=execution["max_attempts"],
        timeout_seconds=execution["timeout_seconds"],
        tokens_per_minute=args.tokens_per_minute,
        coverage_repeats=args.coverage_repeats,
    )
    with (root / "execution_events.jsonl").open("a") as handle:
        handle.write(
            json.dumps(
                {
                    "started_at_utc": datetime.now(UTC).isoformat(),
                    "concurrency": execution["concurrency"],
                    "tokens_per_minute": args.tokens_per_minute,
                    "coverage_repeats": args.coverage_repeats,
                    "token_pacing_estimate": "1.05 * stored Kimi tokens + 48",
                    "source_sha256": {
                        str(p): file_hash(p)
                        for p in (
                            Path(__file__),
                            Path("src/gleipnir/openai_monitor.py"),
                        )
                    },
                }
            )
            + "\n"
        )
    workers = execution["concurrency"]
    score_rows(client, canary, root / "canary_predictions.jsonl", workers)
    score_rows(client, [longest], root / "longest_predictions.jsonl", workers)
    atomic_write_json(
        root / "canary_result.json",
        {
            "passed": True,
            "non_ood_rows": len(canary),
            "longest_ood_id": longest["id"],
            "settings_sha256": digest(config["request"]),
        },
    )
    print(
        "CANARY PASSED: full prompt format, both digit scores, "
        "model/settings, zero reasoning",
        flush=True,
    )
    if args.stop_after_canary:
        return
    scores_path = root / "predictions.jsonl"
    prior = cached_scores(scores_path, rows, client.settings)
    if longest["id"] not in prior:
        with scores_path.open("a") as handle:
            handle.write(
                json.dumps(load_jsonl(root / "longest_predictions.jsonl")[0]) + "\n"
            )
    scores = score_rows(client, rows, scores_path, workers)
    print("COMPLETE COVERAGE; computing frozen metrics", flush=True)
    summary = summarize_scored_rows(
        rows,
        scores,
        bootstrap_resamples=config["scoring"]["bootstrap_resamples"],
        seed=config["scoring"]["seed"],
    )
    attempts = load_jsonl(root / "attempts.jsonl")
    score_ids = {s["id"] for s in scores}
    benchmark_attempts = [r for r in attempts if r["id"] in score_ids]
    summary["campaign_accounting"] = usage_cost(attempts)
    summary["benchmark_accounting"] = usage_cost(benchmark_attempts)
    summary["benchmark_accounting"]["uncached_usd_per_1000"] = (
        summary["benchmark_accounting"]["uncached_standard_usd"] * 1000 / len(scores)
    )
    summary["positive_logprob_roundoff_rows"] = sum(
        s["positive_logprob_roundoff"] for s in scores
    )
    summary["unique_logprob_margins"] = len({s["logprob_margin"] for s in scores})
    values = np.array([s["score"] for s in scores])
    _, ties = np.unique(values, return_counts=True)
    summary["ties"] = {
        "unique_scores": len(ties),
        "rows_in_repeated_score_groups": int(ties[ties > 1].sum()),
        "largest_tie": int(ties.max()),
    }
    labels = np.array([int(r["metadata"]["ground_truth"]) for r in rows])
    summary["calibration_bins"] = []
    for index in range(10):
        selected = (values >= index / 10) & (
            (values < (index + 1) / 10) if index < 9 else (values <= 1)
        )
        if selected.any():
            summary["calibration_bins"].append(
                {
                    "low": index / 10,
                    "high": (index + 1) / 10,
                    "n": int(selected.sum()),
                    "mean_score": float(values[selected].mean()),
                    "positive_fraction": float(labels[selected].mean()),
                }
            )
    summary["weighted_ood"] = {
        metric: sum(g[metric] * g["n"] for g in summary["by_source"]) / len(scores)
        for metric in ("auroc", "pauroc_at_20")
    }
    summary["completed_at_utc"] = datetime.now(UTC).isoformat()
    summary["artifact_sha256"] = {
        p.name: file_hash(p) for p in (frozen, scores_path, root / "attempts.jsonl")
    }
    # Generic teacher helper reports missing provider cost as zero; it is unknown.
    summary["usage"]["reported_cost_usd"] = None
    for group in summary["request_setting_groups"]:
        group["reported_cost_usd"] = None
    atomic_write_json(root / "summary.json", summary)
    print(
        json.dumps(
            {
                "rows": len(scores),
                "mean_ood_auroc": summary["mean_source_auroc"],
                "mean_ood_pauroc_at_20": summary["mean_source_pauroc_at_20"],
                "accounting": summary["campaign_accounting"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
