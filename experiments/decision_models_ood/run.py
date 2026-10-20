"""Run and summarize decision models on the frozen strict-OOD suite.

Every model gets the frozen Jev Noul question through OpenRouter's Decisions
API, pinned to its own provider with fallbacks disabled. Phases per model:
``canary`` (100 non-OOD rows; interface gate), ``longest`` (the longest OOD
state, label-blind) and ``full`` (all 6,395 OOD rows). Caches are resumable and
keyed by state hash and request settings.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import pandas as pd
from dotenv import load_dotenv

from experiments.tool_trajectory_monitoring.teacher_canary import (
    atomic_write_json,
    atomic_write_jsonl,
    load_jsonl,
    sha256_file,
)
from gleipnir.evaluation.metrics import evaluate_binary_monitor
from gleipnir.evaluation.openrouter_decisions import (
    DecisionRecord,
    DecisionsConfig,
    OpenRouterDecisionsError,
    request_settings_sha256,
    score_decision,
    stable_sha256,
)

ROOT = Path(__file__).resolve().parents[2]
CONFIG = Path(__file__).with_name("config.json")
OUT = ROOT / "results/decision_models_ood"
PHASES = ("canary", "longest", "full")
OOD_SOURCES = {
    "agentdojo": 1_500,
    "cua_shade_arena": 955,
    "iac": 1_443,
    "mle_sabotage": 148,
    "rogue_eval": 1_499,
    "shade_arena": 850,
}


def load_config() -> dict[str, Any]:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    if config.get("status") != "frozen_evaluation":
        raise ValueError("config must be frozen before paid calls")
    return config


def decisions_config(config: dict[str, Any], model: str) -> DecisionsConfig:
    contract = json.loads((ROOT / config["question"]["path"]).read_text())
    if stable_sha256(contract["question"]) != config["question"]["question_sha256"]:
        raise ValueError("question contract hash differs from frozen config")
    spec = config["models"][model]
    request = config["request"]
    return DecisionsConfig(
        model=spec["openrouter_id"],
        provider=spec["provider"],
        question_id=contract["question_id"],
        question=contract["question"],
        endpoint=request["endpoint"],
        data_collection=request["data_collection"],
        request_timeout=float(request["request_timeout_seconds"]),
        max_retries=int(request["max_retries"]),
    )


def load_records(config: dict[str, Any], phase: str) -> list[DecisionRecord]:
    spec = config["inputs"][phase]
    path = ROOT / spec["path"]
    if sha256_file(path) != spec["sha256"]:
        raise ValueError(f"{phase} input checksum differs from frozen config")
    rows = load_jsonl(path)
    if phase == "longest":
        rows = [row for row in rows if str(row["id"]) == spec["record_id"]]
    if len(rows) != int(spec["rows"]):
        raise ValueError(f"{phase} row count differs from frozen config")
    records = [
        DecisionRecord(str(r["id"]), str(r["state"]), dict(r["metadata"])) for r in rows
    ]
    for record, row in zip(records, rows, strict=True):
        if record.state_sha256 != row["state_sha256"]:
            raise ValueError(f"state hash drift for id={record.record_id!r}")
    if len({r.record_id for r in records}) != len(records):
        raise ValueError(f"duplicate IDs in {phase} input")
    if phase == "full":
        counts = Counter(r.metadata["source_dataset"] for r in records)
        if counts != Counter(OOD_SOURCES):
            raise ValueError(f"strict-OOD source counts differ: {counts!r}")
    return records


def phase_dir(model: str, phase: str) -> Path:
    return OUT / model / phase


def load_cache(
    path: Path, records: dict[str, DecisionRecord], settings_sha256: str
) -> dict[str, dict[str, Any]]:
    cached: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return cached
    for row in load_jsonl(path):
        record = records.get(str(row["id"]))
        if record is None or str(row["id"]) in cached:
            raise ValueError(f"unknown or duplicate cached ID {row['id']!r} in {path}")
        if row["state_sha256"] != record.state_sha256:
            raise ValueError(f"cached state mismatch for id={row['id']!r}")
        if row["request_settings_sha256"] != settings_sha256:
            raise ValueError(f"cached request settings mismatch for id={row['id']!r}")
        cached[str(row["id"])] = row
    return cached


def reported_cost(rows: list[dict[str, Any]]) -> float:
    return float(sum(float(r["usage"].get("cost") or 0.0) for r in rows))


def run_phase(config: dict[str, Any], model: str, phase: str) -> None:
    if phase == "full":
        canary = phase_dir(model, "canary") / "summary.json"
        if not canary.exists() or not json.loads(canary.read_text())["gate"]["passed"]:
            raise RuntimeError("the canary gate must pass before the full run")
    records_list = load_records(config, phase)
    records = {r.record_id: r for r in records_list}
    decision_config = decisions_config(config, model)
    settings_sha256 = request_settings_sha256(decision_config)
    directory = phase_dir(model, phase)
    scores_path = directory / "scores.jsonl"
    rejections_path = directory / "rejections.jsonl"
    cached = load_cache(scores_path, records, settings_sha256)
    rejected = load_cache(rejections_path, records, settings_sha256)
    pending = [r for r in records_list if r.record_id not in cached | rejected]
    if not pending:
        print(f"{model}/{phase}: all {len(records_list)} rows complete", flush=True)
        return
    load_dotenv(ROOT / ".env")
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is not set")
    spec = config["models"][model]
    ceiling = (
        float(spec["maximum_full_cost_usd"])
        if phase == "full"
        else float(config["inputs"][phase]["maximum_cost_usd"])
    )
    maximum_rejected = int(config["rejections"]["maximum_fraction"] * len(records))
    order = {r.record_id: i for i, r in enumerate(records_list)}
    workers = int(config["request"]["concurrency"])

    def persist() -> None:
        atomic_write_jsonl(
            scores_path, sorted(cached.values(), key=lambda r: order[str(r["id"])])
        )
        atomic_write_jsonl(
            rejections_path,
            sorted(rejected.values(), key=lambda r: order[str(r["id"])]),
        )

    print(
        f"{model}/{phase}: {len(cached)} cached, {len(rejected)} rejected, "
        f"{len(pending)} pending, concurrency={workers}",
        flush=True,
    )
    batch_size = workers * int(config["request"]["checkpoint_every"]) // 4
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for start in range(0, len(pending), batch_size):
            batch = pending[start : start + batch_size]
            futures = {
                executor.submit(score_decision, r, decision_config, api_key): r
                for r in batch
            }
            failures = []
            for future in as_completed(futures):
                record = futures[future]
                try:
                    cached[record.record_id] = future.result()
                except OpenRouterDecisionsError as error:
                    if not error.input_rejected:
                        failures.append(f"{record.record_id}: {error}")
                        continue
                    rejected[record.record_id] = {
                        "id": record.record_id,
                        "state_sha256": record.state_sha256,
                        "state_chars": len(record.state),
                        "metadata": record.metadata,
                        "request_settings_sha256": settings_sha256,
                        "error": str(error)[:2000],
                    }
            persist()
            cost = reported_cost(list(cached.values()))
            print(
                f"{model}/{phase}: {len(cached)}/{len(records_list)} scored, "
                f"{len(rejected)} rejected; reported_cost=${cost:.6f}",
                flush=True,
            )
            if failures:
                raise RuntimeError("batch failed: " + " | ".join(failures[:5]))
            if cost > ceiling:
                raise RuntimeError(f"reported cost ${cost:.6f} exceeded ${ceiling}")
            if len(rejected) > maximum_rejected:
                raise RuntimeError(f"{len(rejected)} input rejections exceed the limit")


def metrics(frame: pd.DataFrame) -> dict[str, Any]:
    by_source = evaluate_binary_monitor(frame, group_column="source")
    pooled = evaluate_binary_monitor(frame.assign(g="pooled"), group_column="g")
    return {
        "mean_source_pauroc_at_20": by_source["macro"]["pauroc_at_20"],
        "mean_source_auroc": by_source["macro"]["auroc"],
        "macro_brier": by_source["macro"]["brier"],
        "pooled": pooled["groups"][0],
        "by_source": by_source["groups"],
    }


def summarize_phase(config: dict[str, Any], model: str, phase: str) -> dict[str, Any]:
    records = load_records(config, phase)
    spec = config["models"][model]
    directory = phase_dir(model, phase)
    scores = load_jsonl(directory / "scores.jsonl")
    rejections_path = directory / "rejections.jsonl"
    rejections = load_jsonl(rejections_path) if rejections_path.exists() else []
    by_id = {r["id"]: r for r in scores}
    rejected_ids = {r["id"] for r in rejections}
    if set(by_id) | rejected_ids != {r.record_id for r in records} or (
        set(by_id) & rejected_ids
    ):
        raise ValueError("scores and rejections do not exactly cover the records")
    uninformative = float(config["rejections"]["uninformative_score"])
    frame = pd.DataFrame(
        [
            {
                "id": r.record_id,
                "source": r.metadata["source_dataset"],
                "label": int(r.metadata["ground_truth"]),
                "score": float(by_id[r.record_id]["score"])
                if r.record_id in by_id
                else uninformative,
                "answered": r.record_id in by_id,
            }
            for r in records
        ]
    )
    input_tokens = sum(int(r["usage"]["input_tokens"]) for r in scores)
    price = float(spec["input_price_usd_per_million"])
    providers = dict(Counter(str(r["provider"]) for r in scores))
    models = dict(Counter(str(r["model"]) for r in scores))
    summary: dict[str, Any] = {
        "model": model,
        "phase": phase,
        "openrouter_id": spec["openrouter_id"],
        "rows": len(records),
        "answered": len(scores),
        "rejected": len(rejections),
        "scores_sha256": sha256_file(directory / "scores.jsonl"),
        "providers": providers,
        "models": models,
        "usage": {
            "input_tokens": input_tokens,
            "output_tokens": sum(int(r["usage"]["output_tokens"]) for r in scores),
            "maximum_input_tokens": max(
                (int(r["usage"]["input_tokens"]) for r in scores), default=0
            ),
            "reported_cost_usd": reported_cost(scores),
            "list_price_usd_per_million": price,
            "list_price_cost_usd": input_tokens * price / 1e6,
            "list_price_cost_usd_per_1000": input_tokens
            * price
            / 1e6
            / len(records)
            * 1000,
        },
        "attempts": dict(Counter(int(r["attempts"]) for r in scores)),
        "score_diagnostics": {
            "unique_scores": int(frame.loc[frame["answered"], "score"].nunique()),
            "minimum": float(frame.loc[frame["answered"], "score"].min()),
            "maximum": float(frame.loc[frame["answered"], "score"].max()),
        },
    }
    if phase == "longest":
        summary["score_withheld"] = True
        del summary["score_diagnostics"]
    else:
        summary["metrics"] = metrics(frame)
        if rejections:
            summary["answered_only_metrics"] = metrics(frame.loc[frame["answered"]])
    revision_pinned = models == {spec["provider_revision"]: len(scores)}
    provider_pinned = providers == {spec["provider"]: len(scores)}
    if phase == "canary":
        conditions = {
            "complete_without_rejections": len(scores) == len(records),
            "provider_pinned": provider_pinned,
            "revision_pinned": revision_pinned,
            "within_cost": summary["usage"]["reported_cost_usd"]
            <= float(config["inputs"]["canary"]["maximum_cost_usd"]),
        }
        summary["gate"] = {"passed": all(conditions.values()), **conditions}
    else:
        summary["provider_pinned"] = provider_pinned
        summary["revision_pinned"] = revision_pinned
    atomic_write_json(directory / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("run", "summarize"))
    parser.add_argument("model")
    parser.add_argument("phase", choices=PHASES)
    args = parser.parse_args()
    config = load_config()
    if args.model not in config["models"]:
        raise ValueError(f"unknown model {args.model!r}")
    if args.command == "run":
        run_phase(config, args.model, args.phase)
    summary = summarize_phase(config, args.model, args.phase)
    printable = {k: v for k, v in summary.items() if k not in ("metrics",)}
    if "metrics" in summary:
        printable["mean_source_pauroc_at_20"] = summary["metrics"][
            "mean_source_pauroc_at_20"
        ]
        printable["mean_source_auroc"] = summary["metrics"]["mean_source_auroc"]
    print(json.dumps(printable, indent=1, default=str))


if __name__ == "__main__":
    main()
