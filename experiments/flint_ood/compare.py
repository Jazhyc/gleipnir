"""Compare Flint with Qwen3.8-27B on the strict OOD suite: quality tables,
and throughput against the stored Qwen3.8-27B run (vLLM 0.31). Reads collected
results only; light enough for the VPS."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/flint_ood"
RUNS = {
    "flint": OUT / "flint",
    "qwen38_vllm031": ROOT / "results/qwen27b_generations_ood/qwen38_27b",
}
SOURCES = [
    "agentdojo",
    "cua_shade_arena",
    "iac",
    "mle_sabotage",
    "rogue_eval",
    "shade_arena",
]


def load_predictions(path: Path) -> dict[str, dict]:
    with (path / "predictions.jsonl").open() as handle:
        return {row["id"]: row for row in map(json.loads, handle)}


def quality(result: dict, predictions: dict[str, dict]) -> dict:
    macro = result["metrics"]["macro"]
    scores = np.array([row["score"] for row in predictions.values()])
    _, counts = np.unique(scores, return_counts=True)
    return {
        "mean_ood_pauroc_at_20": macro["macro"]["pauroc_at_20"],
        "macro_auroc": macro["macro"]["auroc"],
        "macro_brier": macro["macro"]["brier"],
        "macro_balanced_accuracy": macro["macro"]["balanced_accuracy"],
        "tied_rows": int(counts[counts > 1].sum()),
        "by_source": {
            group["group"]: {
                "pauroc_at_20": group["pauroc_at_20"],
                "auroc": group["auroc"],
            }
            for group in macro["groups"]
        },
        "prompt_tokens_total": result["prompt_tokens"]["total"],
    }


def throughput(result: dict) -> dict:
    runtime = result["runtime"]
    # The Qwen runner times the canary and the full suite together; Flint's
    # scorer times them separately and excludes its extra diagnostics.
    seconds = runtime.get("elapsed_seconds_this_invocation")
    if seconds is None:
        seconds = (
            runtime["batched_canary_seconds"]
            + runtime["full_suite_seconds_this_invocation"]
        )
    rows, tokens = result["rows"], result["prompt_tokens"]["total"]
    return {
        "scoring_seconds": seconds,
        "rows_per_second": rows / seconds,
        "prompt_tokens_per_second": tokens / seconds,
        "engine_initialization_seconds": runtime["engine_initialization_seconds"],
        "vllm_version": runtime["vllm_version"],
    }


def drift(a: dict[str, dict], b: dict[str, dict]) -> dict:
    ids = sorted(a)
    if ids != sorted(b):
        raise ValueError("prediction ids differ")
    x = np.array([a[i]["score"] for i in ids])
    y = np.array([b[i]["score"] for i in ids])
    return {
        "score_mae": float(np.abs(x - y).mean()),
        "score_correlation": float(np.corrcoef(x, y)[0, 1]),
        "rows_flipping_at_0.5": int(((x >= 0.5) != (y >= 0.5)).sum()),
    }


def main() -> None:
    report: dict = {"runs": {}}
    predictions = {}
    for name, path in RUNS.items():
        if not (path / "result.json").is_file():
            continue
        result = json.loads((path / "result.json").read_text())
        predictions[name] = load_predictions(path)
        if len(predictions[name]) != 6395:
            raise ValueError(f"{name} is incomplete")
        report["runs"][name] = {
            "quality": quality(result, predictions[name]),
            "throughput": throughput(result),
            "gpu_name": result["runtime"]["gpu_name"],
        }
    runs = report["runs"]
    reference = "qwen38_vllm031"
    if {"flint", reference} <= set(runs):
        flint, qwen = runs["flint"], runs[reference]
        report["flint_vs_qwen38"] = {
            "reference": reference,
            "mean_ood_pauroc_at_20_difference": flint["quality"][
                "mean_ood_pauroc_at_20"
            ]
            - qwen["quality"]["mean_ood_pauroc_at_20"],
            "competitive_at_0.8701": flint["quality"]["mean_ood_pauroc_at_20"]
            >= 0.8701,
            "rows_per_second_ratio": flint["throughput"]["rows_per_second"]
            / qwen["throughput"]["rows_per_second"],
            "score_agreement": drift(predictions["flint"], predictions[reference]),
        }
    (OUT / "comparison.json").write_text(json.dumps(report, indent=2) + "\n")

    names = list(runs)
    print("| strict OOD | " + " | ".join(names) + " |")
    print("|---|" + "---:|" * len(names))
    for label, key in (
        ("Mean-OOD pAUROC@20", "mean_ood_pauroc_at_20"),
        ("macro AUROC", "macro_auroc"),
        ("macro Brier", "macro_brier"),
        ("macro balanced accuracy", "macro_balanced_accuracy"),
    ):
        print(
            f"| {label} | "
            + " | ".join(f"{runs[n]['quality'][key]:.4f}" for n in names)
            + " |"
        )
    print(
        "| tied rows | "
        + " | ".join(f"{runs[n]['quality']['tied_rows']:,}" for n in names)
        + " |"
    )
    for source in SOURCES:
        print(
            f"| {source} pAUROC@20 | "
            + " | ".join(
                f"{runs[n]['quality']['by_source'][source]['pauroc_at_20']:.4f}"
                for n in names
            )
            + " |"
        )
    for label, key, fmt in (
        ("scoring seconds", "scoring_seconds", "{:,.0f}"),
        ("rows/s", "rows_per_second", "{:.2f}"),
        ("prompt tokens/s", "prompt_tokens_per_second", "{:,.0f}"),
    ):
        print(
            f"| {label} | "
            + " | ".join(fmt.format(runs[n]["throughput"][key]) for n in names)
            + " |"
        )
    if "flint_vs_qwen38" in report:
        print("flint_vs_qwen38", json.dumps(report["flint_vs_qwen38"]))


if __name__ == "__main__":
    main()
