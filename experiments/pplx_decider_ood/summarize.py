"""Apply the frozen gates and summarize pplx-decider scores on the Jev Noul states."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from gleipnir.data.monitoring import file_hash, read_rows, write_json
from gleipnir.evaluation.metrics import (
    evaluate_binary_monitor,
    normalized_partial_auroc,
)

ROOT = Path(__file__).resolve().parents[2]
CONFIG = json.loads((ROOT / "experiments/pplx_decider_ood/config.json").read_text())
OUT = ROOT / "results/pplx_decider_ood"


def stem(phase: str, path: str, question: str = "jev") -> str:
    return f"{phase}_{path}" if question == "jev" else f"{question}_{phase}_{path}"


def joined(model: str, phase: str, path: str, question: str = "jev") -> pd.DataFrame:
    """Scores joined to frozen labels and sources by id and state hash."""
    states = read_rows(ROOT / CONFIG["inputs"][phase]["path"])
    scores = {
        row["id"]: row
        for row in read_rows(OUT / model / f"{stem(phase, path, question)}.jsonl")
    }
    if len(scores) != len(states) or set(scores) != {row["id"] for row in states}:
        raise ValueError(
            f"{model} {phase}_{path} does not cover the frozen rows exactly"
        )
    frame = []
    for state in states:
        row = scores[state["id"]]
        if row["state_sha256"] != state["state_sha256"]:
            raise ValueError(f"state mismatch for {state['id']}")
        frame.append(
            {
                "id": state["id"],
                "source": state["metadata"]["source_dataset"],
                "label": int(state["metadata"]["ground_truth"]),
                "margin": float(row["margin"]),
                "score": float(row["noul"]),
                "input_tokens": int(row["input_tokens"]),
            }
        )
    return pd.DataFrame(frame)


def ranking(frame: pd.DataFrame) -> dict:
    """Per-source and macro ranking on the raw logit margin (no saturation ties)."""
    sources = {}
    for source, subset in frame.groupby("source", sort=True):
        if subset["label"].nunique() < 2:
            sources[source] = None
            continue
        labels, margins = subset["label"].to_numpy(), subset["margin"].to_numpy()
        sources[source] = {
            "auroc": float(roc_auc_score(labels, margins)),
            "pauroc_at_20": normalized_partial_auroc(labels, margins),
        }
    defined = [v for v in sources.values() if v]
    return {
        "by_source": sources,
        "mean_auroc": float(np.mean([v["auroc"] for v in defined])),
        "mean_pauroc_at_20": float(np.mean([v["pauroc_at_20"] for v in defined])),
        "pooled_auroc": float(roc_auc_score(frame["label"], frame["margin"])),
        "pooled_pauroc_at_20": normalized_partial_auroc(
            frame["label"].to_numpy(), frame["margin"].to_numpy()
        ),
    }


def summarize(model: str, phase: str, path: str, question: str = "jev") -> dict:
    name = stem(phase, path, question)
    frame = joined(model, phase, path, question)
    rank = ranking(frame)
    calibrated = evaluate_binary_monitor(frame, group_column="source")
    tokens = int(frame["input_tokens"].sum())
    price = CONFIG["models"][model]["input_price_usd_per_million"]
    cost = tokens * price / 1_000_000
    summary = {
        "model": model,
        "phase": phase,
        "path": path,
        "question": question,
        "rows": len(frame),
        "scores_sha256": file_hash(OUT / model / f"{name}.jsonl"),
        "ranking_on_margin": rank,
        "calibrated_noul": {
            "by_source": calibrated["groups"],
            "macro": calibrated["macro"],
        },
        "score_diagnostics": {
            "unique_margins": int(frame["margin"].nunique()),
            "unique_noul": int(frame["score"].nunique()),
            "noul_min": float(frame["score"].min()),
            "noul_max": float(frame["score"].max()),
        },
        "usage": {
            "input_tokens": tokens,
            "max_input_tokens": int(frame["input_tokens"].max()),
            "list_price_input_usd_per_million": price,
            "list_price_cost_usd": cost,
            "list_price_cost_usd_per_1000": cost / len(frame) * 1_000,
        },
    }
    if phase == "canary":
        gates = CONFIG["gates"]
        source_aurocs = [v["auroc"] for v in rank["by_source"].values() if v]
        conditions = {
            "complete": len(frame) == CONFIG["inputs"]["canary"]["rows"],
            "pooled_auroc": rank["pooled_auroc"] >= gates["canary_pooled_auroc_min"],
            "source_auroc": min(source_aurocs) >= gates["canary_source_auroc_min"],
            "unique_scores": summary["score_diagnostics"]["unique_margins"]
            >= gates["canary_unique_scores_min"],
        }
        summary["gate"] = {"passed": all(conditions.values()), "conditions": conditions}
    write_json(OUT / model / f"{name}.summary.json", summary)
    return summary


def parity(model: str) -> dict:
    """Fast packed path against the bit-stable reference path on the canary."""
    fast = joined(model, "canary", "fast").set_index("id")
    reference = joined(model, "canary", "reference").set_index("id").loc[fast.index]
    gates = CONFIG["gates"]
    pearson = float(np.corrcoef(fast["margin"], reference["margin"])[0, 1])
    max_delta = float((fast["score"] - reference["score"]).abs().max())
    report = {
        "model": model,
        "margin_pearson": pearson,
        "max_abs_noul_delta": max_delta,
        "mean_abs_margin_delta": float(
            (fast["margin"] - reference["margin"]).abs().mean()
        ),
        "token_counts_equal": bool(
            (fast["input_tokens"] == reference["input_tokens"]).all()
        ),
        "passed": pearson >= gates["parity_margin_pearson_min"]
        and max_delta <= gates["parity_max_abs_probability_delta"],
    }
    write_json(OUT / model / "canary_parity.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=tuple(CONFIG["models"]), required=True)
    parser.add_argument("--phase", choices=("canary", "full"), required=True)
    parser.add_argument("--question", choices=("jev", "teacher"), default="jev")
    args = parser.parse_args()
    if args.phase == "canary" and args.question == "jev":
        for path in ("reference", "fast"):
            print(path, json.dumps(summarize(args.model, "canary", path)["gate"]))
        print("parity", json.dumps(parity(args.model)))
    elif args.phase == "canary":
        result = summarize(args.model, "canary", "fast", args.question)
        print("fast", json.dumps(result["gate"]))
    else:
        result = summarize(args.model, "full", "fast", args.question)
        print(json.dumps({k: result[k] for k in ("rows", "usage")}, indent=1))
        print("mean_pauroc_at_20", result["ranking_on_margin"]["mean_pauroc_at_20"])
