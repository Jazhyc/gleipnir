#!/usr/bin/env python3
"""Gate master-adapter versus rebased-vLLM OOD canary parity."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def load_predictions(path: Path, prefix: str) -> pd.DataFrame:
    """Normalize eager and resumable OOD-serving prediction schemas."""
    frame = pd.read_json(path, lines=True)
    if {"source", "id", "label", "score"}.issubset(frame):
        frame = frame.rename(columns={"source": "dataset", "id": "index"})
    required = {"dataset", "index", "label", "score"}
    if missing := required - set(frame):
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    return frame[list(required)].rename(columns={"score": f"score_{prefix}"})


def compare_predictions(eager_path: Path, vllm_path: Path) -> dict[str, Any]:
    eager = load_predictions(eager_path, "eager")
    served = load_predictions(vllm_path, "vllm")
    keys = ["dataset", "index", "label"]
    merged = eager.merge(served, on=keys, how="outer", validate="one_to_one")
    if len(merged) != len(eager) or len(merged) != len(served):
        raise ValueError("eager and vLLM prediction identities differ")
    eager_scores = merged["score_eager"].to_numpy(dtype=np.float64)
    vllm_scores = merged["score_vllm"].to_numpy(dtype=np.float64)
    if not np.isfinite(eager_scores).all() or not np.isfinite(vllm_scores).all():
        raise ValueError("parity predictions contain non-finite scores")
    absolute = np.abs(eager_scores - vllm_scores)
    return {
        "rows": len(merged),
        "mean_absolute_score_difference": float(absolute.mean()),
        "max_absolute_score_difference": float(absolute.max()),
        "pearson_score_correlation": float(
            np.corrcoef(eager_scores, vllm_scores)[0, 1]
        ),
    }


def maximum_adapter_effect(base_path: Path, adapter_path: Path) -> float:
    base = load_predictions(base_path, "base")
    adapter = load_predictions(adapter_path, "adapter")
    keys = ["dataset", "index", "label"]
    merged = base.merge(adapter, on=keys, validate="one_to_one")
    return float(
        (merged["score_base"].astype(float) - merged["score_adapter"].astype(float))
        .abs()
        .max()
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eager-root", type=Path, required=True)
    parser.add_argument("--vllm-root", type=Path, required=True)
    parser.add_argument("--model-size", required=True)
    parser.add_argument("--job-name", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    base_relative = Path(args.model_size) / "base" / "base" / "predictions.jsonl"
    adapter_relative = (
        Path(args.model_size) / "adapters" / args.job_name / "predictions.jsonl"
    )
    base = compare_predictions(
        args.eager_root / base_relative, args.vllm_root / base_relative
    )
    adapter = compare_predictions(
        args.eager_root / adapter_relative, args.vllm_root / adapter_relative
    )
    eager_effect = maximum_adapter_effect(
        args.eager_root / base_relative, args.eager_root / adapter_relative
    )
    vllm_effect = maximum_adapter_effect(
        args.vllm_root / base_relative, args.vllm_root / adapter_relative
    )
    passed = (
        base["mean_absolute_score_difference"] <= 0.02
        and adapter["mean_absolute_score_difference"] <= 0.02
        and base["pearson_score_correlation"] >= 0.99
        and adapter["pearson_score_correlation"] >= 0.99
        and eager_effect >= 1e-6
        and vllm_effect >= 1e-6
    )
    report = {
        "passed": passed,
        "model_size": args.model_size,
        "job_name": args.job_name,
        "thresholds": {
            "max_mean_absolute_score_difference": 0.02,
            "min_pearson_score_correlation": 0.99,
            "min_maximum_adapter_effect": 1e-6,
        },
        "base": base,
        "adapter": adapter,
        "maximum_adapter_effect": {"eager": eager_effect, "vllm": vllm_effect},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    if not passed:
        raise RuntimeError("distilled OOD serving parity gate failed")


if __name__ == "__main__":
    main()
