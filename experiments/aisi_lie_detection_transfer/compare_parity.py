#!/usr/bin/env python3
"""Compare causal-master eager scores with rebased-adapter vLLM scores."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def load_predictions(path: Path, prefix: str) -> pd.DataFrame:
    frame = pd.read_json(path, lines=True)
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
    correlation = float(np.corrcoef(eager_scores, vllm_scores)[0, 1])
    return {
        "rows": len(merged),
        "mean_absolute_score_difference": float(absolute.mean()),
        "max_absolute_score_difference": float(absolute.max()),
        "pearson_score_correlation": correlation,
    }


def build_report(
    eager_root: Path,
    vllm_root: Path,
    *,
    strength_id: str,
    seed: int,
    max_mean_difference: float,
    min_correlation: float,
    min_adapter_effect: float,
) -> dict[str, Any]:
    relative = Path(strength_id) / f"seed{seed}" / "predictions.jsonl"
    base_relative = Path("base") / "base" / "predictions.jsonl"
    base = compare_predictions(eager_root / base_relative, vllm_root / base_relative)
    adapter = compare_predictions(eager_root / relative, vllm_root / relative)
    eager_base = load_predictions(eager_root / base_relative, "base")
    eager_adapter = load_predictions(eager_root / relative, "adapter")
    vllm_base = load_predictions(vllm_root / base_relative, "base")
    vllm_adapter = load_predictions(vllm_root / relative, "adapter")
    keys = ["dataset", "index", "label"]

    def maximum_effect(base_frame: pd.DataFrame, adapter_frame: pd.DataFrame) -> float:
        merged = base_frame.merge(adapter_frame, on=keys, validate="one_to_one")
        return float(
            np.abs(
                merged.filter(like="score_base").iloc[:, 0]
                - merged.filter(like="score_adapter").iloc[:, 0]
            ).max()
        )

    eager_effect = maximum_effect(eager_base, eager_adapter)
    vllm_effect = maximum_effect(vllm_base, vllm_adapter)
    passed = (
        base["mean_absolute_score_difference"] <= max_mean_difference
        and adapter["mean_absolute_score_difference"] <= max_mean_difference
        and base["pearson_score_correlation"] >= min_correlation
        and adapter["pearson_score_correlation"] >= min_correlation
        and eager_effect >= min_adapter_effect
        and vllm_effect >= min_adapter_effect
    )
    return {
        "passed": bool(passed),
        "thresholds": {
            "max_mean_absolute_score_difference": max_mean_difference,
            "min_pearson_score_correlation": min_correlation,
            "min_maximum_adapter_effect": min_adapter_effect,
        },
        "base": base,
        "adapter": adapter,
        "maximum_adapter_effect": {"eager": eager_effect, "vllm": vllm_effect},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eager-root", type=Path, required=True)
    parser.add_argument("--vllm-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--strength-id", default="0500")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-mean-difference", type=float, default=0.02)
    parser.add_argument("--min-correlation", type=float, default=0.99)
    parser.add_argument("--min-adapter-effect", type=float, default=1e-6)
    args = parser.parse_args()
    report = build_report(
        args.eager_root.resolve(),
        args.vllm_root.resolve(),
        strength_id=args.strength_id,
        seed=args.seed,
        max_mean_difference=args.max_mean_difference,
        min_correlation=args.min_correlation,
        min_adapter_effect=args.min_adapter_effect,
    )
    args.output.resolve().parent.mkdir(parents=True, exist_ok=True)
    args.output.resolve().write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    if not report["passed"]:
        raise RuntimeError("vLLM serving parity gate failed")


if __name__ == "__main__":
    main()
