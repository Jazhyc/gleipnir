"""Repeat-relative release gates (user decision, 2026-10-20).

The frozen per-row gates in ``validate.py`` fail on every split because the
optimized stack's own run-to-run noise exceeds them. A split instead passes
when, against two same-host passes ``a`` and ``b`` from ``validate.py``:

1. MAE(a, recorded) <= 1.25 x MAE(a, b);
2. corr(a, recorded) >= corr(a, b) - 0.001;
3. |pAUROC@20(a) - pAUROC@20(recorded)| <= 0.005.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from experiments.release_serving.targets import ROOT, TARGETS
from gleipnir.data.monitoring import read_rows, write_json

MAE_RATIO = 1.25
CORRELATION_MARGIN = 0.001
PAUROC_DIFFERENCE = 0.005


def agreement(a: dict[str, float], b: dict[str, float]) -> tuple[float, float]:
    if set(a) != set(b):
        raise ValueError("passes cover different rows")
    ids = sorted(a)
    x = np.array([a[i] for i in ids])
    y = np.array([b[i] for i in ids])
    return float(np.abs(x - y).mean()), float(np.corrcoef(x, y)[0, 1])


def gate(
    pass_a: dict[str, float],
    pass_b: dict[str, float],
    recorded: dict[str, float],
    pauroc_a: float,
    recorded_pauroc: float,
) -> dict:
    recorded_mae, recorded_corr = agreement(pass_a, recorded)
    repeat_mae, repeat_corr = agreement(pass_a, pass_b)
    result = {
        "rows": len(pass_a),
        "recorded_mae": recorded_mae,
        "repeat_mae": repeat_mae,
        "mae_ratio": recorded_mae / repeat_mae if repeat_mae else float("inf"),
        "recorded_correlation": recorded_corr,
        "repeat_correlation": repeat_corr,
        "pauroc_difference": pauroc_a - recorded_pauroc,
    }
    result["passed"] = (
        recorded_mae <= MAE_RATIO * repeat_mae
        and recorded_corr >= repeat_corr - CORRELATION_MARGIN
        and abs(result["pauroc_difference"]) <= PAUROC_DIFFERENCE
    )
    return result


def scores(path: Path) -> dict[str, float]:
    return {r["id"]: float(r["score"]) for r in read_rows(path)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", choices=sorted(TARGETS))
    parser.add_argument("--split", choices=("id", "ood"), default="id")
    parser.add_argument("--pass-a", type=Path, required=True, help="validate report")
    parser.add_argument("--pass-b", type=Path, required=True, help="validate report")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    target = TARGETS[args.model]
    reports = [json.loads(p.read_text()) for p in (args.pass_a, args.pass_b)]
    rows = [
        p.parent / p.stem / f"{args.split}.jsonl" for p in (args.pass_a, args.pass_b)
    ]
    split = reports[0]["splits"][args.split]
    result = gate(
        scores(rows[0]),
        scores(rows[1]),
        scores(ROOT / target.validation[args.split][1]),
        split["pauroc_at_20"],
        split["recorded_pauroc_at_20"],
    )
    report = {"model": args.model, "split": args.split, **result}
    write_json(args.output, report)
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
