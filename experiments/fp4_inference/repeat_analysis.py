"""Compare matched repeated serving runs without changing the scoring runner."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from experiments.local_inference.core import compare, write_json
from gleipnir.qwen35_adapter_rebase import sha256_file

IDENTITY_FIELDS = (
    "subset_sha256",
    "merge_manifest_sha256",
    "reference_sha256",
    "rows",
    "prompt_tokens",
    "software",
    "gpu",
    "runner_sha256",
)
ROW_FIELDS = ("id", "source", "label", "prompt_sha256", "tokens")


def load_run(root: Path) -> tuple[dict, list[dict], np.ndarray, np.ndarray]:
    """Validate complete repeat coverage and stored medians before comparison."""
    result = json.loads((root / "result.json").read_text())
    rows, scores, margins = None, [], []
    for index, repeat in enumerate(result["repeats"]):
        if (
            repeat["repeat"] != index
            or not np.isfinite(repeat["seconds"])
            or repeat["seconds"] <= 0
        ):
            raise ValueError("Invalid repeat timing or order")
        current = json.loads((root / f"predictions_{index}.json").read_text())
        if len(current) != result["rows"] or len({r["id"] for r in current}) != len(
            current
        ):
            raise ValueError("Invalid prediction coverage")
        if rows is None:
            rows = current
        else:
            for left, right in zip(rows, current, strict=True):
                if any(left[k] != right[k] for k in ROW_FIELDS):
                    raise ValueError("Repeat prediction identity mismatch")
        scores.append([r["score"] for r in current])
        margins.append([r["logit_margin"] for r in current])
    matrix, margin_matrix = np.asarray(scores), np.asarray(margins)
    if (
        rows is None
        or not np.isfinite(matrix).all()
        or not np.isfinite(margin_matrix).all()
        or ((matrix < 0) | (matrix > 1)).any()
    ):
        raise ValueError("Missing or nonfinite repeat predictions")
    if not np.allclose(
        np.median(matrix, axis=0), result["median_scores"], atol=1e-12, rtol=0
    ):
        raise ValueError("Stored score medians do not match repeats")
    times = [r["seconds"] for r in result["repeats"]]
    if not np.isclose(np.median(times), result["median_seconds"], atol=1e-12, rtol=0):
        raise ValueError("Stored timing median does not match repeats")
    return result, rows, matrix, margin_matrix


def stability(scores: np.ndarray, margins: np.ndarray) -> dict:
    """Report raw within-engine score and margin ranges without an equivalence claim."""
    return {
        "passes": len(scores),
        "max_score_range": float(np.ptp(scores, axis=0).max()),
        "mean_score_range": float(np.ptp(scores, axis=0).mean()),
        "max_margin_range": float(np.ptp(margins, axis=0).max()),
        "threshold_unstable_rows": int(
            ((scores >= 0.5).any(0) != (scores >= 0.5).all(0)).sum()
        ),
    }


def analyze(baseline: Path, candidate: Path) -> dict:
    """Compare frozen identities, median scores, raw repeats and declared gates."""
    a, left, sa, ma = load_run(baseline)
    b, right, sb, mb = load_run(candidate)
    for key in IDENTITY_FIELDS:
        if a[key] != b[key]:
            raise ValueError(f"Unmatched run identity: {key}")
    if len(sa) != len(sb):
        raise ValueError("Matched comparison requires equal repeat counts")
    for x, y in zip(left, right, strict=True):
        if any(x[k] != y[k] for k in ROW_FIELDS):
            raise ValueError("Unmatched prediction identity")
    x, y = np.median(sa, axis=0), np.median(sb, axis=0)
    qa, qb = [r["metrics"]["macro"]["macro"] for r in (a, b)]
    quality = {
        "macro_auroc_loss": qa["auroc"] - qb["auroc"],
        "macro_brier_increase": qb["brier"] - qa["brier"],
        "macro_pauroc_delta": qb["pauroc_at_20"] - qa["pauroc_at_20"],
    }
    quality["passed"] = (
        quality["macro_auroc_loss"] <= 0.01 and quality["macro_brier_increase"] <= 0.01
    )
    speedup = a["median_seconds"] / b["median_seconds"]
    return {
        "baseline": str(baseline),
        "candidate": str(candidate),
        "analysis_sha256": sha256_file(Path(__file__)),
        "result_sha256": [
            sha256_file(root / "result.json") for root in (baseline, candidate)
        ],
        "identity": {k: a[k] for k in IDENTITY_FIELDS},
        "seconds": [[r["seconds"] for r in run["repeats"]] for run in (a, b)],
        "median_speedup": speedup,
        "stability": {"baseline": stability(sa, ma), "candidate": stability(sb, mb)},
        "score_drift": compare(x.tolist(), y.tolist()),
        "score_drift_by_source": {
            source: compare(
                x[[r["source"] == source for r in left]].tolist(),
                y[[r["source"] == source for r in right]].tolist(),
            )
            for source in sorted({r["source"] for r in left})
        },
        "quality_screen": quality,
        "candidate_serving_parity_passed": b["serving_parity_passed"],
        "development_selection_gates_passed": b["serving_parity_passed"]
        and quality["passed"]
        and speedup > 1.10,
        "metrics": {"baseline": a["metrics"], "candidate": b["metrics"]},
        "note": (
            "Repeated passes in one engine per condition; no confidence interval, "
            "population equivalence, or independent-engine repeatability claim."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    report = analyze(args.baseline, args.candidate)
    write_json(args.output, report)
    print(
        json.dumps(
            {k: report[k] for k in ("median_speedup", "quality_screen", "stability")},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
