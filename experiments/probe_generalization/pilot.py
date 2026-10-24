"""Evaluate the frozen pilot gates and tabulate the step-budget sweep.

Gates (README "Pilot"):

- parity: prompted-backbone margins on the 128 OOD parity rows against the
  archived stock-vLLM base-4B margins, Pearson correlation >= 0.999;
- FP8 (a): every BF16-trained probe scores the FP8 copy with logit correlation
  >= 0.999 and |delta macro pAUROC@20| <= 0.005, at every checkpoint;
- FP8 (b): per architecture, objective and checkpoint, the seed-median macro
  pAUROC@20 of FP8-trained probes (scored on FP8) and BF16-trained probes
  (scored on BF16) differ by at most 0.005.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
PARITY_MIN_CORRELATION = 0.999
LOGIT_MIN_CORRELATION = 0.999
MAX_PAUROC_DELTA = 0.005


def parity(store_root: Path) -> dict:
    path = store_root / "pilot/prompted/parity/rows.jsonl"
    rows = [json.loads(line) for line in path.open()]
    ours = np.array([r["margin"] for r in rows])
    reference = np.array([r["reference_margin"] for r in rows])
    correlation = float(np.corrcoef(ours, reference)[0, 1])
    return {
        "rows": len(rows),
        "correlation": correlation,
        "mae": float(np.abs(ours - reference).mean()),
        "max_abs": float(np.abs(ours - reference).max()),
        "passed": correlation >= PARITY_MIN_CORRELATION,
    }


def pauroc(entry: dict, fmt: str) -> float:
    return entry[fmt]["macro"]["pauroc_at_20"]


def gate_a(run: Path) -> dict:
    summary = json.loads((run / "summary.json").read_text())
    failures, worst_corr, worst_delta = [], 1.0, 0.0
    for epoch, checkpoint in summary["checkpoints"].items():
        bf16 = json.loads((run / f"logits_e{epoch}_bf16.json").read_text())
        fp8 = json.loads((run / f"logits_e{epoch}_fp8.json").read_text())
        if bf16["ids"] != fp8["ids"]:
            raise ValueError("BF16 and FP8 evaluation rows differ")
        for name, entry in checkpoint["probes"].items():
            corr = float(np.corrcoef(bf16["logits"][name], fp8["logits"][name])[0, 1])
            delta = abs(pauroc(entry, "fp8") - pauroc(entry, "bf16"))
            worst_corr, worst_delta = min(worst_corr, corr), max(worst_delta, delta)
            if corr < LOGIT_MIN_CORRELATION or delta > MAX_PAUROC_DELTA:
                failures.append(
                    {"epoch": epoch, "probe": name, "correlation": corr, "delta": delta}
                )
    return {
        "worst_correlation": worst_corr,
        "worst_delta": worst_delta,
        "failures": failures,
        "passed": not failures,
    }


def seed_medians(run: Path, fmt: str) -> dict[tuple[str, str, str], float]:
    summary = json.loads((run / "summary.json").read_text())
    groups = defaultdict(list)
    for epoch, checkpoint in summary["checkpoints"].items():
        for entry in checkpoint["probes"].values():
            key = (epoch, entry["architecture"], entry["objective"])
            groups[key].append(pauroc(entry, fmt))
    return {key: float(np.median(values)) for key, values in groups.items()}


def gate_b(bf16_run: Path, fp8_run: Path) -> dict:
    bf16, fp8 = seed_medians(bf16_run, "bf16"), seed_medians(fp8_run, "fp8")
    if bf16.keys() != fp8.keys():
        raise ValueError("BF16 and FP8 sweeps differ")
    deltas = {"/".join(k): fp8[k] - bf16[k] for k in sorted(bf16)}
    failures = {k: v for k, v in deltas.items() if abs(v) > MAX_PAUROC_DELTA}
    return {
        "worst_abs_delta": max(abs(v) for v in deltas.values()),
        "failures": failures,
        "passed": not failures,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store-root", required=True)
    parser.add_argument("--name", required=True)
    args = parser.parse_args()
    results = ROOT / "results/probe_generalization" / args.name
    report = {"parity": parity(Path(args.store_root)), "backbones": {}}
    for backbone in ("prompted", "ntp"):
        bf16_run, fp8_run = results / f"{backbone}-bf16", results / f"{backbone}-fp8"
        report["backbones"][backbone] = {
            "fp8_gate_a": gate_a(bf16_run),
            "fp8_gate_b": gate_b(bf16_run, fp8_run),
            "budget_seed_median_bf16": {
                "/".join(k): v
                for k, v in sorted(seed_medians(bf16_run, "bf16").items())
            },
        }
    report["fp8_passed"] = all(
        b["fp8_gate_a"]["passed"] and b["fp8_gate_b"]["passed"]
        for b in report["backbones"].values()
    )
    (results / "pilot_summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "backbones"}, indent=2))
    for backbone, b in report["backbones"].items():
        print(
            backbone,
            "gate a",
            b["fp8_gate_a"]["passed"],
            "gate b",
            b["fp8_gate_b"]["passed"],
        )


if __name__ == "__main__":
    main()
