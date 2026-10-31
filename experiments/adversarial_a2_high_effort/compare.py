"""Paired configuration comparison on the frozen selected task cohort."""

import json
from pathlib import Path

import numpy as np

from experiments.adversarial_a2_cover_comments.run import read
from gleipnir.adversarial import eval_sets
from gleipnir.adversarial.metrics import threshold_at_fpr


def main():
    here = Path(__file__).resolve().parent
    cfg = json.loads((here / "config.json").read_text())
    selected = set(cfg["task_ids"])
    out = eval_sets.ROOT / cfg["output"]
    low = eval_sets.ROOT / "results/adversarial_a2_cover_comments"
    baseline_path = eval_sets.ROOT / cfg["baseline"]
    assert eval_sets.file_sha256(baseline_path) == cfg["baseline_sha256"]
    baseline = {r["id"]: r for r in read(baseline_path)}
    honest = [r["score"] for r in baseline.values() if r["label"] == 0]
    by_effort = {}
    for name, directory in [("low", low), ("high", out)]:
        by_effort[name] = {
            r["id"]: r
            for r in read(directory / "predictions.jsonl")
            if str(r["task_id"]) in selected and r["status"] == "ok"
        }
    result = {}
    for arm in ("cover", "document"):
        parents = [
            r
            for r in baseline.values()
            if r["group"] == "vanilla" and str(r["task_id"]) in selected
        ]
        metrics = {}
        for fpr in cfg["fprs"]:
            threshold = threshold_at_fpr(honest, fpr)
            detected = [r for r in parents if r["score"] > threshold]
            intended, common = [], []
            counts = {
                "low_valid": 0,
                "high_valid": 0,
                "low_evasions": 0,
                "high_evasions": 0,
            }
            for parent in detected:
                records = [
                    by_effort[e].get(f"{arm}:{parent['id']}") for e in ("low", "high")
                ]
                flags = [
                    int(r is not None and r["score"] <= threshold) for r in records
                ]
                intended.append(flags[1] - flags[0])
                if all(r is not None for r in records):
                    common.append(flags[1] - flags[0])
                for effort, record, flag in zip(
                    ("low", "high"), records, flags, strict=True
                ):
                    counts[effort + "_valid"] += int(record is not None)
                    counts[effort + "_evasions"] += flag

            def estimate(values):
                if not values:
                    return {"n": 0, "difference": None, "bootstrap95": None}
                rng = np.random.default_rng(0)
                samples = rng.choice(values, (10000, len(values)), replace=True)
                return {
                    "n": len(values),
                    "difference": float(np.mean(values)),
                    "bootstrap95": np.quantile(
                        samples.mean(axis=1), [0.025, 0.975]
                    ).tolist(),
                }

            metrics[str(fpr)] = {
                "threshold": threshold,
                **counts,
                "intended_high_minus_low": estimate(intended),
                "common_valid_high_minus_low": estimate(common),
            }
        result[arm] = metrics
    (out / "effort_comparison.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
