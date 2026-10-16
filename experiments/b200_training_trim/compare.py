"""Matched per-update timing and ID score agreement for the trimmed run."""

from __future__ import annotations

import json
import statistics

from experiments.b200_training_trim.prepare import ARM, CONTROL, ROOT, TIMING_CONTROLS
from experiments.teacher_seed_variation.compare import load, score_difference, summary
from gleipnir.data.monitoring import write_json

WARM = 10  # Updates 1-10 include first-shape compilation.


def metadata(name: str) -> dict:
    path = ROOT / f"results/{name}/4b/monitor/causal_adapter/training_metadata.json"
    return json.loads(path.read_text())


def partitions(meta: dict) -> list:
    return [r["logical_indices"] for r in meta["adaptive_microbatching"]["records"]]


def timing(meta: dict) -> dict:
    durations = meta["optimizer_step_timing"]["durations_seconds"][WARM:]
    return {
        "updates": len(durations),
        "median": statistics.median(durations),
        "mean": statistics.fmean(durations),
        "total": sum(durations),
        "peak_allocated_gib": max(
            r["peak_allocated_bytes"] for r in meta["adaptive_microbatching"]["records"]
        )
        / 2**30,
    }


def compare() -> dict:
    trim = metadata(ARM)
    controls = {name: metadata(name) for name in TIMING_CONTROLS}
    for name, meta in controls.items():
        if partitions(meta) != partitions(trim):
            raise ValueError(f"physical partitions differ from {name}")
    trim_timing = timing(trim)
    control_timing = {name: timing(meta) for name, meta in controls.items()}
    control_median = statistics.median(t["median"] for t in control_timing.values())
    control_total = statistics.median(t["total"] for t in control_timing.values())
    control_rows = load(f"results/{CONTROL}/evaluation/id.jsonl", "id")
    trim_rows = load(f"results/{ARM}/evaluation/id.jsonl", "id")
    report = {
        "canary": trim["last_layer_trim"]["canary"],
        "trimmed_calls": trim["last_layer_trim"]["training"]["trimmed_calls"],
        "timing": {
            "trim": trim_timing,
            "controls": control_timing,
            "median_update_ratio": trim_timing["median"] / control_median,
            "warm_total_ratio": trim_timing["total"] / control_total,
        },
        "id": {
            "control": summary(control_rows),
            "trim": summary(trim_rows),
            "scores": score_difference(trim_rows, control_rows),
        },
    }
    write_json(ROOT / "results/b200_training_trim/comparison.json", report)
    print(json.dumps(report["timing"], indent=1), report["id"]["scores"], flush=True)
    return report


if __name__ == "__main__":
    compare()
