"""Audit and finalize the saved first pass after the user shortened the run."""

from __future__ import annotations

import hashlib
import json
import math
import subprocess

import pandas as pd

from experiments.local_inference.core import CONFIG, DATA, ROOT, read_rows, write_json
from gleipnir.binary_evaluation import metric_views
from gleipnir.qwen35_adapter_rebase import sha256_file


def main() -> None:
    output = ROOT / "baseline"
    if (output / "result.json").exists():
        raise FileExistsError("completed result must not be overwritten")
    config = json.loads(CONFIG.read_text())
    launch = json.loads((output / "launch_config.json").read_text())
    if config["repeats"] != 1 or dict(launch, repeats=1) != config:
        raise ValueError("only the requested repeat-count change is allowed")
    predictions = json.loads((output / "predictions_0.json").read_text())
    timing = json.loads((output / "repeat_0.json").read_text())
    rows = read_rows()
    manifest = json.loads((DATA / "manifest.json").read_text())
    subset_hash = sha256_file(DATA / "subset.jsonl")
    if subset_hash != manifest["subset_sha256"]:
        raise ValueError("subset changed")
    for row, prediction in zip(rows, predictions, strict=True):
        for field in ("id", "source", "label", "prompt_sha256", "tokens"):
            if row[field] != prediction[field]:
                raise ValueError(f"prediction alignment changed: {field}")
        margin = prediction["logprobs"][1] - prediction["logprobs"][0]
        if not math.isfinite(margin) or margin != prediction["logit_margin"]:
            raise ValueError("invalid logprobs")
        score = 1 / (1 + math.exp(-max(-80, min(80, margin))))
        if abs(score - prediction["score"]) > 1e-12:
            raise ValueError("saved score differs from decision logprobs")
    frame = pd.DataFrame(predictions).rename(columns={"source": "dataset"})
    metrics = metric_views(frame)
    if metrics != timing["metrics"]:
        raise ValueError("saved metrics do not reproduce")
    total = sum(r["tokens"] for r in rows)
    if not math.isclose(total / timing["seconds"], timing["prompt_tokens_per_second"]):
        raise ValueError("throughput does not reproduce")
    telemetry = [
        json.loads(line)
        for line in (output / "gpu_telemetry.jsonl").read_text().splitlines()
    ]
    samples = [s["gpu"] for s in telemetry if s["status"].get("repeat") == 0]
    reference = json.loads((ROOT / "reference.json").read_text())
    serving = json.loads((output / "serving_parity.json").read_text())
    if not reference["passed"] or not serving["passed"]:
        raise ValueError("canary did not pass")
    if (
        reference["subset_sha256"] != subset_hash
        or reference["config_sha256"] != sha256_file(output / "launch_config.json")
        or reference["merge_manifest_sha256"]
        != sha256_file(ROOT / "merged_bf16/merge_manifest.json")
    ):
        raise ValueError("reference identity mismatch")
    runner_commit = "29a17bd"
    runner = subprocess.check_output(
        [
            "git",
            "show",
            f"{runner_commit}:experiments/local_inference/benchmark.py",
        ]
    )
    result = {
        "config": config,
        "launch_config": launch,
        "config_sha256": sha256_file(CONFIG),
        "launch_config_sha256": sha256_file(output / "launch_config.json"),
        "subset_sha256": subset_hash,
        "merge_manifest_sha256": sha256_file(ROOT / "merged_bf16/merge_manifest.json"),
        "reference_sha256": sha256_file(ROOT / "reference.json"),
        "rows": len(rows),
        "prompt_tokens": total,
        "completed_passes": 1,
        "repeats": [timing],
        "median_seconds": timing["seconds"],
        "median_prompt_tokens_per_second": timing["prompt_tokens_per_second"],
        "median_scores": [p["score"] for p in predictions],
        "metrics": metrics,
        "repeat_stability": None,
        "stop_reason": (
            "User requested one pass during execution; "
            "saved pass audited after stopping worker"
        ),
        "initialization_seconds": None,
        "initialization_note": (
            "LLM wall time not persisted before stop; "
            "vLLM logged engine initialization at 82.48 seconds"
        ),
        "runner_commit": runner_commit,
        "runner_sha256": hashlib.sha256(runner).hexdigest(),
        "thermal_telemetry": {
            "coverage": "partial first pass; started after throttling was observed",
            "samples": len(samples),
            "max_temperature_c": max(float(s["temperature.gpu"]) for s in samples),
            "sm_clock_min_mhz": min(float(s["clocks.current.sm"]) for s in samples),
            "sm_clock_max_mhz": max(float(s["clocks.current.sm"]) for s in samples),
            "software_thermal_throttle_fraction": sum(
                s["clocks_throttle_reasons.sw_thermal_slowdown"] == "Active"
                for s in samples
            )
            / len(samples),
        },
        "predictions_sha256": sha256_file(output / "predictions_0.json"),
    }
    write_json(output / "result.json", result)
    write_json(output / "status.json", {"state": "complete", "completed_passes": 1})
    write_json(ROOT / "status.json", {"state": "complete", "completed_passes": 1})
    print(
        json.dumps(
            {
                k: result[k]
                for k in (
                    "rows",
                    "prompt_tokens",
                    "median_seconds",
                    "median_prompt_tokens_per_second",
                    "thermal_telemetry",
                )
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
