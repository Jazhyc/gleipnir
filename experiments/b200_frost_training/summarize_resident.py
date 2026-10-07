"""Analyze complete frozen-cohort optimizer updates, including every timed sample."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path


def summarize(root: Path) -> dict:
    config = json.loads(Path(__file__).with_name("config.json").read_text())[
        "full_model_screen"
    ]
    receipts = {
        name: json.loads((root / name / "receipt.json").read_text())
        for name, _ in config["trials"]
    }
    reference = receipts["01baseline"]
    agreement = {}
    timings = {}
    control_samples = []
    for name, variant in config["trials"]:
        receipt = receipts[name]
        samples = receipt["step_seconds"]
        if (
            receipt["status"] != "complete"
            or len(samples) != 20
            or any(not math.isfinite(x) or x <= 0 for x in samples)
        ):
            raise ValueError(f"incomplete/nonfinite trial: {name}")
        timed = samples[10:]
        tokens = sum(
            r["tokens"] for r in receipt["physical_contract"] if 11 <= r["update"] <= 20
        )
        timings[name] = {
            "mean_seconds": statistics.mean(timed),
            "median_seconds": statistics.median(timed),
            "samples_seconds": timed,
            "actual_tokens_per_second": tokens / sum(timed),
            "actual_timed_tokens": tokens,
        }
        if variant == "baseline":
            control_samples.extend(timed)
        agreement[name] = {
            key: receipt[key] == reference[key]
            for key in (
                "initial_master_sha256",
                "final_master_sha256",
                "physical_contract",
                "loss_history",
                "pid",
            )
        }
        agreement[name].update(
            warm=receipt["measured_updates_warm"],
            uninstrumented=not receipt["instrumented"],
            optimizer_reset=receipt["optimizer_reset"],
        )
    validation = json.loads((root / "03direct/candidate_validation.json").read_text())
    execution = json.loads((root / "03direct/candidate_execution.json").read_text())
    control_mean = statistics.mean(control_samples)
    direct_mean = timings["03direct"]["mean_seconds"]
    reduction = 1 - direct_mean / control_mean
    accepted = validation["accepted_for_timing"] and all(
        all(checks.values()) for checks in agreement.values()
    )
    return {
        "status": "complete",
        "worker_pid": reference["pid"],
        "timing_scope": (
            "complete optimizer updates 11--20; "
            "loading, checks, priming and exports excluded"
        ),
        "trials": timings,
        "agreement": agreement,
        "valid_frozen_screen": accepted,
        "pooled_control_mean_seconds": control_mean,
        "direct_mean_seconds": direct_mean,
        "relative_time_reduction": reduction,
        "relative_throughput_increase": control_mean / direct_mean - 1,
        "required_relative_time_reduction": config["minimum_relative_improvement"],
        "eligible_for_selection": accepted
        and reduction >= config["minimum_relative_improvement"],
        "automatic_promotion": False,
        "direct_host_calls_in_gate": validation["direct_host_calls_in_gate"],
        "direct_host_calls_in_training": execution["direct_calls"]
        - validation["binding_state"]["direct_calls"],
        "targeted_validation": validation,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    result = summarize(args.root)
    (args.root / "comparison.json").write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                k: v
                for k, v in result.items()
                if k not in {"trials", "targeted_validation", "agreement"}
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
