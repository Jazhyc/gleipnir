"""Summarize all frozen-reference repeats without masking failed candidates."""

import argparse
import json
import statistics
from pathlib import Path

from experiments.b200_inference_benchmark.run import ROOT, write
from gleipnir.serving.reference import selected_score_reference


def summarize(directory: Path, baseline: dict) -> dict:
    """Bind median timing and complete quality diagnostics to one finished run."""
    report = json.loads((directory / "summary.json").read_text())
    if report["status"] != "complete":
        raise ValueError(f"candidate is not complete: {directory.name}")
    result = {
        "name": directory.name,
        "runtime": report["runtime"],
        "diagnostic_mode": report["finite_canary_diagnostic"],
        "promoted": report["promoted"],
        "canary": json.loads((directory / "canary.json").read_text()),
        "conditions": [],
    }
    for concurrency in (1, 128):
        candidate = [x for x in report["trials"] if x["concurrency"] == concurrency]
        control = [x for x in baseline["trials"] if x["concurrency"] == concurrency]
        if len(candidate) != (3 if concurrency == 1 else 6) or len(control) != len(
            candidate
        ):
            raise ValueError("frozen repeat counts changed")

        def timing(rows):
            return {
                "prompt_tokens_per_second": statistics.median(
                    r["prompt_tokens_per_second"] for r in rows
                ),
                "requests_per_second": statistics.median(
                    r["requests_per_second"] for r in rows
                ),
                "p50_seconds": statistics.median(
                    r["latency"]["p50_seconds"] for r in rows
                ),
                "p95_seconds": statistics.median(
                    r["latency"]["p95_seconds"] for r in rows
                ),
                "prompt_throughput_range": [
                    min(r["prompt_tokens_per_second"] for r in rows),
                    max(r["prompt_tokens_per_second"] for r in rows),
                ],
            }

        old, new = timing(control), timing(candidate)
        quality = json.loads(
            (directory / f"c{concurrency}_comparison.json").read_text()
        )
        result["conditions"].append(
            {
                "concurrency": concurrency,
                "baseline": old,
                "candidate": new,
                "throughput_change_percent": 100
                * (
                    new["prompt_tokens_per_second"] / old["prompt_tokens_per_second"]
                    - 1
                ),
                "auroc_delta_percentage_points": {
                    key: value * 100 if value is not None else None
                    for key, value in quality["ranking"]["auroc_delta"].items()
                    if key != "per_source"
                },
                "scores": quality["scores"],
                "quality_artifact": str(
                    (directory / f"c{concurrency}_comparison.json").relative_to(ROOT)
                ),
            }
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("names", nargs="+")
    args = parser.parse_args()
    selection = selected_score_reference(ROOT)
    baseline = json.loads((ROOT / selection["results"] / "summary.json").read_text())
    for name in args.names:
        if Path(name).name != name:
            raise ValueError("run name must be a stem")
        directory = ROOT / "results/b200_vllm031" / name
        summary = summarize(directory, baseline)
        write(directory / "comparison_summary.json", summary)
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
