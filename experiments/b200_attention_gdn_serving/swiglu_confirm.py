"""Warm throughput, latency and AUROC against the checksum-bound reference."""

import argparse
import asyncio
import json
import statistics

import httpx

from experiments.b200_inference_benchmark.run import (
    ROOT,
    resolve_kernel_baseline,
    sha,
    trial,
    write,
)
from gleipnir.inference_benchmark import (
    measurement_summary,
    paired_score_summary,
    ranking_comparison,
)


async def main(args) -> None:
    root = ROOT / "results/b200_attention_gdn_serving"
    selected = json.loads(
        (ROOT / "experiments/b200_inference_benchmark/baseline.json").read_text()
    )
    resolve_kernel_baseline({"baseline": "selected"})
    before = ROOT / selected["confirmation_results"]
    assert sha(before / "summary.json") == selected["confirmation_summary_sha256"]
    original = root / f"{args.name}_serving01"
    initial = json.loads((original / "high_summary.json").read_text())
    resumed = root / (args.resume or f"{args.name}_sweep_resume01") / "summary.json"
    if initial["status"] != "complete":
        resume = json.loads(resumed.read_text())
        assert resume["status"] == "complete"
        assert resume["original_high_summary_sha256"] == sha(
            original / "high_summary.json"
        )
        assert (
            len(initial["trials"]) == 7
            and resume["concurrency"] == 128
            and resume["repeat"] == 1
        )
    server = json.loads((root / "server.json").read_text())
    assert args.worker in " ".join(server["command"])
    precision = json.loads((root / "loaded_precision.json").read_text())
    if args.mode == "overhead":
        from gleipnir.serving_fp4_swiglu_overhead_validation import validate_runtime

        validate_runtime(
            json.loads((root / "native_swiglu_overhead.json").read_text()),
            precision["worker_pid"],
            args.validation,
        )
    else:
        from gleipnir.serving_fp4_swiglu_native_output_validation import (
            validate_runtime,
        )

        validate_runtime(
            json.loads((root / "native_swiglu_output.json").read_text()),
            precision["worker_pid"],
            args.validation,
        )
    manifest = json.loads((original / "manifest.json").read_text())
    rows = json.loads((ROOT / "data/b200_inference_benchmark/full.json").read_text())
    out = root / f"{args.name}_confirmation01"
    out.mkdir(exist_ok=False)
    report = {
        "status": "running",
        "server": server,
        "worker_reused": True,
        "unchanged_checks_reused_from": str(original),
        "diagnostic_only": True,
        "manifest_sha256": sha(ROOT / "data/b200_inference_benchmark/manifest.json"),
        "baseline_summary_sha256": sha(before / "summary.json"),
        "initial_sweep_status": initial["status"],
        "sweep_resume_summary_sha256": sha(resumed) if resumed.exists() else None,
        "trials": [],
    }

    async def measured(population, concurrency):
        async with httpx.AsyncClient(
            base_url="http://127.0.0.1:8010",
            timeout=300,
            trust_env=False,
            limits=httpx.Limits(max_connections=256, max_keepalive_connections=128),
        ) as client:
            return await trial(client, population, manifest["token_ids"], concurrency)

    warmup, seconds = await measured(rows, 128)
    write(out / "warmup.json", warmup)
    report["excluded_warmup_seconds"] = seconds
    write(out / "summary.json", report)
    values = []
    for repeat in range(5):
        predictions, seconds = await measured(rows, 128)
        values.append(predictions)
        write(out / f"c128_repeat{repeat}.json", predictions)
        result = {
            "concurrency": 128,
            "repeat": repeat,
            **measurement_summary(predictions, seconds),
        }
        report["trials"].append(result)
        write(out / "summary.json", report)
        print("throughput", json.dumps(result), flush=True)
    baseline = [
        json.loads((before / f"c128_repeat{i}.json").read_text()) for i in range(5)
    ]
    report.update(
        median_tokens_s=statistics.median(
            t["prompt_tokens_per_second"] for t in report["trials"]
        ),
        baseline_median_tokens_s=selected["warm_reference_tokens_per_second"],
        paired_scores=paired_score_summary(baseline, values),
        ranking_metrics=ranking_comparison(rows, baseline, values),
        status="complete",
    )
    report["warm_baseline_throughput_ratio"] = (
        report["median_tokens_s"] / report["baseline_median_tokens_s"]
    )
    write(out / "summary.json", report)
    out = root / f"{args.name}_latency_confirmation01"
    out.mkdir(exist_ok=False)
    rows = json.loads((ROOT / "data/b200_inference_benchmark/quick.json").read_text())
    latency = {
        "status": "running",
        "server": server,
        "worker_reused": True,
        "manifest_sha256": report["manifest_sha256"],
        "rows": len(rows),
        "trials": [],
    }
    for repeat in range(2):
        predictions, seconds = await measured(rows, 1)
        write(out / f"c1_repeat{repeat}.json", predictions)
        result = {
            "concurrency": 1,
            "repeat": repeat,
            **measurement_summary(predictions, seconds),
        }
        latency["trials"].append(result)
        write(out / "summary.json", latency)
        print("latency", json.dumps(result), flush=True)
    latency.update(
        median_latency_seconds=statistics.median(
            t["latency"]["p50_seconds"] for t in latency["trials"]
        ),
        median_p95_seconds=statistics.median(
            t["latency"]["p95_seconds"] for t in latency["trials"]
        ),
        status="complete",
    )
    latency["latency_ratio_to_selected"] = (
        latency["median_latency_seconds"] / selected["warm_reference_latency_seconds"]
    )
    write(out / "summary.json", latency)
    print(
        "confirmation_complete",
        args.name,
        report["median_tokens_s"],
        latency["median_latency_seconds"],
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--mode", choices=["overhead", "native"], required=True)
    parser.add_argument("--worker", required=True)
    parser.add_argument("--validation", required=True)
    parser.add_argument(
        "--resume", help="explicit receipt directory for a resumed sweep"
    )
    args = parser.parse_args()
    if "/" in args.name or args.name in {".", ".."}:
        parser.error("name must be a directory stem")
    asyncio.run(main(args))
