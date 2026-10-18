"""Measure one BF16 serving condition of the trained 27B, then retire it.

Launches the BF16 monitor server with a chosen worker class, checks the BF16
audit, warms up, and times the frozen full320 workload at concurrency 128
(repeats) and the quick64 workload at concurrency 1. ``compare`` reports
throughput ratios, score parity and AUROC deviation against a named control.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from experiments.b200_monitor_score.run import trial
from gleipnir.data.monitoring import file_hash, write_json
from gleipnir.serving import bf16_launch
from gleipnir.serving.benchmark import (
    measurement_summary,
    paired_score_summary,
    ranking_comparison,
)

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
DATA = ROOT / "data/b200_inference_benchmark"
OUT = ROOT / "results/qwen38_27b_serving"
LOGS = ROOT / "logs/runpod/qwen38_27b_serving"
MODEL = Path("/tmp/gleipnir-merged/qwen38-27b-s0-01")
SETTINGS = {"port": 8010, "timeout_seconds": 1800}


def workloads() -> dict[str, list[dict]]:
    manifest = json.loads((DATA / "manifest.json").read_text())
    loaded = {}
    for key in ("quick", "full"):
        path = DATA / f"{key}.json"
        if file_hash(path) != manifest["files"][key]:
            raise ValueError(f"frozen workload drift: {key}")
        loaded[key] = json.loads(path.read_text())
    return loaded


def sources() -> dict[str, str]:
    paths = sorted((ROOT / "src/gleipnir/serving").glob("*.py"))
    paths += sorted(HERE.glob("*.py"))
    return {str(p.relative_to(ROOT)): file_hash(p) for p in paths}


async def measure(out: Path, repeats: int) -> None:
    loads = workloads()
    await trial(loads["quick"], 128, SETTINGS)  # warmup and compile
    for repeat in range(repeats):
        rows, seconds = await trial(loads["full"], 128, SETTINGS)
        write_json(
            out / f"c128_repeat{repeat}.json", {"rows": rows, "seconds": seconds}
        )
        summary = measurement_summary(rows, seconds)
        print(
            f"c128 repeat {repeat}: "
            f"{summary['prompt_tokens_per_second']:.0f} prompt tok/s, "
            f"{summary['requests_per_second']:.3f} req/s",
            flush=True,
        )
    rows, seconds = await trial(loads["quick"], 1, SETTINGS)
    write_json(out / "c1_repeat0.json", {"rows": rows, "seconds": seconds})


def run(
    name: str,
    worker: str,
    additional: dict,
    repeats: int,
    quantization: str | None = None,
) -> None:
    out = OUT / name
    out.mkdir(parents=True, exist_ok=False)
    audit = out / "bf16_audit.json"
    selection, command = bf16_launch.bf16_command(
        ROOT, MODEL, sources(), worker=worker, additional=additional
    )
    if quantization is not None:
        # Same server and harness; the quantized server registers the config.
        command[command.index("-m") + 1] = "experiments.qwen38_27b_serving.quant_server"
        command += ["--quantization", quantization]
    env = bf16_launch.environment(ROOT, selection, audit)
    process = bf16_launch.start(
        ROOT, command, env, LOGS / f"{name}.log", out / "server.json"
    )
    try:
        startup = bf16_launch.wait_ready(process, SETTINGS["port"])
        asyncio.run(measure(out, repeats))
        native = json.loads(audit.read_text())
        if not native["passed"] or len(native["attention_calls"]) != native.get(
            "attention_layers"
        ):
            raise ValueError("BF16 model/attention audit incomplete")
        write_json(
            out / "condition.json",
            {
                "worker": worker,
                "additional": additional,
                "startup_seconds": startup,
                "repeats": repeats,
                "model": str(MODEL),
            },
        )
    finally:
        bf16_launch.stop(process)
    print(f"condition_complete {name}", flush=True)


def load_repeats(name: str) -> list[list[dict]]:
    return [
        json.loads(p.read_text())["rows"]
        for p in sorted((OUT / name).glob("c128_repeat*.json"))
    ]


def compare(control: str, candidate: str) -> dict:
    def throughput(name: str) -> list[float]:
        return [
            measurement_summary(**json.loads(p.read_text()))["prompt_tokens_per_second"]
            for p in sorted((OUT / name).glob("c128_repeat*.json"))
        ]

    def c1(name: str) -> dict:
        return measurement_summary(
            **json.loads((OUT / name / "c1_repeat0.json").read_text())
        )

    base, cand = load_repeats(control), load_repeats(candidate)
    rows = workloads()["full"]
    report = {
        "control": control,
        "candidate": candidate,
        "c128_prompt_tokens_per_second": {
            control: throughput(control),
            candidate: throughput(candidate),
        },
        "c1_p50_seconds": {
            n: c1(n)["latency"]["p50_seconds"] for n in (control, candidate)
        },
        "scores": paired_score_summary(base, cand),
        "bitwise_equal": all(
            [r["score"] for r in a] == [r["score"] for r in b]
            for a, b in zip(base, cand, strict=False)
        ),
        "ranking": ranking_comparison(rows, base, cand),
    }
    t = report["c128_prompt_tokens_per_second"]
    report["c128_mean_ratio"] = (sum(t[candidate]) / len(t[candidate])) / (
        sum(t[control]) / len(t[control])
    )
    write_json(OUT / f"compare_{candidate}_vs_{control}.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    r = sub.add_parser("run")
    r.add_argument("name")
    r.add_argument("--worker", default=bf16_launch.BF16_WORKER)
    r.add_argument("--additional", default="{}")
    r.add_argument("--repeats", type=int, default=3)
    r.add_argument("--quantization")
    c = sub.add_parser("compare")
    c.add_argument("control")
    c.add_argument("candidate")
    args = parser.parse_args()
    if args.action == "run":
        run(
            args.name,
            args.worker,
            json.loads(args.additional),
            args.repeats,
            args.quantization,
        )
    else:
        report = compare(args.control, args.candidate)
        print(
            json.dumps(
                {
                    k: report[k]
                    for k in ("c128_mean_ratio", "c1_p50_seconds", "bitwise_equal")
                },
                indent=1,
            )
        )
        print(json.dumps(report["scores"], indent=1)[:800])


if __name__ == "__main__":
    main()
