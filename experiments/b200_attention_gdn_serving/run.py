"""Matched high-concurrency attention/GDN serving trials on one B200."""

import argparse
import asyncio
import json
import statistics
from pathlib import Path

import httpx
import yaml

from experiments.b200_inference_benchmark.run import EXPERIMENT as BENCHMARK
from experiments.b200_inference_benchmark.run import (
    ROOT,
    benchmark,
    prepared_manifest,
    sha,
    trial,
    verify_merged_model,
    write,
)
from gleipnir.inference_benchmark import (
    measurement_summary,
    paired_score_summary,
    ranking_comparison,
)

EXPERIMENT = Path(__file__).parent
OUTPUT = ROOT / "results/b200_attention_gdn_serving"
SOURCES = [
    "src/gleipnir/vllm_frost_fp4.py",
    "src/gleipnir/cudnn_fp4_gemm.py",
    "src/gleipnir/cudnn_fp4_epilogue.py",
    "src/gleipnir/cudnn_fp4_mlp.py",
    "src/gleipnir/native_fp4_training.py",
    "src/gleipnir/nvfp4_pack.py",
    "experiments/b200_frost_inference/worker.py",
    "experiments/b200_attention_gdn_serving/worker.py",
    "src/gleipnir/serving_precision.py",
]


def resolve_condition(condition: dict, hashes: dict) -> dict:
    attention_backend = condition.get("attention_backend", "FLASHINFER")
    if attention_backend not in {"FLASHINFER", "FLASH_ATTN"}:
        raise ValueError("unsupported attention backend")
    if attention_backend == "FLASH_ATTN" and (
        condition["attention_precision"] != "bf16"
        or not condition["worker_cls"].endswith("Fa4ServingAuditWorker")
    ):
        raise ValueError("Blackwell FA4 requires BF16 and its native dispatch audit")
    if condition["attention_precision"] not in {"bf16", "fp8_e4m3", "nvfp4"}:
        raise ValueError("unsupported attention precision")
    if condition["gdn_projection_precision"] not in {"bf16", "fp8", "fp4"}:
        raise ValueError("unsupported GDN projection precision")
    if condition["gdn_backend"] not in {"flashinfer", "cutedsl", "flashqla"}:
        raise ValueError("unsupported GDN prefill backend")
    if condition["gdn_backend"] != "flashinfer" and (
        not condition.get("gdn_validation")
        or not condition["worker_cls"].endswith("GdnServingAuditWorker")
    ):
        raise ValueError("alternative GDN requires its validation and audited worker")
    if (
        condition["gdn_projection_precision"] in {"fp8", "fp4"}
        and condition["quantization"]
        != {"fp8": "gleipnir_frost_gdn", "fp4": "gleipnir_frost_gdn_fp4"}[
            condition["gdn_projection_precision"]
        ]
    ):
        raise ValueError(
            "GDN precision requires the matching mixed-precision quantizer"
        )
    overrides = condition["serving_config_overrides"]
    if set(overrides) - {
        "max_num_seqs",
        "gpu_memory_utilization",
        "max_num_batched_tokens",
    }:
        raise ValueError("override would change the frozen data/scoring contract")
    args = [
        "--quantization",
        condition["quantization"],
        "--worker-cls",
        condition["worker_cls"],
        "--attention-backend",
        attention_backend,
        "--additional-config",
        json.dumps(
            {"gleipnir_frost_fp4": hashes, "serving_condition": condition},
            sort_keys=True,
        ),
    ]
    if condition["attention_precision"] == "fp8_e4m3":
        args.extend(["--kv-cache-dtype", "fp8_e4m3", "--calculate-kv-scales"])
    elif condition["attention_precision"] == "nvfp4":
        args.extend(["--kv-cache-dtype", "nvfp4"])
    if condition.get("profiler_config"):
        args.extend(["--profiler-config", json.dumps(condition["profiler_config"])])
    return {
        **condition,
        "extra_server_args": args,
        "startup_audit": "results/b200_attention_gdn_serving/native_attention.json",
    }


async def high_concurrency(
    config: dict, manifest: dict, condition: dict, out: Path
) -> None:
    rows = json.loads((ROOT / "data/b200_inference_benchmark/full.json").read_text())
    assert len(rows) == 320
    report = {
        "status": "running",
        "rows": 320,
        "trials": [],
        "comparison": {},
        "manifest_sha256": sha(ROOT / "data/b200_inference_benchmark/manifest.json"),
        "server": json.loads((OUTPUT / "server.json").read_text()),
        "serving_config_overrides": condition["serving_config_overrides"],
        "worker_reused": True,
        "parity_reused_from": str(out / "http_parity.json"),
    }
    write(out / "high_summary.json", report)
    async with httpx.AsyncClient(
        base_url=f"http://127.0.0.1:{config['port']}",
        timeout=300,
        trust_env=False,
        limits=httpx.Limits(max_connections=256),
    ) as client:
        for c in condition["high_concurrency"]:
            candidate = []
            for repeat in (0, 1):
                values, seconds = await trial(client, rows, manifest["token_ids"], c)
                candidate.append(values)
                write(out / f"high_c{c}_repeat{repeat}.json", values)
                measurement = {
                    "concurrency": c,
                    "repeat": repeat,
                    **measurement_summary(values, seconds),
                }
                report["trials"].append(measurement)
                write(out / "high_summary.json", report)
                print(json.dumps(measurement), flush=True)
            if condition.get("high_reference"):
                ref = ROOT / condition["high_reference"]
                before = json.loads((ref / "high_summary.json").read_text())
                if (
                    before["status"] != "complete"
                    or before["manifest_sha256"] != report["manifest_sha256"]
                    or before["serving_config_overrides"]
                    != report["serving_config_overrides"]
                ):
                    raise ValueError("unmatched high-concurrency reference")
                baseline = [
                    json.loads((ref / f"high_c{c}_repeat{i}.json").read_text())
                    for i in (0, 1)
                ]
                old = statistics.median(
                    t["prompt_tokens_per_second"]
                    for t in before["trials"]
                    if t["concurrency"] == c
                )
                new = statistics.median(
                    t["prompt_tokens_per_second"]
                    for t in report["trials"]
                    if t["concurrency"] == c
                )
                report["comparison"][str(c)] = {
                    "throughput_ratio": new / old,
                    "paired_scores": paired_score_summary(baseline, candidate),
                    "ranking_metrics": ranking_comparison(rows, baseline, candidate),
                }
                report["high_reference_summary_sha256"] = sha(ref / "high_summary.json")
            else:
                report["comparison"][str(c)] = {
                    "ranking_metrics": ranking_comparison(rows, candidate, candidate)
                }
            write(out / "high_summary.json", report)
    report["status"] = "complete"
    write(out / "high_summary.json", report)
    print("high_concurrency_complete server_retained=true", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--condition", type=Path, default=EXPERIMENT / "fp4_gdn_projection.json"
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--reuse-server", action="store_true")
    args = parser.parse_args()
    args.condition = args.condition.resolve()
    if Path(args.output).name != args.output:
        raise ValueError("output must be a directory name")
    base = yaml.safe_load((BENCHMARK / "config.yaml").read_text())
    manifest = prepared_manifest(base)
    raw = json.loads(args.condition.read_text())
    sources = list(SOURCES)
    if ".combined_worker." in raw["worker_cls"]:
        sources.append("experiments/b200_attention_gdn_serving/combined_worker.py")
    if raw.get("attention_backend") == "FLASH_ATTN":
        sources.append("experiments/b200_attention_gdn_serving/fa4_worker.py")
    if raw["gdn_backend"] != "flashinfer":
        sources.extend(
            [
                "src/gleipnir/serving_gdn_kernels.py",
                "src/gleipnir/flashqla_training.py",
                "experiments/b200_attention_gdn_serving/gdn_worker.py",
                "experiments/b200_attention_gdn_serving/gdn_canary.py",
            ]
        )
        if raw["gdn_backend"] == "flashqla":
            sources.append("experiments/b200_attention_gdn_serving/gdn_server.py")
    if raw["gdn_projection_precision"] in {"fp8", "fp4"}:
        sources.extend(
            [
                "src/gleipnir/vllm_frost_gdn.py",
                "experiments/b200_attention_gdn_serving/mixed_worker.py",
                "experiments/b200_attention_gdn_serving/server.py",
            ]
        )
        if raw["gdn_projection_precision"] == "fp4":
            sources.append("src/gleipnir/vllm_frost_gdn_fp4.py")
    condition = resolve_condition(raw, {p: sha(ROOT / p) for p in sources})
    condition["config_sha256"] = sha(args.condition)
    config = {
        **base,
        **condition["serving_config_overrides"],
        "port": condition["port"],
        "gdn_prefill_backend": "cutedsl"
        if condition["gdn_backend"] == "cutedsl"
        else "flashinfer",
    }
    out = OUTPUT / args.output
    out.mkdir(parents=True, exist_ok=False)
    for source in [
        *EXPERIMENT.glob("*"),
        *BENCHMARK.glob("*"),
        args.condition,
        *(ROOT / p for p in sources),
        ROOT / "src/gleipnir/inference_benchmark.py",
    ]:
        if source.is_file():
            target = out / "executed_sources" / source.relative_to(ROOT)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(source.read_bytes())
    write(out / "condition.json", condition)
    write(out / "manifest.json", manifest)
    try:
        merged = Path(condition["merged_model"])
        write(out / "merged_artifact.json", verify_merged_model(base, merged))
        asyncio.run(
            benchmark(config, manifest, out, 64, args.reuse_server, merged, condition)
        )
        asyncio.run(high_concurrency(config, manifest, condition, out))
    except BaseException as error:
        write(out / "failure.json", {"error": f"{type(error).__name__}: {error}"})
        raise
    finally:
        for name in ("loaded_precision.json", "native_attention.json"):
            if (OUTPUT / name).exists():
                write(out / name, json.loads((OUTPUT / name).read_text()))


if __name__ == "__main__":
    main()
