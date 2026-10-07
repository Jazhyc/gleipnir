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
    resolve_kernel_baseline,
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
    "src/gleipnir/serving_compile_cache.py",
    "src/gleipnir/serving_runtime.py",
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
    if condition["attention_precision"] not in {"bf16", "fp8_e4m3", "nvfp4", "mxfp8"}:
        raise ValueError("unsupported attention precision")
    if condition["attention_precision"] == "mxfp8" and (
        attention_backend != "FLASHINFER"
        or not condition.get("mxfp8_validation")
        or not condition["worker_cls"].endswith("Mxfp8ServingAuditWorker")
    ):
        raise ValueError("MXFP8 requires its validated forward-only serving worker")
    if condition.get("fp4_preparation") and (
        condition["fp4_preparation"] not in {"vendor", "silu", "norm", "combined"}
        or not condition.get("fp4_prepare_validation")
        or not condition["worker_cls"].endswith("PreparationMxfp8ServingAuditWorker")
        or condition["attention_precision"] != "mxfp8"
        or condition["gdn_projection_precision"] != "fp4"
    ):
        raise ValueError("FP4 preparation requires its bound native receipt and worker")
    if condition.get("fp4_preparation") == "combined" and (
        not isinstance(condition["fp4_prepare_validation"], dict)
        or set(condition["fp4_prepare_validation"]) != {"vendor", "silu", "norm"}
        or not all(condition["fp4_prepare_validation"].values())
    ):
        raise ValueError("combined FP4 preparation requires all three native receipts")
    if condition.get("gemm_tuning_validation") and (
        condition.get("fp4_preparation") != "combined"
        or not condition["worker_cls"].endswith(
            "TunedPreparationMxfp8ServingAuditWorker"
        )
    ):
        raise ValueError("GEMM tuning requires the bound combined preparation worker")
    if condition["worker_cls"].endswith("TunedPreparationMxfp8ServingAuditWorker") and (
        not condition.get("gemm_tuning_validation")
    ):
        raise ValueError("GEMM tuning requires a native validation receipt")
    fusion_worker = condition["worker_cls"].endswith(
        "SwigluPreparationMxfp8ServingAuditWorker"
    )
    if bool(condition.get("swiglu_fusion_validation")) != fusion_worker or (
        fusion_worker
        and (
            condition.get("fp4_preparation") != "combined"
            or not condition.get("gemm_reference_validation")
            or condition.get("gemm_tuning_validation")
        )
    ):
        raise ValueError(
            "SwiGLU fusion requires its native receipt and tuned reference"
        )
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
        != {
            "fp8": "gleipnir_frost_gdn",
            "fp4": "gleipnir_frost_attention_fp4"
            if condition.get("attention_projection_precision") == "fp4"
            else "gleipnir_frost_gdn_fp4",
        }[condition["gdn_projection_precision"]]
    ):
        raise ValueError(
            "GDN precision requires the matching mixed-precision quantizer"
        )
    attention_projection_worker = condition["worker_cls"].endswith(
        "AttentionTunedPreparationMxfp8ServingAuditWorker"
    )
    if bool(
        condition.get("attention_projection_precision")
    ) != attention_projection_worker or (
        attention_projection_worker
        and (
            condition.get("attention_projection_precision") != "fp4"
            or not condition.get("attention_projection_validation")
            or condition.get("quantization") != "gleipnir_frost_attention_fp4"
            or condition.get("fp4_preparation") != "combined"
            or not condition.get("gemm_tuning_validation")
            or condition.get("swiglu_fusion_validation")
        )
    ):
        raise ValueError("FP4 attention projections require a validated audited worker")
    overhead_worker = condition["worker_cls"].endswith(
        "OverheadAttentionTunedPreparationMxfp8ServingAuditWorker"
    )
    if bool(condition.get("swiglu_overhead_validation")) != overhead_worker or (
        overhead_worker and not condition.get("attention_projection_validation")
    ):
        raise ValueError(
            "symbolic SwiGLU requires its receipt and attention-FP4 worker"
        )
    overrides = condition["serving_config_overrides"]
    native_output_worker = condition["worker_cls"].endswith(
        "NativeOutputAttentionTunedPreparationMxfp8ServingAuditWorker"
    )
    if bool(
        condition.get("swiglu_native_output_validation")
    ) != native_output_worker or (
        native_output_worker
        and (
            not condition.get("attention_projection_validation")
            or not condition.get("swiglu_overhead_reference")
        )
    ):
        raise ValueError(
            "native FP4 output requires its receipt and attention-FP4 worker"
        )
    direct_gdn_worker = condition["worker_cls"].endswith(
        "DirectGdnNativeOutputAttentionTunedPreparationMxfp8ServingAuditWorker"
    )
    if bool(condition.get("gdn_direct_output_validation")) != direct_gdn_worker or (
        direct_gdn_worker and condition["gdn_backend"] != "flashinfer"
    ):
        raise ValueError("direct GDN output requires its receipt and FlashInfer worker")
    if set(overrides) - {
        "max_num_seqs",
        "gpu_memory_utilization",
        "max_num_batched_tokens",
    }:
        raise ValueError("override would change the frozen data/scoring contract")
    graph_worker = ".prefill_graph_worker." in condition["worker_cls"]
    if (
        bool(condition.get("prefill_graphs")) != graph_worker
        or bool(condition.get("compilation_config")) != graph_worker
    ):
        raise ValueError("prefill graphs require their bounded audited worker")
    if graph_worker:
        from gleipnir.serving_prefill_graphs import validate_graph_config

        validate_graph_config(condition)
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
    if attention_backend == "FLASH_ATTN":
        index = args.index("--attention-backend")
        attention_config = {"backend": "FLASH_ATTN", "flash_attn_version": 4}
        if condition.get("external_fa4"):
            attention_config["flash_attn_max_num_splits_for_cuda_graph"] = 1
        args[index : index + 2] = [
            "--attention-config",
            json.dumps(attention_config),
        ]
        if condition.get("external_fa4"):
            args.extend(["--block-size", "128"])
    if condition["attention_precision"] == "fp8_e4m3":
        args.extend(["--kv-cache-dtype", "fp8_e4m3", "--calculate-kv-scales"])
    elif condition["attention_precision"] == "nvfp4":
        args.extend(["--kv-cache-dtype", "nvfp4"])
    if condition.get("profiler_config"):
        args.extend(["--profiler-config", json.dumps(condition["profiler_config"])])
    if graph_worker:
        args.extend(
            ["--compilation-config", json.dumps(condition["compilation_config"])]
        )
    return {
        **condition,
        "extra_server_args": args,
        "startup_audit": "results/b200_attention_gdn_serving/native_attention.json",
    }


def resolve_high_reference(condition: dict) -> str | None:
    """Bind selected full-cohort controls without rewriting historical trials."""
    reference = condition.get("high_reference")
    if reference != "selected":
        return reference
    resolved = resolve_kernel_baseline({"baseline": "selected"})
    selection = json.loads((BENCHMARK / "baseline.json").read_text())
    path = ROOT / resolved["baseline"] / "high_summary.json"
    if sha(path) != selection["high_summary_sha256"]:
        raise ValueError("selected high-concurrency baseline identity drift")
    return resolved["baseline"]


async def high_concurrency(
    config: dict, manifest: dict, condition: dict, out: Path
) -> None:
    rows = json.loads((ROOT / "data/b200_inference_benchmark/full.json").read_text())
    assert len(rows) == 320
    reference = resolve_high_reference(condition)
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
            if reference:
                ref = ROOT / reference
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
        "--condition", type=Path, default=EXPERIMENT / "fp4_swiglu_native_output.json"
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
    if raw.get("prefill_graphs"):
        sources.extend(
            [
                "src/gleipnir/serving_prefill_graphs.py",
                "experiments/b200_attention_gdn_serving/prefill_graph_worker.py",
                "experiments/b200_attention_gdn_serving/prefill_graph_canary.py",
            ]
        )
    if raw.get("gdn_direct_output_validation"):
        sources.extend(
            [
                "src/gleipnir/serving_gdn_direct_output.py",
                "src/gleipnir/serving_gdn_direct_caller.py",
                "experiments/b200_attention_gdn_serving/gdn_direct_worker.py",
                "experiments/b200_attention_gdn_serving/gdn_direct_canary.py",
            ]
        )
    if raw.get("swiglu_native_output_validation"):
        sources.extend(
            [
                "src/gleipnir/serving_fp4_swiglu.py",
                "src/gleipnir/serving_fp4_swiglu_pack.py",
                "src/gleipnir/serving_fp4_swiglu_overhead.py",
                "src/gleipnir/serving_fp4_swiglu_overhead_validation.py",
                "src/gleipnir/serving_fp4_swiglu_native_output.py",
                "src/gleipnir/serving_fp4_swiglu_block_reference.py",
                "src/gleipnir/serving_fp4_swiglu_native_output_validation.py",
                "src/gleipnir/serving_fp4_swiglu_native_output_integration.py",
                "experiments/b200_attention_gdn_serving/swiglu_native_output_worker.py",
                "experiments/b200_attention_gdn_serving/fp4_swiglu_native_output_compare.py",
            ]
        )
    if raw.get("swiglu_overhead_validation"):
        sources.extend(
            [
                "src/gleipnir/serving_fp4_swiglu.py",
                "src/gleipnir/serving_fp4_swiglu_pack.py",
                "src/gleipnir/serving_fp4_swiglu_overhead.py",
                "src/gleipnir/serving_fp4_swiglu_padding.py",
                "src/gleipnir/serving_fp4_swiglu_overhead_integration.py",
                "src/gleipnir/serving_fp4_swiglu_overhead_validation.py",
                "experiments/b200_attention_gdn_serving/swiglu_overhead_worker.py",
                "experiments/b200_attention_gdn_serving/fp4_swiglu_overhead_compare.py",
            ]
        )
    if raw.get("attention_projection_precision") == "fp4":
        sources.extend(
            [
                "src/gleipnir/serving_attention_fp4.py",
                "src/gleipnir/vllm_frost_attention_fp4.py",
                "experiments/b200_attention_gdn_serving/attention_fp4_worker.py",
                "experiments/b200_attention_gdn_serving/attention_fp4_canary.py",
            ]
        )
    if raw.get("gemm_tuning_validation") or raw.get("gemm_reference_validation"):
        sources.extend(
            [
                "src/gleipnir/serving_fp4_tuning.py",
                "src/gleipnir/serving_fp4_tuning_validation.py",
                "experiments/b200_attention_gdn_serving/fp4_gemm_tune.py",
                "experiments/b200_attention_gdn_serving/tuned_worker.py",
            ]
        )
    if raw.get("swiglu_fusion_validation"):
        sources.extend(
            [
                "src/gleipnir/serving_fp4_swiglu.py",
                "src/gleipnir/serving_fp4_swiglu_pack.py",
                "src/gleipnir/serving_fp4_swiglu_integration.py",
                "src/gleipnir/serving_fp4_swiglu_validation.py",
                "experiments/b200_attention_gdn_serving/swiglu_worker.py",
                "experiments/b200_attention_gdn_serving/fp4_swiglu_compare.py",
            ]
        )
    if raw["attention_precision"] == "mxfp8":
        sources.extend(
            [
                "src/gleipnir/serving_mxfp8.py",
                "src/gleipnir/serving_mxfp8_source.py",
                "src/gleipnir/nvidia_mxfp8_attention.py",
                "src/gleipnir/nvidia_mxfp8_fused_quantize.py",
                "experiments/b200_attention_gdn_serving/mxfp8_worker.py",
                "experiments/b200_attention_gdn_serving/mxfp8_canary.py",
            ]
        )
    if raw.get("fp4_preparation"):
        sources.extend(
            [
                "src/gleipnir/serving_fp4_prepare.py",
                "src/gleipnir/serving_fp4_fusion.py",
                "src/gleipnir/serving_fp4_integration.py",
                "experiments/b200_attention_gdn_serving/prepare_worker.py",
                "experiments/b200_attention_gdn_serving/fp4_prepare_canary.py",
                "experiments/b200_attention_gdn_serving/fp4_fusion_canary.py",
            ]
        )
    if ".combined_worker." in raw["worker_cls"] or raw.get("external_fa4"):
        sources.append("experiments/b200_attention_gdn_serving/combined_worker.py")
    if raw.get("external_fa4"):
        sources.extend(
            [
                "src/gleipnir/serving_fa4.py",
                "experiments/b200_attention_gdn_serving/external_fa4_worker.py",
            ]
        )
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
        for name in (
            "loaded_precision.json",
            "native_attention.json",
            "native_preparation.json",
            "native_gemm_tuning.json",
            "native_swiglu.json",
            "native_swiglu_overhead.json",
            "native_swiglu_output.json",
            "native_attention_projections.json",
            "native_gdn_direct_output.json",
            "prefill_graphs.json",
        ):
            if (OUTPUT / name).exists():
                write(out / name, json.loads((OUTPUT / name).read_text()))


if __name__ == "__main__":
    main()
