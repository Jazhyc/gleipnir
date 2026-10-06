"""Benchmark validated training-forward FROST arithmetic inside vLLM."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

import yaml

from experiments.b200_inference_benchmark.run import EXPERIMENT as BENCHMARK
from experiments.b200_inference_benchmark.run import (
    ROOT,
    benchmark,
    prepared_manifest,
    sha,
    verify_merged_model,
    write,
)

EXPERIMENT = Path(__file__).parent
OUTPUT = ROOT / "results/b200_frost_inference"


def resolve_condition(condition: dict, source_hashes: dict) -> dict:
    if (
        condition["quantization"] != "gleipnir_frost_fp4"
        or not condition["training_fp4_environment"]
    ):
        raise ValueError("requires training-forward FROST runtime")
    return {
        **condition,
        "extra_server_args": [
            "--quantization",
            condition["quantization"],
            "--worker-cls",
            condition["worker_cls"],
            "--attention-backend",
            "FLASHINFER",
            "--additional-config",
            json.dumps({"gleipnir_frost_fp4": source_hashes}, sort_keys=True),
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="frost01")
    parser.add_argument("--reuse-server", action="store_true")
    args = parser.parse_args()
    if Path(args.output).name != args.output:
        raise ValueError("output must be a directory name")
    config = yaml.safe_load((BENCHMARK / "config.yaml").read_text())
    manifest = prepared_manifest(config)
    if sha(ROOT / config["parity"]) != config["parity_sha256"]:
        raise ValueError("master reference drift")
    sources = [
        "src/gleipnir/vllm_frost_fp4.py",
        "src/gleipnir/cudnn_fp4_gemm.py",
        "src/gleipnir/cudnn_fp4_epilogue.py",
        "src/gleipnir/cudnn_fp4_mlp.py",
        "src/gleipnir/native_fp4_training.py",
        "src/gleipnir/nvfp4_pack.py",
        "experiments/b200_frost_inference/worker.py",
    ]
    condition = resolve_condition(
        json.loads((EXPERIMENT / "config.json").read_text()),
        {p: sha(ROOT / p) for p in sources},
    )
    condition["config_sha256"] = sha(EXPERIMENT / "config.json")
    config["port"] = condition["port"]
    output = OUTPUT / args.output
    output.mkdir(parents=True, exist_ok=False)
    for source in [
        *EXPERIMENT.glob("*"),
        *BENCHMARK.glob("*"),
        *(ROOT / p for p in sources),
        ROOT / "src/gleipnir/inference_benchmark.py",
        ROOT / "src/gleipnir/merged_lora.py",
    ]:
        if source.is_file():
            target = output / "executed_sources" / source.relative_to(ROOT)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(source.read_bytes())
    write(output / "manifest.json", manifest)
    write(output / "condition.json", condition)
    try:
        merged = Path(condition["merged_model"])
        write(output / "merged_artifact.json", verify_merged_model(config, merged))
        asyncio.run(
            benchmark(
                config, manifest, output, 64, args.reuse_server, merged, condition
            )
        )
    except BaseException as error:
        write(output / "failure.json", {"error": f"{type(error).__name__}: {error}"})
        raise
    finally:
        audit = OUTPUT / "loaded_precision.json"
        if audit.exists():
            write(output / "loaded_precision.json", json.loads(audit.read_text()))


if __name__ == "__main__":
    main()
