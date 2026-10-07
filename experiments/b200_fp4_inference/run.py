"""Matched HTTP inference benchmark for native cuDNN FP4 MLPs."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

import yaml

from experiments.b200_inference_benchmark.run import (
    EXPERIMENT as BENCHMARK,
)
from experiments.b200_inference_benchmark.run import (
    ROOT,
    benchmark,
    prepared_manifest,
    sha,
    verify_merged_model,
    write,
)

EXPERIMENT = Path(__file__).parent
OUTPUT = ROOT / "results/b200_fp4_inference"


def resolve_condition(condition: dict, source_hashes: dict) -> dict:
    if (
        condition["quantization"] != "gleipnir_b200_nvfp4"
        or condition["linear_backend"] != "flashinfer_cudnn"
    ):
        raise ValueError("requires native cuDNN online FP4")
    return {
        **condition,
        "extra_server_args": [
            "--quantization",
            condition["quantization"],
            "--linear-backend",
            condition["linear_backend"],
            "--worker-cls",
            condition["worker_cls"],
            "--additional-config",
            json.dumps({"gleipnir_b200_nvfp4": source_hashes}, sort_keys=True),
        ],
        "log_directory": "logs/runpod/b200_fp4_inference",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="cudnn01")
    parser.add_argument("--reuse-server", action="store_true")
    args = parser.parse_args()
    if Path(args.output).name != args.output:
        raise ValueError("output must be a directory name")
    config = yaml.safe_load((BENCHMARK / "config.yaml").read_text())
    manifest = prepared_manifest(config)
    if sha(ROOT / config["parity"]) != config["parity_sha256"]:
        raise ValueError("master reference drift")
    sources = [
        "src/gleipnir/__init__.py",
        "src/gleipnir/_compat.py",
        "src/gleipnir/serving/vllm/online_nvfp4.py",
        "src/gleipnir/serving/vllm/nvfp4.py",
        "src/gleipnir/kernels/fp4/nvfp4_reference.py",
        "experiments/b200_fp4_inference/worker.py",
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
        ROOT / "src/gleipnir/__init__.py",
        ROOT / "src/gleipnir/_compat.py",
        *EXPERIMENT.glob("*"),
        *BENCHMARK.glob("*"),
        *(ROOT / p for p in sources),
        ROOT / "src/gleipnir/serving/benchmark.py",
        ROOT / "src/gleipnir/adapters/merge.py",
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
