"""Screen native inference GEMMs against the frozen merged HTTP control."""

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
OUTPUT = ROOT / "results/b200_inference_kernels"


def condition_args(condition: dict) -> list[str]:
    """Use the stock online quantizer with an explicit MLP-only ignore rule."""
    if condition["name"] != "fp8_mlp" or condition["quantization"] != "fp8_per_channel":
        raise ValueError("unsupported kernel condition")
    if condition["quantization_config"] != {
        "ignore": [r"re:^(?!.*\.layers\.\d+\.mlp\.).*$"]
    }:
        raise ValueError("MLP precision scope drift")
    if condition["port"] == 8000:
        raise ValueError("preserve the merged control endpoint")
    return [
        "--quantization",
        condition["quantization"],
        "--quantization-config",
        json.dumps(condition["quantization_config"]),
        "--worker-cls",
        condition["worker_cls"],
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="fp8_mlp01")
    parser.add_argument("--reuse-server", action="store_true")
    args = parser.parse_args()
    if Path(args.output).name != args.output:
        raise ValueError("output must be one directory name")
    config = yaml.safe_load((BENCHMARK / "config.yaml").read_text())
    manifest = prepared_manifest(config)
    if sha(ROOT / config["parity"]) != config["parity_sha256"]:
        raise ValueError("archived master score reference drift")
    condition = json.loads((EXPERIMENT / "config.json").read_text())
    condition["extra_server_args"] = condition_args(condition)
    condition["config_sha256"] = sha(EXPERIMENT / "config.json")
    config["port"] = condition["port"]
    output = OUTPUT / args.output
    output.mkdir(parents=True, exist_ok=False)
    for directory in (EXPERIMENT, BENCHMARK):
        for source in directory.iterdir():
            if source.is_file() and source.suffix in {".py", ".md", ".json", ".yaml"}:
                target = output / "executed_sources" / source.relative_to(ROOT)
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(source.read_bytes())
    for source in (
        "src/gleipnir/inference_benchmark.py",
        "src/gleipnir/merged_lora.py",
    ):
        target = output / "executed_sources" / source
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT / source).read_bytes())
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
