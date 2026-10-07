"""Launch the isolated native check with existing persistent kernel caches."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

from gleipnir.monitoring_systems_screen import sha256_file
from gleipnir.packed_benchmark import benchmark_environment

ROOT = Path(__file__).resolve().parents[2]


def environment(config: dict) -> dict[str, str]:
    """Retain pinned FlashQLA dependencies and append the isolated NVIDIA overlay."""
    overlay = ROOT / config["overlay"]
    result = benchmark_environment(
        {
            **config,
            "kernel_overlays": [config["fa4_overlay"], str(overlay / "frontend")],
        },
        ROOT,
    )
    result["LD_LIBRARY_PATH"] = (
        str(overlay / "runtime/nvidia/cudnn/lib")
        + ":"
        + result.get("LD_LIBRARY_PATH", "")
    )
    result["CUDNN_FRONTEND_ENABLE_FROST_ENGINES"] = "1"
    result["CUDNN_FRONTEND_COMPILED_CACHE"] = str(
        ROOT / ".cache/training/shared/cudnn_frontend"
    )
    result["CUDNN_FRONTEND_COMPILED_CACHE_MAX_BYTES"] = "0"
    result["CUTE_DSL_CACHE_DIR"] = str(ROOT / ".cache/training/shared/cute_dsl")
    result["GLEIPNIR_NVIDIA_SOURCE"] = str(overlay / "source")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "experiments/b200_nvidia_mxfp8/config.yaml",
    )
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    output = ROOT / config["output"]
    logs = ROOT / config["logs"]
    output.mkdir(parents=True, exist_ok=True)
    logs.mkdir(parents=True, exist_ok=True)
    if (output / "kernel_canary.json").exists():
        raise ValueError(
            "preserve existing native receipts; use a separate attempt directory"
        )
    env = environment(config)
    report = {
        "status": "native_canary_starting",
        "config": config,
        "cache_paths": {k: v for k, v in env.items() if "CACHE" in k},
        "source_sha256": {
            str(p): sha256_file(p)
            for p in [
                ROOT / "src/gleipnir/__init__.py",
                ROOT / "src/gleipnir/_compat.py",
                Path(__file__),
                args.config,
                ROOT / "experiments/b200_nvidia_mxfp8/kernel_canary.py",
                ROOT / "src/gleipnir/kernels/mxfp8/nvidia_mxfp8_attention.py",
                ROOT
                / config.get(
                    "source_archive",
                    "results/b200_nvidia_mxfp8/setup/cudnn-source.tar.gz",
                ),
            ]
        },
    }
    archive = output / "executed_source"
    for path in report["source_sha256"]:
        source = Path(path)
        if source.suffix != ".gz":
            destination = archive / source.resolve().relative_to(ROOT)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
    (output / "launch.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)
    with (logs / "native-canary.log").open("x") as handle:
        process = subprocess.run(
            [
                sys.executable,
                "-m",
                "experiments.b200_nvidia_mxfp8.kernel_canary",
                "--config",
                str(args.config),
            ],
            cwd=ROOT,
            env=env,
            stdout=handle,
            stderr=subprocess.STDOUT,
        )
    report.update(
        status="native_canary_complete"
        if process.returncode == 0
        else "native_canary_failed",
        returncode=process.returncode,
    )
    (output / "launch.json").write_text(json.dumps(report, indent=2) + "\n")
    if process.returncode:
        raise SystemExit(process.returncode)


if __name__ == "__main__":
    main()
