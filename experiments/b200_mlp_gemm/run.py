"""Run isolated MLP variants with the retained pinned runtime and shared caches."""

import argparse
import json
import os
import signal
import subprocess
import sys
from pathlib import Path

import yaml

from experiments.b200_nvidia_mxfp8.run import environment
from gleipnir.monitoring_systems_screen import sha256_file


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--attempt", required=True)
    parser.add_argument(
        "--variants",
        nargs="+",
        default=["compiled", "merged", "merged_compiled", "cudnn"],
    )
    args = parser.parse_args()
    if not args.attempt.isalnum():
        raise ValueError("attempt must be alphanumeric")
    cfg = yaml.safe_load(Path("experiments/b200_mlp_gemm/config.yaml").read_text())
    root = Path("results/b200_mlp_gemm") / args.attempt
    logs = Path("logs/runpod/b200_mlp_gemm") / args.attempt
    root.mkdir(parents=True, exist_ok=False)
    logs.mkdir(parents=True, exist_ok=False)
    env = {**os.environ, **environment(cfg)}
    # The container's temporary environment is lost on termination; bind the
    # retained model cache explicitly rather than relying on a sourced shell.
    cache_root = Path.cwd() / ".cache/huggingface"
    env.update(
        HF_HOME=str(cache_root),
        HF_HUB_CACHE=str(cache_root / "hub"),
        HF_DATASETS_CACHE=str(cache_root / "datasets"),
        PYTHONUNBUFFERED="1",
        TOKENIZERS_PARALLELISM="false",
        TORCHINDUCTOR_COMPILE_THREADS="16",
        MAX_JOBS="16",
    )
    sources = [
        *Path("experiments/b200_mlp_gemm").glob("*.py"),
        Path("experiments/b200_mlp_gemm/config.yaml"),
        Path("src/gleipnir/mlp_gemm.py"),
        Path("src/gleipnir/cudnn_lora_mlp.py"),
        Path("src/gleipnir/cudnn_fp4_gemm.py"),
        Path("src/gleipnir/cudnn_fp4_mlp.py"),
        Path("src/gleipnir/nvfp4_pack.py"),
        Path("experiments/b200_mlp_gemm/README.md"),
    ]
    report = {
        "status": "running",
        "candidates": [],
        "cache_paths": {k: v for k, v in env.items() if "CACHE" in k},
        "source_sha256": {str(p): sha256_file(p) for p in sources},
    }
    for p in sources:
        dest = root / "executed_sources" / p
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(p.read_bytes())
    for variant in args.variants:
        command = [
            sys.executable,
            "-m",
            "experiments.b200_mlp_gemm.probe",
            "--variant",
            variant,
            "--output",
            str(root / variant),
        ]
        if variant in {"fp4", "fp4optimized", "fp4row", "fp4chunks"}:
            command[2] = "experiments.b200_mlp_gemm.fp4_probe"
            command = [command[0], "-m", command[2], "--output", str(root / variant)]
        if variant in {"fp4optimized", "fp4row", "fp4chunks"}:
            command.append("--optimized")
        if variant in {"fp4row", "fp4chunks"}:
            command.append("--row-scaled")
        if variant == "fp4chunks":
            command.append("--chunked-rows")
        if variant in {"integrated", "integratedprofile"}:
            command = [
                sys.executable,
                "-m",
                "experiments.b200_mlp_gemm.integrated_profile"
                if variant == "integratedprofile"
                else "experiments.b200_mlp_gemm.integrated_probe",
                "--output",
                str(root / variant),
            ]
        row = {"variant": variant, "status": "starting"}
        report["candidates"].append(row)
        (root / "launch.json").write_text(json.dumps(report, indent=2) + "\n")
        print(f"starting {variant}", flush=True)
        with (logs / (variant + ".log")).open("x") as handle:
            proc = subprocess.Popen(
                command,
                env=env,
                stdout=handle,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            try:
                code = proc.wait(
                    timeout=1800
                    if variant in {"integrated", "integratedprofile"}
                    else 1200
                    if variant in {"fp4", "fp4optimized", "fp4row", "fp4chunks"}
                    else cfg["timeout_seconds"]
                )
                row.update(
                    status="complete" if code == 0 else "failed", returncode=code
                )
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
                row.update(status="timeout", returncode=proc.returncode)
        (root / "launch.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(row), flush=True)
    report["status"] = "complete"
    (root / "launch.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
