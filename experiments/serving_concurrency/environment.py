"""Record the host and software of the measurement Pod."""

from __future__ import annotations

import importlib.metadata
import json
import os
import platform
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PACKAGES = (
    "vllm",
    "torch",
    "flashinfer-python",
    "transformers",
    "nvidia-cudnn-cu13",
    "nvidia-cudnn-frontend",
    "nvidia-cutlass-dsl",
    "triton",
)


def read(path: str) -> str | None:
    try:
        return Path(path).read_text().strip()
    except OSError:
        return None


def version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def collect() -> dict:
    gpu = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,driver_version,uuid",
            "--format=csv,noheader",
        ],
        text=True,
    ).strip()
    cpu = [
        line.split(":", 1)[1].strip()
        for line in Path("/proc/cpuinfo").read_text().splitlines()
        if line.startswith("model name")
    ]
    return {
        "datacenter": os.environ.get("RUNPOD_DC_ID"),
        "pod_id": os.environ.get("RUNPOD_POD_ID"),
        "gpu": gpu,
        "cpu_model": cpu[0] if cpu else None,
        "host_cpu_threads": len(cpu),
        "affinity_cpus": len(os.sched_getaffinity(0)),
        "cgroup_cpu_max": read("/sys/fs/cgroup/cpu.max"),
        "cgroup_memory_max": read("/sys/fs/cgroup/memory.max"),
        "kernel": platform.release(),
        "os": read("/etc/os-release"),
        "python": platform.python_version(),
        "packages": {name: version(name) for name in PACKAGES},
    }


def main() -> None:
    out = ROOT / "results/serving_concurrency/environment.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(collect(), indent=1) + "\n")
    print(out.read_text())


if __name__ == "__main__":
    main()
