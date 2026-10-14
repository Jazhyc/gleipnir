"""Record node hardware, driver and runtime identity, which can drift between Pods."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import time
import xml.etree.ElementTree as ElementTree
from pathlib import Path

PACKAGES = (
    "vllm",
    "torch",
    "triton",
    "transformers",
    "flashinfer-python",
    "nvidia-cudnn-cu13",
    "nvidia-cudnn-frontend",
    "nvidia-cutlass-dsl",
    "apache-tvm-ffi",
)
# Runpod's own identifiers only; never credentials or arbitrary environment values.
RUNPOD_KEYS = ("RUNPOD_POD_ID", "RUNPOD_DC_ID", "RUNPOD_GPU_COUNT", "RUNPOD_CPU_COUNT")
GPU_FIELDS = {
    "product_name": "product_name",
    "uuid": "uuid",
    "serial": "serial",
    "vbios_version": "vbios_version",
    "pci_bus_id": "pci/pci_bus_id",
    "pcie_generation_max": "pci/pci_gpu_link_info/pcie_gen/max_link_gen",
    "pcie_generation_current": "pci/pci_gpu_link_info/pcie_gen/current_link_gen",
    "pcie_width_max": "pci/pci_gpu_link_info/link_widths/max_link_width",
    "pcie_width_current": "pci/pci_gpu_link_info/link_widths/current_link_width",
    "persistence_mode": "persistence_mode",
    "compute_mode": "compute_mode",
    "mig_mode": "mig_mode/current_mig",
    "ecc_mode": "ecc_mode/current_ecc",
    "memory_total": "fb_memory_usage/total",
    "memory_used": "fb_memory_usage/used",
    "power_limit": "gpu_power_readings/current_power_limit",
    "power_limit_max": "gpu_power_readings/max_power_limit",
    "clock_sm_max": "max_clocks/sm_clock",
    "clock_memory_max": "max_clocks/mem_clock",
    "clock_sm_application": "applications_clocks/graphics_clock",
    "clock_sm_current": "clocks/sm_clock",
    "temperature": "temperature/gpu_temp",
    "performance_state": "performance_state",
}
SAMPLE_FIELDS = (
    "timestamp,clocks.sm,clocks.mem,power.draw,temperature.gpu,"
    "utilization.gpu,memory.used,clocks_event_reasons.active"
)


def sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def run(argv: list[str]) -> str:
    return subprocess.run(argv, check=True, capture_output=True, text=True).stdout


def parse_gpus(xml: str) -> list[dict]:
    """Extract stable identity and configuration fields from `nvidia-smi -q -x`."""
    root = ElementTree.fromstring(xml)
    gpus = []
    for gpu in root.findall("gpu"):
        values = {}
        for name, path in GPU_FIELDS.items():
            node = gpu.find(path)
            values[name] = None if node is None else (node.text or "").strip()
        gpus.append(values)
    return [
        {
            "driver_version": (root.findtext("driver_version") or "").strip(),
            "cuda_version": (root.findtext("cuda_version") or "").strip(),
            **gpu,
        }
        for gpu in gpus
    ]


def cpu_summary(lscpu: str) -> dict:
    wanted = {
        "Model name",
        "CPU(s)",
        "Thread(s) per core",
        "Socket(s)",
        "NUMA node(s)",
        "L3",
    }
    fields = {}
    for line in lscpu.splitlines():
        key, _, value = line.partition(":")
        if key.strip() in wanted:
            fields[key.strip()] = value.strip()
    return fields


def cgroup_cpus() -> float | None:
    path = Path("/sys/fs/cgroup/cpu.max")
    if not path.exists():
        return None
    quota, period = path.read_text().split()
    return None if quota == "max" else int(quota) / int(period)


def package_versions(packages: tuple[str, ...] = PACKAGES) -> dict:
    versions = {}
    for name in packages:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def distribution_digest() -> str:
    """Hash every installed distribution name/version in the running interpreter."""
    rows = sorted(
        f"{d.metadata['Name']}=={d.version}" for d in importlib.metadata.distributions()
    )
    return hashlib.sha256("\n".join(rows).encode()).hexdigest()


def snapshot(root: Path, *, raw_directory: Path | None = None) -> dict:
    """Capture without creating a CUDA context, so it is safe beside a live server."""
    xml = run(["nvidia-smi", "-q", "-x"])
    lscpu = run(["lscpu"])
    if raw_directory is not None:
        raw_directory.mkdir(parents=True, exist_ok=True)
        (raw_directory / "nvidia_smi.xml").write_text(xml)
        (raw_directory / "lscpu.txt").write_text(lscpu)
    meminfo = Path("/proc/meminfo").read_text().splitlines()[0].split()
    os_release = dict(
        line.split("=", 1)
        for line in Path("/etc/os-release").read_text().splitlines()
        if "=" in line
    )
    return {
        "captured_at_unix": time.time(),
        "hostname": platform.node(),
        "kernel": platform.release(),
        "os": os_release.get("PRETTY_NAME", "").strip('"'),
        "runpod": {k: os.environ.get(k) for k in RUNPOD_KEYS},
        "gpus": parse_gpus(xml),
        "cpu": cpu_summary(lscpu),
        "cgroup_cpus": cgroup_cpus(),
        "affinity_cpus": len(os.sched_getaffinity(0)),
        "memory_kib": int(meminfo[1]),
        "python": platform.python_version(),
        "packages": package_versions(),
        "distributions_sha256": distribution_digest(),
        "lock_sha256": sha(root / "uv.lock"),
        "nvcc": run(["/usr/local/cuda/bin/nvcc", "--version"]).strip().splitlines()[-1]
        if Path("/usr/local/cuda/bin/nvcc").exists()
        else None,
    }


def drift(reference: dict, current: dict) -> dict:
    """Name every identity field that differs between two node snapshots."""
    fields = ("driver_version", "cuda_version", "product_name", "vbios_version")
    changes = {}
    for key in fields:
        before = [g.get(key) for g in reference.get("gpus", [])]
        after = [g.get(key) for g in current.get("gpus", [])]
        if before != after:
            changes[key] = {"reference": before, "current": after}
    for key in ("kernel", "os", "python", "packages", "lock_sha256"):
        if reference.get(key) != current.get(key):
            changes[key] = {
                "reference": reference.get(key),
                "current": current.get(key),
            }
    return changes


class GpuSampler:
    """Sample clocks, power and throttle reasons while one length is measured."""

    def __init__(self, path: Path, interval_ms: int = 250) -> None:
        self.path = path
        self.interval_ms = interval_ms
        self.process: subprocess.Popen | None = None

    def __enter__(self) -> GpuSampler:
        self.handle = self.path.open("x")
        self.process = subprocess.Popen(
            [
                "nvidia-smi",
                f"--query-gpu={SAMPLE_FIELDS}",
                "--format=csv,nounits",
                f"-lms={self.interval_ms}",
            ],
            stdout=self.handle,
            stderr=subprocess.STDOUT,
        )
        return self

    def __exit__(self, *_: object) -> None:
        assert self.process is not None
        self.process.terminate()
        self.process.wait(timeout=10)
        self.handle.close()


def summarize_samples(text: str) -> dict:
    """Median/minimum SM clock, peak power/temperature and any active throttling."""
    rows = [line.split(", ") for line in text.strip().splitlines()[1:]]
    rows = [r for r in rows if len(r) == 8 and r[1].strip().isdigit()]
    if not rows:
        return {"samples": 0}
    clocks = sorted(int(r[1]) for r in rows)
    reasons = sorted({r[7].strip() for r in rows})
    return {
        "samples": len(rows),
        "sm_clock_median_mhz": clocks[len(clocks) // 2],
        "sm_clock_min_mhz": clocks[0],
        "power_max_w": max(float(r[3]) for r in rows),
        "temperature_max_c": max(int(r[4]) for r in rows),
        "event_reasons": reasons,
    }


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    receipt = snapshot(root, raw_directory=args.output.with_suffix(""))
    args.output.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt["gpus"], indent=2))


if __name__ == "__main__":
    main()
