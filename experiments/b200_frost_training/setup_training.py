"""Install only missing pinned training overlays in the existing CUDA 13 runtime."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tarfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def ensure_flashqla_overlay(root: Path, environment: dict[str, str]) -> None:
    """Stage the independent FlashQLA dependency job on local disk."""
    target = root / ".cache/kernels/flashqla-da06429"
    local_target = Path("/tmp/gleipnir-flashqla-da06429")
    archive = root / ".cache/kernels/sources/flashqla-runtime-da06429.tar.gz"
    if not target.exists():
        local_target.mkdir(exist_ok=True)
        if archive.exists():
            with tarfile.open(archive) as source:
                source.extractall(local_target, filter="data")
        if target.is_symlink():
            target.unlink()
        target.symlink_to(local_target, target_is_directory=True)
    manifest = target / "install_manifest.json"
    if not manifest.exists() or not json.loads(manifest.read_text()).get(
        "package_sha256"
    ):
        script = (root / "experiments/fp4_stability/bootstrap_flashqla.sh").read_text()
        script = script.replace("source .cache-runtime.env", "export MAX_JOBS=16")
        script = script.replace(".venv/bin/python", sys.executable)
        subprocess.run(["bash", "-c", script], env=environment, check=True)
    subprocess.run(
        [sys.executable, "-m", "experiments.fp4_stability.patch_tilelang_fp16"],
        env=environment,
        check=True,
    )
    subprocess.run(
        [
            "uv",
            "pip",
            "install",
            "--python",
            sys.executable,
            "--target",
            str(target.resolve()),
            "--no-deps",
            "quack-kernels==0.6.5",
        ],
        env=environment,
        check=True,
    )
    if target.resolve() == local_target:
        temporary = archive.with_suffix(".tmp")
        with tarfile.open(temporary, "w:gz", dereference=True) as output:
            for path in sorted(local_target.iterdir()):
                output.add(path, arcname=path.name)
        temporary.replace(archive)


def prepare_overlays(root: Path, environment: dict[str, str]) -> None:
    """Run three independent install/build jobs; propagate every failure."""
    from gleipnir.training.backends.qwen35 import (
        ensure_causal_conv1d,
        ensure_fla_kernels,
    )

    with ThreadPoolExecutor(max_workers=3) as pool:
        jobs = [
            pool.submit(
                ensure_fla_kernels, python=Path(sys.executable), base=environment
            ),
            pool.submit(
                ensure_causal_conv1d, python=Path(sys.executable), base=environment
            ),
            pool.submit(ensure_flashqla_overlay, root, environment),
        ]
        for job in jobs:
            job.result()


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    os.chdir(root)
    target = Path("/tmp/gleipnir-triton-3.7.1")
    if not target.exists():
        target.symlink_to(root / ".cache/kernels/triton", target_is_directory=True)
    environment = dict(os.environ)
    environment.update(
        MAX_JOBS="16", TORCH_CUDA_ARCH_LIST="10.0", CUDA_HOME="/usr/local/cuda"
    )
    prepare_overlays(root, environment)
    print("training_overlays_ready", flush=True)


if __name__ == "__main__":
    main()
