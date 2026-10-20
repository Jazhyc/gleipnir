"""Turn the open() traces of both release launches into the image layer spec.

Layers (absolute paths on the build Pod, kept at their original locations):

- ``runtime``: the vLLM 0.31 serving interpreter, the native FA4/MXFP8/Triton
  trees and the Gigatoken build. The legacy ``serving-runtime/venv`` is left
  out unless a launch opened it.
- ``caches``: compiler-cache files the launches read, plus the small native
  caches whole (``dlopen``/cuDNN loads are invisible to the trace).
- ``repository``: every repository file except local state and artifact
  trees, plus exactly the traced ``results/`` receipts. Any traced ``data/``
  file aborts the build, because the image must not carry dataset rows.
- ``release``: ``/opt/gleipnir-release`` (entrypoint, 27B source copies).
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

REPO = Path("/workspace/gleipnir")
RUNTIME = [
    "/tmp/gleipnir-vllm031-runtime",
    "/tmp/gleipnir-serving-runtime/fa4",
    "/tmp/gleipnir-serving-runtime/mxfp8",
    "/tmp/gleipnir-serving-runtime/triton",
    "/tmp/gleipnir-serving-runtime/manifest.json",
    "/tmp/gleipnir-gigatoken-0.10.0",
]
WHOLE_CACHES = [
    "/workspace/gleipnir/.cache/flashinfer",
    "/workspace/gleipnir/.cache/training/shared/cudnn_frontend",
    "/workspace/gleipnir/.cache/training/shared/cute_dsl",
    "/tmp/torchinductor_root",
    "/root/.cache/flashinfer",
    "/root/.cache/vllm",
    "/root/.cache/cudnn_gemm",
    "/root/.cache/torch",
    "/root/.nv",
]
# Repository-relative top-level entries that are local state or artifacts.
REPO_EXCLUDE = {
    ".git",
    ".cache",
    ".venv",
    ".venv-vllm031",
    "results",
    "data",
    "logs",
    "outputs",
    "wandb",
    ".runpod",
    ".env",
    ".pytest_cache",
    ".ruff_cache",
}


def traced(directory: Path) -> set[str]:
    paths: set[str] = set()
    for trace in directory.glob("*/*.txt"):
        paths.update(line for line in trace.read_text().splitlines() if line)
    return {p for p in paths if os.path.isfile(p)}


def repository_files() -> list[str]:
    files = []
    for entry in sorted(REPO.iterdir()):
        if entry.name in REPO_EXCLUDE or entry.name.startswith(".env"):
            continue
        if entry.is_file():
            files.append(str(entry))
            continue
        for root, dirs, names in os.walk(entry):
            dirs[:] = [d for d in dirs if d != "__pycache__"]
            files += [os.path.join(root, n) for n in names]
    return files


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--traces", type=Path, default=Path("/tmp/gleipnir-trace"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    paths = traced(args.traces)
    data = sorted(p for p in paths if p.startswith(f"{REPO}/data/"))
    if data:
        raise SystemExit(f"traced dataset files must not enter the image: {data[:5]}")
    if any(p.startswith("/tmp/gleipnir-serving-runtime/venv/") for p in paths):
        RUNTIME.append("/tmp/gleipnir-serving-runtime/venv")
    receipts = sorted(p for p in paths if p.startswith(f"{REPO}/results/"))
    caches = sorted(
        p
        for p in paths
        if p.startswith(f"{REPO}/.cache/")
        and not any(p.startswith(c + "/") for c in WHOLE_CACHES)
    )
    spec = {
        "runtime": RUNTIME,
        "caches": [c for c in WHOLE_CACHES if os.path.exists(c)] + caches,
        "repository": repository_files() + receipts,
        "release": ["/opt/gleipnir-release"],
    }
    args.output.write_text(json.dumps(spec, indent=1))
    print(json.dumps({k: len(v) for k, v in spec.items()}), flush=True)


if __name__ == "__main__":
    main()
