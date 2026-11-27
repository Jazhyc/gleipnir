"""Apply a checked, isolated TileLang 0.1.12 FP16 packing overload."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TARGET = ROOT / ".cache/kernels/flashqla-da06429"
HEADER = "tilelang/src/tl_templates/cuda/copy_sm100.h"
ORIGINAL_SHA256 = "da858d5cf8a7f5f780aced7a914059135d3ef9011dc9198d51d3d52624118f4f"
SIGNATURE = "pack_float16x4(const half x, const half y, const half z, const half w) {"


def patched_header(source: bytes) -> bytes:
    """Add a bit-preserving overload without removing the CUDA-half overload."""
    if hashlib.sha256(source).hexdigest() != ORIGINAL_SHA256:
        raise ValueError("unexpected original TileLang header")
    text = source.decode()
    start = text.index("__device__ __forceinline__ unsigned long long\n" + SIGNATURE)
    end = text.index("\n}", start) + 2
    original = text[start:end]
    overload = original.replace(
        SIGNATURE,
        "pack_float16x4(const cutlass::half_t x, const cutlass::half_t y, "
        "const cutlass::half_t z, const cutlass::half_t w) {\n"
        "  static_assert(sizeof(cutlass::half_t) == 2);",
    )
    return (text[:end] + "\n\n" + overload + text[end:]).encode()


def main() -> None:
    manifest_path = TARGET / "install_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    path = TARGET / HEADER
    source = path.read_bytes()
    current_sha = hashlib.sha256(source).hexdigest()
    patches = manifest.get("compiler_patches", [])
    if patches:
        if len(patches) != 1 or patches[0]["path"] != HEADER:
            raise ValueError("unexpected existing compiler patches")
        if current_sha != patches[0]["after_sha256"]:
            raise ValueError("patched TileLang header drift")
        if (
            patches[0]["script_sha256"]
            != hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        ):
            raise ValueError("compiler patch script drift")
        print("FP16 packing patch already verified")
        return
    result = patched_header(source)
    path.write_bytes(result)
    manifest["compiler_patches"] = [
        dict(
            path=HEADER,
            before_sha256=current_sha,
            after_sha256=hashlib.sha256(result).hexdigest(),
            script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            reason="cutlass::half_t overload for bit-preserving masked FP16 stores",
        )
    ]
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest["compiler_patches"], indent=2))


if __name__ == "__main__":
    main()
