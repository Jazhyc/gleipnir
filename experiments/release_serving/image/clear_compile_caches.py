"""Write a layer that empties the partially traced compile caches.

The open() trace sees index files but not the cubins Inductor and Triton
load natively, so the first image shipped incomplete cache entries, which
fail vLLM's JIT warm-up (``KeyError: 'cubin'``). Cold compilation is short
(4B ready about 2 minutes, 27B about 2.5 minutes after download), so these
directories ship empty: an OCI opaque whiteout hides the lower-layer files.
"""

from __future__ import annotations

import argparse
import io
import tarfile
from pathlib import Path

COMPILE_CACHES = [
    "workspace/gleipnir/.cache/vllm/student_injection_awareness_v1",
    "workspace/gleipnir/.cache/training/shared/gpu-0/triton",
    "workspace/gleipnir/.cache/torchinductor/student_injection_awareness_v1",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    with tarfile.open(args.output, "w") as tar:
        for directory in COMPILE_CACHES:
            info = tarfile.TarInfo(directory)
            info.type, info.mode = tarfile.DIRTYPE, 0o755
            tar.addfile(info)
            marker = tarfile.TarInfo(f"{directory}/.wh..wh..opq")
            marker.size = 0
            tar.addfile(marker, io.BytesIO(b""))
    print("wrote", args.output, flush=True)


if __name__ == "__main__":
    main()
