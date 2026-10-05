"""Compile native backward with runtime packed extents before integration."""

import hashlib
import json
from pathlib import Path

from gleipnir.nvidia_mxfp8_attention import UPSTREAM_REVISION, _cudnn
from gleipnir.nvidia_mxfp8_varlen_host import compile_packed_backward


def main():
    _cudnn()
    source = (
        Path(__file__).resolve().parents[2] / "src/gleipnir/nvidia_mxfp8_varlen_host.py"
    )
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    print(json.dumps({"status": "compiling", "source_sha256": digest}), flush=True)
    entry = compile_packed_backward((16, 4), f"{UPSTREAM_REVISION}:{digest}:16:4")
    print(
        json.dumps({"status": "compiled", "entry_type": type(entry).__name__}),
        flush=True,
    )


if __name__ == "__main__":
    main()
