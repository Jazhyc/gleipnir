"""Compile native backward with runtime packed extents before integration."""

import hashlib
import json

from gleipnir.nvidia_mxfp8_attention import UPSTREAM_REVISION, _cudnn
from gleipnir.nvidia_mxfp8_varlen_host import compile_packed_backward
from gleipnir.repository import repo_root


def main():
    _cudnn()
    source = (
        repo_root(__file__) / "src/gleipnir/kernels/mxfp8/nvidia_mxfp8_varlen_host.py"
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
