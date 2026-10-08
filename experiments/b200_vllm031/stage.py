"""Stage the completed candidate locally while retaining its persistent source."""

import hashlib
import json
from pathlib import Path

from gleipnir.serving.runtime import copy_dependency_tree


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    source = root / ".venv-vllm031"
    target = Path("/tmp/gleipnir-vllm031-runtime")
    files = [source / "pyvenv.cfg"]
    packages = source / "lib/python3.12/site-packages"
    for pattern in (
        "vllm-*/METADATA",
        "torch-*/METADATA",
        "triton-*/METADATA",
        "transformers-*/METADATA",
        "flashinfer_python-*/METADATA",
    ):
        matches = list(packages.glob(pattern))
        if len(matches) != 1:
            raise ValueError(f"candidate package identity is incomplete: {pattern}")
        files.extend(matches)
    hashes = {
        str(p.relative_to(source)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in files
    }
    copy_dependency_tree(source, target)
    for relative, digest in hashes.items():
        if hashlib.sha256((target / relative).read_bytes()).hexdigest() != digest:
            raise ValueError(f"candidate staging identity mismatch: {relative}")
    receipt = {
        "source": str(source),
        "python": str(target / "bin/python"),
        "metadata_sha256": hashes,
        "persistent_caches_preserved": True,
    }
    output = root / "results/b200_vllm031/staged_runtime.json"
    output.write_text(json.dumps(receipt, indent=2) + "\n")
    print("vllm031_staged", receipt, flush=True)


if __name__ == "__main__":
    main()
