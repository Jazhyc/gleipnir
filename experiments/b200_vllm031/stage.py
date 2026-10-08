"""Stage the completed candidate locally while retaining its persistent source."""

import hashlib
import json
from pathlib import Path

from gleipnir.serving.runtime import copy_dependency_tree
from gleipnir.serving.tokenizer_setup import prepare_native_tokenizer


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    source = root / ".venv-vllm031"
    target = Path("/tmp/gleipnir-vllm031-runtime")
    packages = source / "lib/python3.12/site-packages"
    tvm_metadata = list(packages.glob("apache_tvm_ffi-*.dist-info/METADATA"))
    if len(tvm_metadata) != 1:
        raise ValueError("locked TVM FFI metadata is incomplete")
    tvm_digest = hashlib.sha256(tvm_metadata[0].read_bytes()).hexdigest()
    for overlay in (
        root / ".cache/kernels/fa4",
        Path("/tmp/gleipnir-serving-runtime/fa4"),
    ):
        overlay_metadata = list(overlay.glob("apache_tvm_ffi-*.dist-info/METADATA"))
        if (
            len(overlay_metadata) != 1
            or hashlib.sha256(overlay_metadata[0].read_bytes()).hexdigest()
            != tvm_digest
        ):
            raise ValueError(f"restore the locked TVM FFI dependency in {overlay}")
    selection = json.loads(
        (root / "experiments/b200_inference_benchmark/serving_default.json").read_text()
    )
    parent_path = root / selection["host_parent"]
    if (
        hashlib.sha256(parent_path.read_bytes()).hexdigest()
        != selection["artifact_bindings"][selection["host_parent"]]
    ):
        raise ValueError("native tokenizer parent receipt changed")
    frontend = json.loads(parent_path.read_text())["frontend"]
    tokenizer = prepare_native_tokenizer(
        Path(frontend["package_path"]),
        root / ".cache/runtime-resume/gigatoken-0.10.0-exact.tar.gz",
        [source / "lib/python3.12/site-packages"],
        frontend["package_files"],
    )
    files = [source / "pyvenv.cfg"]
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
        "native_tokenizer": tokenizer,
        "tvm_ffi_metadata_sha256": tvm_digest,
    }
    output = root / "results/b200_vllm031/staged_runtime.json"
    output.write_text(json.dumps(receipt, indent=2) + "\n")
    print("vllm031_staged", receipt, flush=True)


if __name__ == "__main__":
    main()
