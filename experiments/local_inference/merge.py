"""Verify the published adapter layouts and safe-merge a BF16 serving artifact."""

from __future__ import annotations

import json
import time

from experiments.local_inference.core import CONFIG, ROOT, write_json
from gleipnir.qwen35_adapter_rebase import rebase_key, sha256_file


def main() -> None:
    import torch
    from peft import PeftModel
    from safetensors import safe_open
    from transformers import AutoModelForImageTextToText, AutoTokenizer

    torch.set_num_threads(8)
    config = json.loads(CONFIG.read_text())
    adapter = ROOT / "adapter"
    release = json.loads((adapter / "release_manifest.json").read_text())
    if release["base_revision"] != config["base_revision"]:
        raise ValueError("released base revision mismatch")
    checked = {}
    for relative, metadata in release["files"].items():
        path = adapter / relative
        if relative in {"README.md", "LICENSE"}:
            continue
        checked[relative] = sha256_file(path)
        if checked[relative] != metadata["sha256"]:
            raise ValueError(f"release checksum mismatch: {relative}")
    with (
        safe_open(adapter / "adapter_model.safetensors", framework="pt") as master,
        safe_open(
            adapter / "vllm/adapter_model.safetensors", framework="pt"
        ) as serving,
    ):
        if {rebase_key(k) for k in master.keys()} != set(serving.keys()):
            raise ValueError("released adapter key layout mismatch")
        for key in master.keys():
            tensor = master.get_tensor(key)
            if tensor.dtype != torch.float32:
                raise ValueError("master is not FP32")
            if not torch.equal(tensor, serving.get_tensor(rebase_key(key))):
                raise ValueError(f"released adapter value mismatch: {key}")
    target = ROOT / "merged_bf16"
    if target.exists():
        manifest = json.loads((target / "merge_manifest.json").read_text())
        for name, expected in manifest["files"].items():
            if sha256_file(target / name) != expected:
                raise ValueError(f"merged artifact changed: {name}")
        if manifest["release_checksums"] != checked:
            raise ValueError("merge source mismatch")
        print("Reusing verified merged artifact", flush=True)
        return
    started = time.perf_counter()
    print("Loading BF16 base on CPU for safe merge", flush=True)
    base = AutoModelForImageTextToText.from_pretrained(
        ROOT / "base",
        dtype=torch.bfloat16,
        device_map="cpu",
        local_files_only=True,
    )
    adapted = PeftModel.from_pretrained(base, adapter / "vllm", local_files_only=True)
    merged = adapted.merge_and_unload(safe_merge=True).to(dtype=torch.bfloat16)
    target.mkdir()
    merged.save_pretrained(target, safe_serialization=True, max_shard_size="4GB")
    AutoTokenizer.from_pretrained(ROOT / "base").save_pretrained(target)
    write_json(
        target / "merge_manifest.json",
        {
            "base_id": config["base_id"],
            "base_revision": config["base_revision"],
            "adapter_id": config["adapter_id"],
            "adapter_revision": json.loads((ROOT / "downloads.json").read_text())[
                "adapter_revision"
            ],
            "release_checksums": checked,
            "master_dtype": "float32",
            "merge": "PEFT safe_merge=True into BF16 base; BF16 serving weights",
            "layout": "published rebased adapter; tensor-exact against causal master",
            "seconds": time.perf_counter() - started,
            "files": {
                p.name: sha256_file(p) for p in sorted(target.iterdir()) if p.is_file()
            },
            "base_files": {
                p.name: sha256_file(p)
                for p in sorted((ROOT / "base").iterdir())
                if p.is_file()
            },
        },
    )
    print(f"Merged artifact saved to {target}", flush=True)


if __name__ == "__main__":
    main()
