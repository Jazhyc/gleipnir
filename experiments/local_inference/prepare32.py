"""Freeze the agreed stratified 32-row slice of the existing 512-row benchmark."""

import json
from pathlib import Path

from experiments.local_inference.core import DATA, read_rows, select_subset, write_json
from gleipnir.qwen35_adapter_rebase import sha256_file

CONFIG = Path("experiments/local_inference/iteration32.json")


def main() -> None:
    config = json.loads(CONFIG.read_text())
    source = DATA / "subset.jsonl"
    parent_hash = "f5800ce52b38184bf3854fbf8e7a91257a3774c2f0a859c598c97e26f5829ebf"
    if sha256_file(source) != parent_hash:
        raise ValueError("512-row parent identity mismatch")
    selected = select_subset(read_rows(source), config["subset_counts"], config["seed"])
    if len(selected) != 32 or sum(r["tokens"] for r in selected) != 338_780:
        raise ValueError("agreed 32-row selection changed")
    path = Path(config["input"])
    payload = "".join(json.dumps(r, sort_keys=True) + "\n" for r in selected)
    if path.exists() and path.read_text() != payload:
        raise ValueError("existing 32-row artifact changed")
    path.write_text(payload)
    manifest = {
        "subset_sha256": sha256_file(path),
        "rows": len(selected),
        "prompt_tokens": sum(r["tokens"] for r in selected),
        "min_tokens": min(r["tokens"] for r in selected),
        "max_tokens": max(r["tokens"] for r in selected),
        "reference_input": str(source),
        "reference_input_sha256": parent_hash,
        "source_manifest_sha256": sha256_file(DATA / "manifest.json"),
        "seed": config["seed"],
        "source_label_counts": config["subset_counts"],
        "selection": (
            "source/label quotas and four length bins; "
            "same seeded ID selection as 512-row preparation"
        ),
        "config_sha256": sha256_file(CONFIG),
    }
    write_json(Path(config["manifest"]), manifest)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
