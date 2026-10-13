"""Prune authorized persistent merged weights after validating source artifacts."""

import argparse
import json
import struct
from pathlib import Path

from gleipnir.data.monitoring import file_hash, write_json

ROOTS = (
    "results/b200_shutdown",
    "results/b200_augmented_seed_repeat_storage",
    "results/b200_historical_augmented_id_storage",
)


def candidates(root: Path):
    """Keep all metadata, adapters, caches and active ephemeral models untouched."""
    verified, hashes = [], {}

    def checksum(path):
        if path not in hashes:
            hashes[path] = file_hash(path)
        return hashes[path]

    for prefix in ROOTS:
        for path in sorted((root / prefix).rglob("merge_manifest.json")):
            if path.is_symlink():
                raise ValueError("refuse symlinked manifest")
            manifest = json.loads(path.read_text())
            adapter = Path(manifest["adapter"])
            if not adapter.is_relative_to(root) or not adapter.is_dir():
                raise ValueError(f"missing persistent adapter: {adapter}")
            weights = adapter / "adapter_model.safetensors"
            if checksum(weights) != manifest["adapter_sha256"]:
                raise ValueError(f"adapter identity drift: {adapter}")
            if (
                checksum(adapter / "adapter_config.json")
                != manifest["adapter_config_sha256"]
            ):
                raise ValueError("adapter config drift")
            with weights.open("rb") as stream:
                header = json.loads(stream.read(struct.unpack("<Q", stream.read(8))[0]))
            if not any(k != "__metadata__" for k in header) or any(
                v["dtype"] != "F32" for k, v in header.items() if k != "__metadata__"
            ):
                raise ValueError("require preserved full-precision adapter")
            base = (
                root
                / ".cache/huggingface/hub"
                / ("models--" + manifest["model"].replace("/", "--"))
                / "snapshots"
                / manifest["revision"]
            )
            for name, expected in manifest["source_files_sha256"].items():
                if checksum(base / name) != expected:
                    raise ValueError(f"base identity drift: {base / name}")
            shards = []
            for name, expected in manifest["files_sha256"].items():
                if name.startswith("model") and name.endswith(".safetensors"):
                    shard = path.parent / name
                    if (
                        shard.is_symlink()
                        or shard.parent != path.parent
                        or not shard.is_file()
                    ):
                        raise ValueError("invalid or missing archived merged shard")
                    shards.append(
                        {
                            "path": str(shard.relative_to(root)),
                            "bytes": shard.stat().st_size,
                            "recorded_sha256": expected,
                        }
                    )
            if shards:
                verified.append(
                    {
                        "manifest": str(path.relative_to(root)),
                        "manifest_sha256": checksum(path),
                        "adapter": str(adapter.relative_to(root)),
                        "adapter_sha256": manifest["adapter_sha256"],
                        "base": str(base.relative_to(root)),
                        "source_artifacts_verified": True,
                        "shards": shards,
                    }
                )
    return verified


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    root = Path.cwd().resolve()
    output = root / "results/network-volume-cleanup01"
    plan = candidates(root)
    write_json(
        output / "plan.json",
        {
            "checkpoints": plan,
            "bytes": sum(s["bytes"] for p in plan for s in p["shards"]),
        },
    )
    if args.execute:
        if (output / "complete.json").exists():
            raise ValueError("preserve completed cleanup receipt")
        removed = []
        for item in plan:
            for shard in item["shards"]:
                path = root / shard["path"]
                if path.stat().st_size != shard["bytes"]:
                    raise ValueError("merged shard changed after planning")
                path.unlink()
                removed.append(shard)
                write_json(output / "progress.json", {"removed": removed})
            write_json(
                (root / item["manifest"]).parent / "weights_pruned.json",
                {"policy": "reconstruct_from_verified_base_and_FP32_adapter", **item},
            )
        write_json(
            output / "complete.json",
            {
                "removed": removed,
                "bytes": sum(r["bytes"] for r in removed),
                "checkpoints": len(plan),
                "adapters_and_caches_preserved": True,
            },
        )
    print(
        json.dumps(
            {
                "execute": args.execute,
                "checkpoints": len(plan),
                "gib": sum(s["bytes"] for p in plan for s in p["shards"]) / 2**30,
            }
        )
    )


if __name__ == "__main__":
    main()
