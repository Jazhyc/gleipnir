"""Build the disposable merged monitor on Runpod's ephemeral container disk."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from experiments.b200_inference_benchmark.run import EXPERIMENT, OUTPUT, ROOT, write
from gleipnir.merged_lora import merge_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--destination",
        type=Path,
        default=Path("/tmp/gleipnir-merged/fp4-full-training-bf16"),
    )
    args = parser.parse_args()
    destination = args.destination.resolve()
    if not destination.is_relative_to(Path("/tmp")) or destination.is_relative_to(ROOT):
        raise ValueError("merged Runpod weights must use ephemeral /tmp storage")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.parent.stat().st_dev == ROOT.stat().st_dev:
        raise ValueError(
            "ephemeral destination is on the persistent project filesystem"
        )
    config = yaml.safe_load((EXPERIMENT / "config.yaml").read_text())
    base = (
        ROOT
        / ".cache/huggingface/hub"
        / ("models--" + config["model"].replace("/", "--"))
        / "snapshots"
        / config["revision"]
    )
    print(f"merge_start destination={destination} base={base}", flush=True)
    import torch

    torch.set_num_threads(16)
    manifest = merge_checkpoint(
        base,
        ROOT / config["adapter"],
        destination,
        expected_adapter_sha256=config["adapter_sha256"],
        model_id=config["model"],
        revision=config["revision"],
    )
    manifest["ephemeral_storage"] = True
    write(OUTPUT / "merged_artifact.json", manifest)
    print(
        json.dumps(
            {
                "merge_complete": True,
                "modules": manifest["merged_projection_count"],
                "seconds": manifest["seconds"],
                "destination": str(destination),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
