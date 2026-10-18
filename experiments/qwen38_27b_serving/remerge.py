"""Rebuild the campaign's merged BF16 27B on a new Pod and require identity.

Merged checkpoints live on ephemeral container disk. This reruns the
campaign's FP32 merge (`merge_checkpoint`) from the pinned base and the
volume's serving adapter, and requires the source shards and every output
file to match `results/qwen38-27b-s0-01/merged_artifact.json` byte for byte,
so the campaign's merged-parity and evaluation receipts apply unchanged.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from gleipnir.adapters.merge import merge_checkpoint
from gleipnir.data.monitoring import write_json

ROOT = Path(__file__).resolve().parents[2]
CAMPAIGN = ROOT / "results/qwen38-27b-s0-01"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads((CAMPAIGN / "config.json").read_text())
    recorded = json.loads((CAMPAIGN / "merged_artifact.json").read_text())
    report = merge_checkpoint(
        ROOT / config["base_model"],
        Path(recorded["adapter"]),
        Path(recorded["destination"]),
        expected_adapter_sha256=recorded["adapter_sha256"],
        model_id=config["model"]["id"],
        revision=config["model"]["revision"],
    )
    identical = {
        key: report[key] == recorded[key]
        for key in ("source_files_sha256", "files_sha256", "changed_projection_count")
    }
    write_json(args.output, {"identical": identical, "merge": report})
    if not all(identical.values()):
        raise ValueError(f"re-merged checkpoint differs from the campaign: {identical}")
    print("remerge_identical", report["seconds"], flush=True)


if __name__ == "__main__":
    main()
