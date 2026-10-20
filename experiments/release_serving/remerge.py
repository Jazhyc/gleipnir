"""Rebuild a release target's merged BF16 checkpoint and require identity.

Reruns the recorded FP32 merge (``merge_checkpoint``) from the pinned base and
the recorded adapter into the recorded destination, and requires the source
shards and every output file to match the run's ``merged_artifact.json`` byte
for byte, so that run's parity and evaluation receipts apply to the upload.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.release_serving.targets import ROOT, TARGETS
from gleipnir.adapters.merge import merge_checkpoint
from gleipnir.data.monitoring import write_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", choices=sorted(TARGETS))
    parser.add_argument("--base", type=Path, required=True, help="pinned base snapshot")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--adapter",
        type=Path,
        help="adapter copy to merge instead of the recorded path; the recorded"
        " adapter SHA-256 is still required",
    )
    args = parser.parse_args()
    recorded = json.loads((ROOT / TARGETS[args.model].merged_artifact).read_text())
    report = merge_checkpoint(
        args.base,
        args.adapter or Path(recorded["adapter"]),
        Path(recorded["destination"]),
        expected_adapter_sha256=recorded["adapter_sha256"],
        model_id=recorded["model"],
        revision=recorded["revision"],
    )
    identical = {
        key: report[key] == recorded[key]
        for key in ("source_files_sha256", "files_sha256", "changed_projection_count")
    }
    write_json(args.output, {"identical": identical, "merge": report})
    if not all(identical.values()):
        raise ValueError(
            f"re-merged checkpoint differs from the recorded run: {identical}"
        )
    print("remerge_identical", args.model, report["seconds"], flush=True)


if __name__ == "__main__":
    main()
