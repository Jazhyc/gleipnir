"""Freeze the optimized-serving OOD contexts for the Gleipnir release candidates.

Each release candidate is scored on the frozen 6,395-row OOD suite through
its optimized serving stack, with no QAD, to measure the gap to its existing
BF16 scores; that gap is what QAD has to recover. The 4B context is the
v0.2 average's BF16 OOD context with only its identity and serving precision
changed (the 4B optimized default: FP8 attention projections, FP4 MLP/GDN,
MXFP8 attention). The model is re-merged and must reproduce the parent merge.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import yaml

from gleipnir.campaigns.monitoring.contract import Campaign
from gleipnir.data.monitoring import file_hash, write_json

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
OUT = ROOT / "results/optimized_gap"
BF16_4B = "teacher-seed-sdpa-avg012-ood01"
OPT_4B = "gleipnir4b-v02-ood-optimized01"


def optimized_4b_config(gpu_uuid: str, diagnostic: bool) -> dict:
    parent = yaml.safe_load((ROOT / "data" / BF16_4B / "config.yaml").read_text())
    config = copy.deepcopy(parent)
    config.update(
        campaign_id=OPT_4B,
        gpu_uuid=gpu_uuid,
        merged_model=f"/tmp/gleipnir-merged/{OPT_4B}",
        extra_sources=sorted(
            {
                *parent["extra_sources"],
                *(str(p.relative_to(ROOT)) for p in HERE.glob("*.py")),
            }
        ),
    )
    config["evaluation"]["serving_precision"] = "optimized"
    config["evaluation"]["failed_parity_diagnostic"] = diagnostic
    return config


def prepare_4b(gpu_uuid: str, diagnostic: bool = False) -> dict:
    data = ROOT / "data" / OPT_4B
    if data.exists():
        raise ValueError(f"refuse to overwrite {data}")
    data.mkdir(parents=True)
    path = data / "config.yaml"
    path.write_text(
        yaml.safe_dump(optimized_4b_config(gpu_uuid, diagnostic), sort_keys=False)
    )
    ctx = Campaign(ROOT, path, yaml.safe_load(path.read_text()))
    manifest = {
        "files_sha256": {
            str(path.relative_to(ROOT)): file_hash(path),
            **{s["path"]: s["sha256"] for s in ctx.config["inputs"].values()},
        },
        "source_sha256": ctx.sources(),
        "parent": ctx.config["parent_campaign"],
    }
    write_json(data / "manifest.json", manifest)
    print("optimized_gap_prepared", json.dumps({"context": OPT_4B}), flush=True)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu-uuid", required=True)
    parser.add_argument("--diagnostic", action="store_true")
    args = parser.parse_args()
    prepare_4b(args.gpu_uuid, args.diagnostic)
