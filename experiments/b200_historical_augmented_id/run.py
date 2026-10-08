"""Reuse campaign merge/reference/serving stages without any training stage."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

from gleipnir.campaigns.monitoring.__main__ import worker
from gleipnir.campaigns.monitoring.contract import Campaign
from gleipnir.data.monitoring import file_hash, read_rows, write_json

ROOT = Path(__file__).resolve().parents[2]
CONFIG = Path(__file__).with_name("config.yaml")


def canonicalize_historical_predictions(
    original: list[dict], canonical: list[dict], workload: list[dict]
) -> list[dict]:
    """Rewrap provenance only after binding the exact original scored prompts."""
    if len(original) != len(canonical) or len(original) != len(workload):
        raise ValueError("historical baseline coverage changed")
    result = []
    for old, row, rendered in zip(original, canonical, workload, strict=True):
        meta = row["metadata"]
        if (
            old["id"] != row["id"]
            or old["id"] != rendered["id"]
            or old["ground_truth"] != meta["ground_truth"]
            or old["source_dataset"] != meta["source_dataset"]
            or old["prompt_sha256"] != rendered["prompt_sha256"]
            or old["prompt_tokens"] != rendered["prompt_tokens"]
        ):
            raise ValueError("historical baseline prompt/label identity changed")
        result.append(
            {**old, "historical_original_metadata": old["original_metadata"], **meta}
        )
    return result


def bind_completed_adapter(ctx: Campaign) -> None:
    """Keep old masters untouched; put fresh references in a new wrapper."""
    historical = ctx.root / ctx.config["historical_adapter"]
    complete_bytes = ctx.input("completed_training").read_bytes()
    complete = json.loads(complete_bytes)
    if complete["status"] != "trained" or complete["steps"] != 272:
        raise ValueError("historical adapter is not the fixed completed checkpoint")
    for layout, key in (("causal_adapter", "master"), ("model", "serving")):
        directory = historical / layout
        if (
            file_hash(directory / "adapter_model.safetensors")
            != complete[key + "_sha256"]
            or ctx.input("completed_" + key).resolve()
            != (directory / "adapter_model.safetensors").resolve()
        ):
            raise ValueError("historical master/export identity changed")
        ctx.adapter.mkdir(parents=True, exist_ok=True)
        link = ctx.adapter / layout
        if link.exists() or link.is_symlink():
            if link.resolve() != directory.resolve():
                raise ValueError("adapter wrapper points to different weights")
        else:
            link.symlink_to(directory, target_is_directory=True)
    target = ctx.adapter / "complete.json"
    if target.exists() and target.read_bytes() != complete_bytes:
        raise ValueError("historical completion receipt changed")
    target.write_bytes(complete_bytes)
    write_json(
        ctx.output / "historical_adapter.json",
        {
            "source": str(historical),
            "master_sha256": complete["master_sha256"],
            "serving_sha256": complete["serving_sha256"],
            "completed_sha256": file_hash(ctx.input("completed_training")),
            "training_metadata_sha256": file_hash(ctx.input("completed_metadata")),
            "new_training": False,
        },
    )


def paired_controls(ctx: Campaign) -> None:
    """Compare identical ordered held-out identities without reselecting cases."""
    current = read_rows(ctx.output / "evaluation/id.jsonl")
    y = np.array([r["score"] for r in current])
    results = {}
    for name in ("historical_id", "fp4_trained_bf16_id", "bf16_trained_bf16_id"):
        previous = read_rows(ctx.input(name))
        if [r["id"] for r in previous] != [r["id"] for r in current]:
            raise ValueError("paired ID membership/order changed")
        x = np.array([r["score"] for r in previous])
        results[name] = {
            "mae": float(np.abs(y - x).mean()),
            "mean_score_delta": float((y - x).mean()),
            "correlation": float(np.corrcoef(x, y)[0, 1]),
            "fixed_half_decision_flips": int(((x >= 0.5) != (y >= 0.5)).sum()),
        }
    write_json(ctx.output / "paired_controls.json", results)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("prepare", "all"), required=True)
    args = parser.parse_args()
    ctx = Campaign.load(ROOT, CONFIG)
    if args.stage == "prepare":
        ctx.prepare()
        bind_completed_adapter(ctx)
        write_json(ctx.output / "status.json", {"stage": "prepared"})
        print("historical_id_prepared", flush=True)
        return
    ctx.check()
    bind_completed_adapter(ctx)
    if (ctx.output / "stage_launches").exists():
        raise ValueError("historical evaluation already attempted; retain receipts")
    active = ctx.serving / "server.json"
    if json.loads(active.read_text()) != json.loads(
        ctx.input("resident_server").read_text()
    ):
        raise ValueError("recorded resident model changed")
    subprocess.run(
        [
            sys.executable,
            "-m",
            "experiments.b200_vllm031.stop",
            "--archive-name",
            ctx.config["campaign_id"] + "_parent",
        ],
        cwd=ctx.root,
        env=dict(os.environ),
        check=True,
    )
    try:
        for stage in ("master-reference", "merge", "merged-reference", "evaluate"):
            write_json(ctx.output / "status.json", {"stage": stage})
            print("historical_id_stage", stage, flush=True)
            worker(ctx, stage)
        paired_controls(ctx)
        write_json(ctx.output / "status.json", {"stage": "complete", "rows": 3012})
        print("historical_id_complete", flush=True)
    except BaseException as error:
        write_json(ctx.output / "status.json", {"stage": "failed", "error": str(error)})
        raise


if __name__ == "__main__":
    main()
