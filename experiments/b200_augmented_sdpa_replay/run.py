"""Replay historical SDPA training and reuse scores only on exact adapter identity."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from gleipnir.campaigns.monitoring.__main__ import execute
from gleipnir.campaigns.monitoring.contract import Campaign
from gleipnir.data.monitoring import file_hash, write_json

ROOT = Path(__file__).resolve().parents[2]
CONFIG = Path(__file__).with_name("config.yaml")


def historical_job_comparison(ctx: Campaign) -> dict:
    old = json.loads(ctx.input("historical_job").read_text())
    job = ctx.job()
    differences = {
        k: [old.get(k), job.get(k)]
        for k in old.keys() | job.keys()
        if old.get(k) != job.get(k)
    }
    allowed = {
        "job_name",
        "output_dir",
        "causal_adapter_dir",
        "model_dir",
        "hydra_log_dir",
        "expected_initial_master_sha256",
        "startup_validation_reference_sha256",
    }
    if set(differences) - allowed:
        raise ValueError(f"historical training recipe drift: {differences}")
    if (
        job["expected_initial_master_sha256"]
        != ctx.config["model"]["initial_tensor_sha256"]
    ):
        raise ValueError("initializer identity assertion changed")
    return {"passed": True, "differences": differences}


def adapter_identity(ctx: Campaign) -> dict:
    proof = {}
    for folder, weight, config in (
        ("causal_adapter", "historical_master", "historical_master_config"),
        ("model", "historical_export", "historical_export_config"),
    ):
        new_config = json.loads(
            (ctx.adapter / folder / "adapter_config.json").read_text()
        )
        old_config = json.loads(ctx.input(config).read_text())
        for value in (new_config, old_config):
            value["target_modules"] = sorted(value["target_modules"])
        actual = file_hash(ctx.adapter / folder / "adapter_model.safetensors")
        expected = file_hash(ctx.input(weight))
        proof[folder] = {
            "actual_sha256": actual,
            "historical_sha256": expected,
            "weights_identical": actual == expected,
            "configs_equivalent": new_config == old_config,
        }
    return {
        "identical": all(
            p["weights_identical"] and p["configs_equivalent"] for p in proof.values()
        ),
        "artifacts": proof,
    }


def validate_replay(ctx: Campaign) -> dict:
    new = json.loads(
        (ctx.adapter / "causal_adapter/training_metadata.json").read_text()
    )
    old = json.loads(ctx.input("historical_metadata").read_text())
    packing = new["sequence_packing"]
    if (
        packing.get("attention_backend", "sdpa") != "sdpa"
        or packing["full_attention"] != "segmented_causal_sdpa"
        or packing["initial_master_sha256"]
        != old["sequence_packing"]["initial_master_sha256"]
        or packing["bf16_matmul"] != old["sequence_packing"]["bf16_matmul"]
        or new["optimization"] != old["optimization"]
        or new["seed"] != old["seed"]
        or new["gated_delta_backend"]["package_sha256"]
        != old["gated_delta_backend"]["package_sha256"]
    ):
        raise ValueError("completed historical training arithmetic/settings drift")
    fields = (
        "examples",
        "logical_indices",
        "max_length",
        "padded_tokens",
        "tokens",
        "update",
    )
    a, b = (
        new["adaptive_microbatching"]["records"],
        old["adaptive_microbatching"]["records"],
    )
    if len(a) != len(b) or any(
        any(x[k] != y[k] for k in fields) for x, y in zip(a, b, strict=True)
    ):
        raise ValueError("historical physical batch/order drift")
    return {
        "passed": True,
        "matched_physical_batches": len(a),
        "tokens": sum(x["tokens"] for x in a),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("prepare", "run"), required=True)
    args = parser.parse_args()
    ctx = Campaign.load(ROOT, CONFIG)
    comparison = historical_job_comparison(ctx)
    if args.stage == "prepare":
        ctx.prepare()
        write_json(ctx.output / "historical_job_comparison.json", comparison)
        print("sdpa_replay_prepared", flush=True)
        return
    execute(ctx, through="train")
    write_json(ctx.output / "training_replay_audit.json", validate_replay(ctx))
    identity = adapter_identity(ctx)
    write_json(ctx.output / "historical_adapter_identity.json", identity)
    if identity["identical"]:
        write_json(
            ctx.output / "reused_evaluation.json",
            {
                "status": "complete",
                "scope": "historical_sdpa_training_replay",
                "new_evaluation_performed": False,
                "identity": identity,
                "predictions": ctx.config["inputs"]["historical_bf16_id"],
                "reason": (
                    "Exact master/export weights and equivalent configs; "
                    "reuse historical BF16 ID scores."
                ),
            },
        )
        write_json(
            ctx.output / "status.json",
            {
                "stage": "complete",
                "new_evaluation_performed": False,
            },
        )
        print("historical_sdpa_replay_identical_reused_scores", flush=True)
    else:
        print("historical_sdpa_replay_changed_fresh_bf16_evaluation", flush=True)
        execute(ctx, resume=True)


if __name__ == "__main__":
    main()
