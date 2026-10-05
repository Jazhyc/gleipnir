"""Frozen full-epoch inputs, completion checks and the paired ID comparison."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any

import yaml
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from experiments.monitoring_hard_labels.prepare import validate_holdout
from experiments.tool_trajectory_monitoring.benchmark_gpt_oss_ood import summarize
from experiments.tool_trajectory_monitoring.prompting import load_prompt_set
from gleipnir.monitoring_campaign_data import (
    digest,
    file_hash,
    read_rows,
    validate_targets,
    write_json,
)
from gleipnir.native_fp4_training import REFERENCE_SHA256
from gleipnir.validated_startup import validation_reference

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data/b200_fp4_full_training"
OUTPUT = ROOT / "results/b200_fp4_full_training"
DIRECTORY = OUTPUT / "4b/fp4"
CONFIG = Path(__file__).with_name("config.yaml")
SOURCES = (
    "training_source",
    "teacher_source",
    "id_source",
    "canonical_id_source",
    "canonical_id_manifest",
    "control_metadata",
    "control_id_result",
    "control_id_predictions",
    "control_token_audit",
)


def configuration() -> dict[str, Any]:
    return yaml.safe_load(CONFIG.read_text())


def profile(config: dict[str, Any]) -> dict[str, Any]:
    with initialize_config_dir(
        version_base=None, config_dir=str(ROOT / "src/gleipnir/configs/systems_screen")
    ):
        return OmegaConf.to_container(
            compose(config_name=config["profile"]), resolve=True
        )


def verify_sources(config: dict[str, Any]) -> None:
    for key in SOURCES:
        if file_hash(ROOT / config[key]) != config[f"{key}_sha256"]:
            raise ValueError(f"frozen replication source drift: {key}")
    if load_prompt_set().student.template_sha256 != config["template_sha256"]:
        raise ValueError("regular student instruction drift")


def prepare() -> dict[str, Any]:
    config = configuration()
    verify_sources(config)
    rows = read_rows(ROOT / config["training_source"])
    targets = read_rows(ROOT / config["teacher_source"])
    evaluation = read_rows(ROOT / config["id_source"])
    validate_targets(rows, targets)
    validate_holdout(rows, evaluation)
    if len(rows) != config["training_rows"] or len(evaluation) != 3012:
        raise ValueError("replication population drift")
    control = read_rows(ROOT / config["control_id_predictions"])
    validate_prediction_membership(control, evaluation)
    links = {
        "fp4/student_rows.jsonl": "training_source",
        "soft_targets.jsonl": "teacher_source",
        "fp4/id.jsonl": "id_source",
    }
    for relative, key in links.items():
        path = DATA / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            path.symlink_to(os.path.relpath(ROOT / config[key], path.parent))
        if file_hash(path) != config[f"{key}_sha256"]:
            raise ValueError(f"prepared replication input drift: {relative}")
    resolved = profile(config)
    manifest = {
        "campaign_id": config["campaign_id"],
        "config_sha256": file_hash(CONFIG),
        "profile_sha256": digest(json.dumps(resolved, sort_keys=True)),
        "sources_sha256": {k: config[f"{k}_sha256"] for k in SOURCES},
        "template_sha256": {"fp4": config["template_sha256"]},
        "files_sha256": {p: config[f"{k}_sha256"] for p, k in links.items()},
        "training_rows": len(rows),
        "id_rows": len(evaluation),
        "teacher_targets_unchanged": True,
        "selection": "fixed final one-epoch checkpoint; no held-out tuning",
    }
    path = DATA / "manifest.json"
    if path.exists() and json.loads(path.read_text()) != manifest:
        raise ValueError("existing replication manifest drift")
    write_json(path, manifest)
    write_json(OUTPUT / "resolved_profile.json", resolved)
    return manifest


def make_job(config: dict[str, Any], recipe: dict[str, Any]) -> dict[str, Any]:
    return {
        **recipe,
        "job_name": "full-native-fp4-regular",
        "student_rows": str(DATA / "fp4/student_rows.jsonl"),
        "soft_targets": str(DATA / "soft_targets.jsonl"),
        "selection_manifest": None,
        "seed": config["seed"],
        "train_rows": config["training_rows"],
        "num_train_epochs": config["num_train_epochs"],
        "max_steps": -1,
        "learning_rate": config["learning_rate"],
        "soft_loss_weight": config["soft_loss_weight"],
        "direct_loss_weight": config["direct_loss_weight"],
        "save_steps": 1000000,
        "expected_initial_master_sha256": config["initial_master_sha256"],
        "output_dir": str(DIRECTORY),
        "causal_adapter_dir": str(DIRECTORY / "causal_adapter"),
        "model_dir": str(DIRECTORY / "model"),
        "hydra_log_dir": str(ROOT / "logs/runpod/b200_fp4_full_training/hydra"),
    }


def validate_training(metadata: dict[str, Any], config: dict[str, Any]) -> None:
    """Require one complete epoch with the actual selected precision/update contract."""
    job = make_job(config, profile(config)["recipe"])
    packing = metadata["sequence_packing"]
    reference = metadata["startup_validation"]
    path = Path(job["startup_validation_reference"])
    trusted = validation_reference(
        ROOT / path,
        native_fp4_mlp=True,
        expected_sha256=REFERENCE_SHA256,
    )
    adaptive = metadata["adaptive_microbatching"]
    native = metadata["quantization"]["full_bf16_lora"]["native_fp4_mlp"]
    expected_sizes = [32] * (config["training_rows"] // 32)
    if remainder := config["training_rows"] % 32:
        expected_sizes.append(remainder)
    sizes = adaptive["logical_batch_sizes"]
    records = adaptive["records"]
    audit = json.loads((ROOT / config["control_token_audit"]).read_text())
    if (
        metadata["training_state"]["global_step"] != config["expected_steps"]
        or metadata["train_metrics"]["epoch"] != 1.0
        or sizes != expected_sizes
        or sum(r["tokens"] for r in records) != audit["total"]
        or any(r["tokens"] != r["padded_tokens"] for r in records)
        or packing["initial_master_sha256"] != config["initial_master_sha256"]
        or packing["final_master_sha256"] == config["initial_master_sha256"]
        or reference["reference_sha256"] != trusted["reference_sha256"]
        or reference["policy"] != "reuse_selected_finite_native_fp4_recipe"
        or not adaptive["require_finite_gradients"]
        or native["master_dtype"] != "float32"
        or native["hardware_packing"] is not True
        or native["fused_descale"] is not True
        or len(native["modules"]) != 32
        or packing["attention_backend"] != "flash_attention_4"
        or packing["attention_version"] != "4.0.0b33"
        or metadata["gradient_checkpointing"]
        or metadata["quantization"]["enabled"]
        or metadata["optimization"]["learning_rate"] != config["learning_rate"]
        or metadata["losses"]["soft_weight"] != config["soft_loss_weight"]
        or metadata["losses"]["direct_weight"] != config["direct_loss_weight"]
    ):
        raise ValueError("full replication precision/coverage/update contract failed")
    for update, size in enumerate(sizes, 1):
        partitions = [r for r in records if r["update"] == update]
        if sorted(i for r in partitions for i in r["logical_indices"]) != list(
            range(size)
        ):
            raise ValueError("full replication physical coverage drift")
    durations = metadata["optimizer_step_timing"]["durations_seconds"]
    if len(durations) != config["expected_steps"] or not all(
        math.isfinite(t) and t > 0 for t in durations
    ):
        raise ValueError("full replication missing/nonfinite update timing")


def validate_prediction_membership(predictions: list[dict], inputs: list[dict]) -> None:
    expected = {r["id"]: r for r in inputs}
    if len(expected) != len(inputs) or len(predictions) != len(inputs):
        raise ValueError("ID prediction coverage drift")
    if len({r["id"] for r in predictions}) != len(predictions):
        raise ValueError("duplicate ID prediction")
    for row in predictions:
        if row["id"] not in expected:
            raise ValueError("unexpected ID prediction")
        source = expected[row["id"]]["metadata"]
        if (
            row["label"] != source["ground_truth"]
            or row["source"] != source["source_dataset"]
            or row["source_prompt_sha256"] != source["rendered_prompt_sha256"]
            or not math.isfinite(row["score"])
        ):
            raise ValueError("ID prediction input/label/score drift")


def write_comparison() -> dict[str, Any]:
    config = configuration()
    prepare()
    metadata = json.loads(
        (DIRECTORY / "causal_adapter/training_metadata.json").read_text()
    )
    validate_training(metadata, config)
    parity = json.loads((DIRECTORY / "serving_parity.json").read_text())
    if not parity["passed"]:
        raise ValueError("comparison requires passed adapter serving parity")
    inputs = read_rows(ROOT / config["id_source"])
    populations = {
        "fp4_fa4": read_rows(DIRECTORY / "id/predictions.jsonl"),
        "historical_bf16_sdpa": read_rows(ROOT / config["control_id_predictions"]),
    }
    results = {}
    for name, rows in populations.items():
        validate_prediction_membership(rows, inputs)
        results[name] = summarize(rows)
    macros = {k: v["metrics"]["macro"]["macro"] for k, v in results.items()}
    from gleipnir.calibration import binary_calibration

    calibration = {
        name: {
            source: binary_calibration(
                [r["label"] for r in subset],
                [r["score"] for r in subset],
                10,
            )
            for source, subset in {
                "pooled": rows,
                **{
                    s: [r for r in rows if r["source"] == s]
                    for s in sorted({r["source"] for r in rows})
                },
            }.items()
        }
        for name, rows in populations.items()
    }
    result = {
        "status": "complete",
        "campaign_id": config["campaign_id"],
        "rows_trained": config["training_rows"],
        "updates": config["expected_steps"],
        "id_rows": len(inputs),
        "metrics": results,
        "calibration": calibration,
        "macro_differences": {
            k: macros["fp4_fa4"][k] - macros["historical_bf16_sdpa"][k]
            for k in macros["fp4_fa4"]
        },
        "training": {
            "metrics": metadata["train_metrics"],
            "optimizer_step_timing": metadata["optimizer_step_timing"],
        },
        "serving_parity": parity["comparisons"],
        "config_sha256": file_hash(CONFIG),
        "control_result_sha256": config["control_id_result_sha256"],
        "candidate_master_sha256": file_hash(
            DIRECTORY / "causal_adapter/adapter_model.safetensors"
        ),
        "candidate_predictions_sha256": file_hash(DIRECTORY / "id/predictions.jsonl"),
        "limitations": [
            "one seed",
            "historical control uses BF16/SDPA; FP4/FA4 effects are combined",
            "no held-out selection or equivalence claim",
        ],
        "promotion": False,
        "ood_evaluated": False,
    }
    write_json(OUTPUT / "summary.json", result)
    return result
