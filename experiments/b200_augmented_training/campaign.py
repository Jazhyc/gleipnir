"""Freeze the original augmentation and current training/evaluation contracts."""

from __future__ import annotations

import json
import math
from pathlib import Path

import yaml
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from experiments.monitor_injection_augmentation.audit import audit_rows
from experiments.monitoring_hard_labels.prepare import validate_holdout
from experiments.monitoring_hard_labels.train import validate_training_metadata
from gleipnir.data.monitoring import (
    digest,
    file_hash,
    read_rows,
    validate_targets,
    write_json,
)
from gleipnir.training.backends.native_fp4 import REFERENCE_SHA256

ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT = Path(__file__).parent
CONFIG = EXPERIMENT / "config.yaml"
DATA = ROOT / "data/b200_augmented_training"
OUTPUT = ROOT / "results/b200_augmented_training"
ADAPTER = OUTPUT / "4b/augmented"
LOGS = ROOT / "logs/runpod/b200_augmented_training"


def configuration() -> dict:
    return yaml.safe_load(CONFIG.read_text())


def input_path(config: dict, name: str) -> Path:
    return ROOT / config["inputs"][name]["path"]


def profile(config: dict) -> dict:
    with initialize_config_dir(
        version_base=None, config_dir=str(ROOT / "src/gleipnir/configs/systems_screen")
    ):
        return OmegaConf.to_container(
            compose(config_name=config["profile"]), resolve=True
        )


def make_job(config: dict, recipe: dict) -> dict:
    """Change data and output paths while retaining the selected training recipe."""
    return {
        **recipe,
        "job_name": "fp4-augmented-replication",
        "model": config["model"]["id"],
        "model_revision": config["model"]["revision"],
        "seed": config["seed"],
        "student_rows": str(input_path(config, "training")),
        "soft_targets": str(input_path(config, "teacher")),
        "selection_manifest": None,
        "train_rows": config["training_rows"],
        "num_train_epochs": config["num_train_epochs"],
        "max_steps": -1,
        "learning_rate": config["learning_rate"],
        "soft_loss_weight": 1.0,
        "direct_loss_weight": 0.0,
        "completion_loss_weight": 0.0,
        "save_steps": 1000000,
        "expected_initial_master_sha256": config["model"]["initial_master_sha256"],
        "output_dir": str(ADAPTER),
        "causal_adapter_dir": str(ADAPTER / "causal_adapter"),
        "model_dir": str(ADAPTER / "model"),
        "hydra_log_dir": str(LOGS / "hydra"),
    }


def source_paths() -> list[str]:
    """Archive the actual current framework and explicit reused experiment helpers."""
    paths = [str(p.relative_to(ROOT)) for p in EXPERIMENT.glob("*") if p.is_file()]
    paths += [
        "experiments/deception_distillation/train_student_sft.py",
        "experiments/monitor_injection_augmentation/audit.py",
        "experiments/monitor_injection_augmentation/prepare.py",
        "experiments/monitoring_hard_labels/prepare.py",
        "experiments/monitoring_hard_labels/train.py",
        "experiments/tool_trajectory_monitoring/prompting.py",
        "src/gleipnir/__init__.py",
        "src/gleipnir/_compat.py",
    ]
    paths += [
        "experiments/b200_apps/run.py",
        "experiments/b200_vllm031/runtime.py",
        "experiments/b200_attention_precision/startup.py",
    ]
    for directory in (
        "src/gleipnir/campaigns",
        "src/gleipnir/training",
        "src/gleipnir/evaluation",
        "src/gleipnir/data",
        "src/gleipnir/adapters",
        "src/gleipnir/configs/systems_screen",
    ):
        paths += [
            str(p.relative_to(ROOT))
            for p in (ROOT / directory).rglob("*")
            if p.suffix in (".py", ".yaml")
        ]
    paths += [
        f"src/gleipnir/kernels/fp4/{name}"
        for name in (
            "cudnn_fp4_mlp.py",
            "cudnn_fp4_gemm.py",
            "cudnn_fp4_epilogue.py",
            "nvfp4_pack.py",
        )
    ]
    return sorted(set(paths))


def check_binding(config: dict) -> dict:
    binding = json.loads((DATA / "manifest.json").read_text())
    if file_hash(CONFIG) != binding["config_sha256"]:
        raise ValueError("augmented campaign configuration drift")
    for name, expected in binding["source_sha256"].items():
        if file_hash(ROOT / name) != expected:
            raise ValueError(f"launched augmented source changed: {name}")
    for name, spec in config["inputs"].items():
        if file_hash(input_path(config, name)) != spec["sha256"]:
            raise ValueError(f"frozen augmented artifact changed: {name}")
    return binding


def prepare() -> dict:
    config = configuration()
    if (DATA / "manifest.json").exists():
        return check_binding(config)
    for name, spec in config["inputs"].items():
        if file_hash(input_path(config, name)) != spec["sha256"]:
            raise ValueError(f"frozen replication input changed: {name}")
    original = json.loads(input_path(config, "augmentation_manifest").read_text())
    completed = json.loads(input_path(config, "old_completion").read_text())
    if (
        completed["manifest_sha256"]
        != config["inputs"]["augmentation_manifest"]["sha256"]
    ):
        raise ValueError("trusted prior tokenizer audit population changed")
    rows, clean = (
        read_rows(input_path(config, "training")),
        read_rows(input_path(config, "clean_training")),
    )
    ledger, bank = (
        read_rows(input_path(config, "ledger")),
        json.loads(input_path(config, "templates").read_text()),
    )
    audited = audit_rows(clean, rows, ledger, bank)
    if (
        audited["rows"] != config["training_rows"]
        or audited["injected_rows"] != config["injected_rows"]
    ):
        raise ValueError("frozen 40% augmentation coverage changed")
    if (
        original["files_sha256"]["student_rows.jsonl"]
        != config["inputs"]["training"]["sha256"]
    ):
        raise ValueError("original materialization identity changed")
    targets = read_rows(input_path(config, "teacher"))
    validate_targets(rows, targets)
    ids = read_rows(input_path(config, "id"))
    validate_holdout(clean, ids)
    validate_holdout(rows, ids)
    if len(ids) != config["evaluation"]["id_rows"]:
        raise ValueError("canonical ID population changed")
    audit = json.loads(input_path(config, "token_audit").read_text())
    if (
        audit["total"] != config["expected_training_tokens"]
        or audit["rows"] != len(rows)
        or audit["maximum"] > 29696
        or audit["truncated"]
    ):
        raise ValueError("trusted tokenizer audit no longer fits the recipe")
    recipe = profile(config)
    job = make_job(config, recipe["recipe"])
    if job["startup_validation_reference_sha256"] != REFERENCE_SHA256:
        raise ValueError("selected native FP4 startup reference changed")
    if (
        len(rows) != 8688
        or math.ceil(len(rows) / config["logical_batch_size"])
        != config["expected_steps"]
    ):
        raise ValueError("one-epoch update count changed")
    link = DATA / "augmented/student_rows.jsonl"
    link.parent.mkdir(parents=True, exist_ok=True)
    if link.exists():
        if link.resolve() != input_path(config, "training").resolve():
            raise ValueError("prepared augmentation link changed")
    else:
        link.symlink_to(input_path(config, "training"))
    sources = {name: file_hash(ROOT / name) for name in source_paths()}
    for name in sources:
        target = OUTPUT / "executed_sources" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT / name).read_bytes())
    binding = {
        "config_sha256": file_hash(CONFIG),
        "inputs": config["inputs"],
        "source_sha256": sources,
        "profile_sha256": digest(json.dumps(recipe, sort_keys=True)),
        "augmentation_audit": audited,
        "token_audit_reused": True,
        "token_audit": audit,
        "selection": "fixed final 272-update adapter; no held-out selection",
    }
    write_json(DATA / "manifest.json", binding)
    write_json(OUTPUT / "resolved_profile.json", recipe)
    write_json(ADAPTER / "job.json", job)
    write_json(ADAPTER / "token_audit.json", audit)
    print("augmented_inputs_frozen", len(rows), len(ledger), audit["total"], flush=True)
    return binding


def validate_completion(metadata: dict, config: dict, job: dict) -> None:
    """Check every logical update and token, alongside the shared precision guard."""
    validate_training_metadata(
        metadata,
        job,
        config["model"]["initial_master_sha256"],
        expected_steps=config["expected_steps"],
    )
    adaptive = metadata["adaptive_microbatching"]
    sizes = [32] * 271 + [16]
    records = adaptive["records"]
    if (
        adaptive["logical_batch_sizes"] != sizes
        or metadata["train_metrics"]["epoch"] != 1.0
        or sum(r["tokens"] for r in records) != config["expected_training_tokens"]
        or any(r["tokens"] != r["padded_tokens"] for r in records)
    ):
        raise ValueError("augmented epoch/token/packing coverage failed")
    for update, size in enumerate(sizes, 1):
        if sorted(
            i for r in records if r["update"] == update for i in r["logical_indices"]
        ) != list(range(size)):
            raise ValueError("missing or repeated logical training row")
    durations = metadata["optimizer_step_timing"]["durations_seconds"]
    if len(durations) != 272 or not all(math.isfinite(t) and t > 0 for t in durations):
        raise ValueError("nonfinite or incomplete training timings")
