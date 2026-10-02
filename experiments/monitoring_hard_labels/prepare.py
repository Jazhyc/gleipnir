"""Freeze monitoring-only source labels, teacher targets and CoT-removed ID."""

from __future__ import annotations

import json
from pathlib import Path

import yaml
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from experiments.tool_trajectory_monitoring.prompting import load_prompt_set
from gleipnir.monitoring_campaign_data import (
    digest,
    file_hash,
    read_rows,
    rerender_evaluation,
    rerender_training,
    validate_targets,
    write_json,
    write_rows,
)

ROOT = Path(__file__).resolve().parents[2]
CONFIG = Path(__file__).with_name("config.yaml")
DATA = ROOT / "data/monitoring_hard_labels"
OUTPUT = ROOT / "results/monitoring_hard_labels"
FRACTIONS = {"hard000": 0.0, "hard010": 0.1, "hard030": 0.3, "hard100": 1.0}


def configuration() -> dict:
    config = yaml.safe_load(CONFIG.read_text())
    if config["hard_label_fractions"] != FRACTIONS or config["learning_rate"] != 2e-5:
        raise ValueError("frozen four-cell design drift")
    return config


def validate_holdout(training: list[dict], evaluation: list[dict]) -> None:
    """Check original lineage hashes as well as transformed visible inputs."""
    training_hashes = {row["trajectory_sha256"] for row in training}
    evaluation_hashes = set()
    for row in evaluation:
        metadata = row["metadata"]
        evaluation_hashes.add(metadata["trajectory_sha256"])
        evaluation_hashes.add(metadata.get("original_trajectory_sha256"))
        original = metadata.get("original_metadata", {})
        evaluation_hashes.add(original.get("trajectory_sha256"))
    if training_hashes & evaluation_hashes:
        raise ValueError("training/ID trajectory lineage overlap")


def resolved_profile() -> dict:
    config = configuration()
    with initialize_config_dir(
        version_base=None, config_dir=str(ROOT / "src/gleipnir/configs/systems_screen")
    ):
        return OmegaConf.to_container(
            compose(config_name=config["profile"]), resolve=True
        )


def profile_hash() -> str:
    return digest(json.dumps(resolved_profile(), sort_keys=True))


def verify_preparation() -> dict:
    manifest = json.loads((DATA / "manifest.json").read_text())
    if manifest["config_sha256"] != file_hash(CONFIG):
        raise ValueError("prepared config drift")
    if manifest["profile_sha256"] != profile_hash():
        raise ValueError("resolved systems profile drift")
    reference = configuration().get("startup_validation_reference")
    if (
        reference
        and file_hash(ROOT / reference)
        != manifest["startup_validation_reference_sha256"]
    ):
        raise ValueError("startup validation reference checksum drift")
    template = load_prompt_set().student
    if any(h != template.template_sha256 for h in manifest["template_sha256"].values()):
        raise ValueError("student instruction drift")
    for relative, expected in manifest["files_sha256"].items():
        if file_hash(DATA / relative) != expected:
            raise ValueError(f"prepared input drift: {relative}")
    return manifest


def main() -> None:
    config = configuration()
    keys = ("training_source", "teacher_source", "id_source", "id_manifest")
    hashes = {key: file_hash(ROOT / config[key]) for key in keys}
    if any(hashes[key] != config[f"{key}_sha256"] for key in keys):
        raise ValueError("frozen source checksum drift")
    if (DATA / "manifest.json").exists():
        if verify_preparation()["sources_sha256"] != hashes:
            raise ValueError("source provenance drift")
        print("Reusing verified preparation", flush=True)
        return
    training = read_rows(ROOT / config["training_source"])
    targets = read_rows(ROOT / config["teacher_source"])
    evaluation = read_rows(ROOT / config["id_source"])
    if len(training) != 8688 or len(evaluation) != 3012:
        raise ValueError("population drift")
    if len({r["id"] for r in evaluation}) != 3012:
        raise ValueError("duplicate ID rows")
    validate_targets(training, targets)
    validate_holdout(training, evaluation)
    template = load_prompt_set().student
    training = rerender_training(training, template)
    evaluation = rerender_evaluation(evaluation, template)
    write_rows(DATA / "soft_targets.jsonl", targets)
    files = ["soft_targets.jsonl"]
    for variant in FRACTIONS:
        write_rows(DATA / variant / "student_rows.jsonl", training)
        write_rows(DATA / variant / "id.jsonl", evaluation)
        files.extend([f"{variant}/student_rows.jsonl", f"{variant}/id.jsonl"])
    write_json(
        DATA / "manifest.json",
        {
            "campaign_id": config["campaign_id"],
            "config_sha256": file_hash(CONFIG),
            "profile_sha256": profile_hash(),
            "startup_validation_reference_sha256": file_hash(
                ROOT / config["startup_validation_reference"]
            ),
            "startup_validation_policy": "reuse_validated_recipe_at_user_request",
            "sources_sha256": hashes,
            "template_sha256": {v: template.template_sha256 for v in FRACTIONS},
            "files_sha256": {p: file_hash(DATA / p) for p in files},
            "training_rows": len(training),
            "id_rows": len(evaluation),
            "hard_label_provenance": "unchanged source label; never teacher argmax",
            "soft_label_provenance": "unchanged prompt-aware Kimi K3 cache",
            "selection": config["selection"],
            "checkpoint_selection": "final one-epoch checkpoint only",
            "ood_used": False,
        },
    )
    print("Prepared four matched hard-label candidates and canonical ID", flush=True)


if __name__ == "__main__":
    main()
