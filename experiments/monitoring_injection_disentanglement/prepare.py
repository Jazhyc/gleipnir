"""Freeze approved student-only intervention and unchanged evaluation content."""

from __future__ import annotations

import json
import shutil
from dataclasses import replace
from pathlib import Path

import yaml
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from experiments.monitoring_hard_labels.prepare import validate_holdout
from experiments.tool_trajectory_monitoring.prompting import load_prompt_set
from gleipnir.monitoring_campaign_data import (
    digest,
    file_hash,
    read_rows,
    rerender_evaluation,
    rerender_training,
    trajectory_from_prompt,
    validate_targets,
    write_json,
    write_rows,
)

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
CONFIG = HERE / "config.yaml"
DATA = ROOT / "data/monitoring_injection_disentanglement"
OUTPUT = ROOT / "results/monitoring_injection_disentanglement"
PROMPTS = ("neutral", "aggressive", "conservative")


def configuration() -> dict:
    return yaml.safe_load(CONFIG.read_text())


def templates() -> dict:
    original = load_prompt_set().student
    aggressive = (
        ROOT
        / "experiments/tool_trajectory_monitoring/prompts"
        / "student_prompt_injection_aware.txt"
    )
    return {
        "neutral": original,
        "aggressive": replace(
            original,
            instruction=aggressive.read_text(),
            prompt_set_id="tool-trajectory-student-injection-aware-v1",
        ),
        "conservative": replace(
            original,
            instruction=(HERE / "student_prompt.txt").read_text(),
            prompt_set_id="tool-trajectory-student-behavior-grounded-v1",
        ),
    }


def resolved_profile() -> dict:
    with initialize_config_dir(
        version_base=None, config_dir=str(ROOT / "src/gleipnir/configs/systems_screen")
    ):
        return OmegaConf.to_container(
            compose(config_name=configuration()["profile"]), resolve=True
        )


def rerender_benchmark(rows: list[dict], template) -> list[dict]:
    """Replace only the known neutral envelope, preserving arbitrary payload text."""
    neutral = templates()["neutral"]
    suffix = neutral.trajectory_close + "\n"
    result = []
    for row in rows:
        prompt = row["prompt"]
        if not prompt.startswith(neutral.cache_prefix) or not prompt.endswith(suffix):
            raise ValueError("benchmark source envelope drift")
        trajectory = prompt[len(neutral.cache_prefix) : -len(suffix)]
        if neutral.render(trajectory) != prompt:
            raise ValueError("benchmark trajectory recovery drift")
        result.append({**row, "prompt": template.render(trajectory)})
    return result


def verify_preparation() -> dict:
    manifest = json.loads((DATA / "manifest.json").read_text())
    if manifest["config_sha256"] != file_hash(CONFIG):
        raise ValueError("prepared configuration drift")
    if manifest["profile_sha256"] != digest(
        json.dumps(resolved_profile(), sort_keys=True)
    ):
        raise ValueError("prepared profile drift")
    if manifest["template_sha256"] != {
        k: v.template_sha256 for k, v in templates().items()
    }:
        raise ValueError("prepared prompt drift")
    for p, h in manifest["files_sha256"].items():
        if file_hash(DATA / p) != h:
            raise ValueError(f"prepared file drift: {p}")
    for p, h in manifest["source_sha256"].items():
        if file_hash(ROOT / p) != h:
            raise ValueError(f"source identity drift: {p}")
    return manifest


def main() -> None:
    config = configuration()
    if (DATA / "manifest.json").exists():
        verify_preparation()
        print("Reusing verified preparation", flush=True)
        return
    reviewed = (
        ROOT
        / "experiments/student_injection_awareness"
        / "student_prompt_behavior_grounded_draft.txt"
    )
    if reviewed.read_bytes() != (HERE / "student_prompt.txt").read_bytes():
        raise ValueError("approved instruction differs from reviewed draft")
    keys = (
        "training_source",
        "teacher_source",
        "id_source",
        "id_manifest",
        "startup_validation_reference",
    )
    sources = {}
    for key in keys:
        p = ROOT / config[key]
        if file_hash(p) != config[key + "_sha256"]:
            raise ValueError(f"frozen source drift: {key}")
        sources[config[key]] = file_hash(p)
    training = read_rows(ROOT / config["training_source"])
    targets = read_rows(ROOT / config["teacher_source"])
    if len(training) != config["training_rows"]:
        raise ValueError("training coverage drift")
    validate_targets(training, targets)
    identity = read_rows(ROOT / config["id_source"])
    if len(identity) != 3012 or len({r["id"] for r in identity}) != 3012:
        raise ValueError("ID coverage drift")
    validate_holdout(training, identity)
    DATA.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(ROOT / config["teacher_source"], DATA / "soft_targets.jsonl")
    files = ["soft_targets.jsonl"]
    all_templates = templates()
    write_rows(
        DATA / "train/student_rows.jsonl",
        rerender_training(training, all_templates["conservative"]),
    )
    files.append("train/student_rows.jsonl")
    benchmark_root = (ROOT / config["benchmark_root"]).resolve()
    populations = {}
    for key, split, expected in (
        ("benchmark_source", "benchmark", 5184),
        ("controls_source", "honest_controls", 3930),
    ):
        p = benchmark_root / config[key]
        if file_hash(p) != config[key + "_sha256"]:
            raise ValueError(f"benchmark source drift: {split}")
        population = read_rows(p)
        if (
            len(population) != expected
            or len({r["id"] for r in population}) != expected
        ):
            raise ValueError(f"benchmark coverage drift: {split}")
        populations[split] = population
        sources[str(Path(config["benchmark_root"]) / config[key])] = file_hash(p)
    groups = sorted({(r["source_dataset"], r["label"]) for r in training})
    cohort = []
    for source, label in groups:
        cohort.extend(
            sorted(
                (
                    r
                    for r in training
                    if (r["source_dataset"], r["label"]) == (source, label)
                ),
                key=lambda r: (len(r["student_prompt"]), r["index"]),
            )[:2]
        )
    for name, template in all_templates.items():
        write_rows(
            DATA / "id" / f"{name}.jsonl", rerender_evaluation(identity, template)
        )
        files.append(f"id/{name}.jsonl")
        for split, population in populations.items():
            write_rows(
                DATA / split / f"{name}.jsonl", rerender_benchmark(population, template)
            )
            files.append(f"{split}/{name}.jsonl")
        canaries = [
            {
                "id": r["index"],
                "prompt": template.render(
                    trajectory_from_prompt(r["student_prompt"], r["trajectory_sha256"])
                ),
                "metadata": {
                    "source_dataset": r["source_dataset"],
                    "ground_truth": r["label"],
                },
            }
            for r in cohort
        ]
        write_rows(DATA / "canaries" / f"{name}.jsonl", canaries)
        files.append(f"canaries/{name}.jsonl")
    write_json(
        DATA / "manifest.json",
        {
            "campaign_id": config["campaign_id"],
            "config_sha256": file_hash(CONFIG),
            "profile_sha256": digest(json.dumps(resolved_profile(), sort_keys=True)),
            "template_sha256": {k: v.template_sha256 for k, v in all_templates.items()},
            "source_sha256": sources,
            "files_sha256": {p: file_hash(DATA / p) for p in files},
            "training_rows": 8688,
            "teacher_targets_unchanged": True,
            "selection": (
                "final one-epoch checkpoint; all three prompts fixed before scoring"
            ),
        },
    )
    print("Prepared training, ID, injection benchmark and honest controls", flush=True)


if __name__ == "__main__":
    main()
