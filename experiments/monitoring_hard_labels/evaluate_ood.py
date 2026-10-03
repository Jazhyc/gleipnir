"""Evaluate the fixed ID winner and matched control on the frozen OOD suite."""

from __future__ import annotations

import argparse
import fcntl
import json
import math
import os
import shutil
from collections import Counter
from pathlib import Path

import yaml

from experiments.monitoring_hard_labels.prepare import (
    CONFIG,
    DATA,
    OUTPUT,
    ROOT,
    configuration,
    validate_holdout,
)
from experiments.tool_trajectory_monitoring.benchmark_gpt_oss_ood import summarize
from experiments.tool_trajectory_monitoring.prompting import load_prompt_set
from gleipnir.monitoring_campaign_data import (
    file_hash,
    read_rows,
    rerender_evaluation,
    write_json,
    write_rows,
)
from gleipnir.monitoring_campaign_evaluation import EvaluationContext, serving

OOD_CONFIG = Path(__file__).with_name("ood_config.yaml")
MANIFEST = DATA / "ood.manifest.json"
PREPARED = DATA / "ood/prompts.jsonl"


def followup_configuration() -> dict:
    config = yaml.safe_load(OOD_CONFIG.read_text())
    if file_hash(CONFIG) != config["id_config_sha256"]:
        raise ValueError("original ID configuration drift")
    if (config["candidate"], config["control"]) != ("hard030", "hard000"):
        raise ValueError("fixed OOD comparison drift")
    return config


def install_input(source: Path, config: dict) -> None:
    """Install byte-identical inputs without altering the original ID manifest."""
    if file_hash(source) != config["prepared_sha256"]:
        raise ValueError("prepared OOD checksum drift")
    for variant in (config["candidate"], config["control"]):
        destination = DATA / variant / "ood.jsonl"
        if destination.exists():
            if file_hash(destination) != config["prepared_sha256"]:
                raise ValueError("existing OOD input drift")
        else:
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)


def prepare() -> None:
    config = followup_configuration()
    for key in ("source", "source_manifest"):
        if file_hash(ROOT / config[key]) != config[f"{key}_sha256"]:
            raise ValueError("frozen OOD provenance drift")
    rows = read_rows(ROOT / config["source"])
    if len(rows) != config["rows"] or len({r["id"] for r in rows}) != config["rows"]:
        raise ValueError("OOD membership drift")
    if (
        dict(Counter(r["metadata"]["source_dataset"] for r in rows))
        != config["source_rows"]
    ):
        raise ValueError("OOD source coverage drift")
    base = configuration()
    training_path = ROOT / base["training_source"]
    if file_hash(training_path) != base["training_source_sha256"]:
        raise ValueError("training lineage source drift")
    validate_holdout(read_rows(training_path), rows)
    template = load_prompt_set().student
    rows = rerender_evaluation(rows, template)
    write_rows(PREPARED, rows)
    install_input(PREPARED, config)
    variants = (config["candidate"], config["control"])
    write_json(
        MANIFEST,
        {
            "campaign_id": config["campaign_id"],
            "config_sha256": file_hash(OOD_CONFIG),
            "original_id_config_sha256": config["id_config_sha256"],
            "sources_sha256": {
                key: config[f"{key}_sha256"] for key in ("source", "source_manifest")
            },
            "training_lineage_sha256": base["training_source_sha256"],
            "training_overlap": False,
            "template_sha256": {v: template.template_sha256 for v in variants},
            "files_sha256": {
                f"{v}/ood.jsonl": config["prepared_sha256"] for v in variants
            },
            "rows": config["rows"],
            "source_rows": config["source_rows"],
            "selection": config["selection"],
            "promotion": False,
        },
    )
    print("ood_prepared 6395 rows; zero training lineage overlap", flush=True)


def verify_inputs(config: dict) -> dict:
    manifest = json.loads(MANIFEST.read_text())
    if manifest["config_sha256"] != file_hash(OOD_CONFIG):
        raise ValueError("OOD manifest configuration drift")
    template = load_prompt_set().student.template_sha256
    for relative, expected in manifest["files_sha256"].items():
        if (
            file_hash(DATA / relative) != expected
            or expected != config["prepared_sha256"]
        ):
            raise ValueError("OOD input drift")
    if any(value != template for value in manifest["template_sha256"].values()):
        raise ValueError("OOD instruction drift")
    return manifest


def collect_summary(config: dict) -> None:
    results = {}
    for variant in (config["candidate"], config["control"]):
        directory = OUTPUT / "4b" / variant / "ood"
        result = json.loads((directory / "result.json").read_text())
        inputs = read_rows(DATA / variant / "ood.jsonl")
        predictions = read_rows(directory / "predictions.jsonl")
        if (
            len(predictions) != config["rows"]
            or len({r["id"] for r in predictions}) != config["rows"]
        ):
            raise ValueError("incomplete or duplicate OOD predictions")
        for row, prediction in zip(inputs, predictions, strict=True):
            if (
                row["id"] != prediction["id"]
                or row["metadata"]["ground_truth"] != prediction["label"]
                or row["metadata"]["source_dataset"] != prediction["source"]
                or row["metadata"]["rendered_prompt_sha256"]
                != prediction["source_prompt_sha256"]
                or result["config_sha256"] != prediction["config_sha256"]
                or not math.isfinite(prediction["score"])
            ):
                raise ValueError("OOD prediction identity/score drift")
        recomputed = summarize(predictions)
        if any(result[key] != value for key, value in recomputed.items()):
            raise ValueError("OOD metric audit failed")
        results[variant] = {
            "result_sha256": file_hash(directory / "result.json"),
            "predictions_sha256": file_hash(directory / "predictions.jsonl"),
            "parity_reuse_sha256": file_hash(directory / "parity_reuse.json"),
            "macro": result["metrics"]["macro"],
            "pooled": result["metrics"]["pooled"],
            "calibration": result["calibration"],
            "prompt_tokens": result["prompt_tokens"],
            "rows": result["rows"],
        }
    candidate = results[config["candidate"]]["macro"]["macro"]
    control = results[config["control"]]["macro"]["macro"]
    write_json(
        OUTPUT / "ood_summary.json",
        {
            "campaign_id": config["campaign_id"],
            "results": results,
            "candidate_minus_control": {
                k: candidate[k] - control[k] for k in ("pauroc_at_20", "auroc", "brier")
            },
            "candidate_selected_on": "canonical ID only; fixed before OOD",
            "one_seed_only": True,
            "promotion": False,
            "config_sha256": file_hash(OOD_CONFIG),
            "manifest_sha256": file_hash(MANIFEST),
            "membership_and_metrics_verified": True,
        },
    )


def run() -> None:
    config = followup_configuration()
    manifest = verify_inputs(config)
    base = configuration()
    base["campaign_id"] = config["campaign_id"]
    base["ood_followup_sha256"] = file_hash(OOD_CONFIG)
    context = EvaluationContext(
        DATA,
        OUTPUT,
        (config["candidate"], config["control"]),
        manifest["template_sha256"],
        splits=("ood",),
        manifest_path=MANIFEST,
    )
    status = OUTPUT / "ood_status.json"
    write_json(
        status,
        {
            "status": "running",
            "pid": os.getpid(),
            "config_sha256": file_hash(OOD_CONFIG),
        },
    )
    try:
        serving(context, base, "4b", parity_reuse=config["parity_reuse"])
        collect_summary(config)
    except Exception as error:
        write_json(
            status, {"status": "failed", "error": str(error), "pid": os.getpid()}
        )
        raise
    write_json(
        status,
        {
            "status": "complete",
            "summary_sha256": file_hash(OUTPUT / "ood_summary.json"),
        },
    )
    print("ood_campaign_complete", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--install-cached-input", type=Path)
    args = parser.parse_args()
    if args.prepare:
        prepare()
    elif args.install_cached_input:
        install_input(args.install_cached_input, followup_configuration())
        verify_inputs(followup_configuration())
    else:
        os.environ.update(
            PATH=os.pathsep.join(
                [
                    str(ROOT / ".venv/bin"),
                    f"{os.environ.get('CUDA_HOME', '/usr/local/cuda')}/bin",
                    os.environ.get("PATH", ""),
                ]
            ),
            VLLM_CACHE_ROOT=str(ROOT / ".cache/vllm/monitoring_hard_labels_v1"),
            TORCHINDUCTOR_CACHE_DIR=str(
                ROOT / ".cache/torchinductor/monitoring_hard_labels_v1"
            ),
            PYTHONUNBUFFERED="1",
            WANDB_MODE="disabled",
        )
        with (OUTPUT / "ood.campaign.lock").open("a") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            run()


if __name__ == "__main__":
    main()
