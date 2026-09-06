"""Freeze the paired-prefix training arms using the existing monitoring recipe."""

import argparse
import importlib.metadata
import json
import math
import subprocess
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from experiments.monitoring_lr_sweep.prepare import (
    DEFAULT_DATA_DIR,
    DEFAULT_ID_INPUT,
    inspect_training_inputs,
    validate_id_separation,
)
from experiments.monitoring_lr_sweep.prepare import (
    make_jobs as make_lr_jobs,
)
from experiments.monitoring_prefix_supervision.prepare_training import prepare_training
from gleipnir.monitoring_systems_screen import (
    atomic_write_json,
    atomic_write_jsonl,
    sha256_file,
)


def make_jobs(config: dict, paired_manifest: dict) -> list[dict]:
    """Change only paired inputs and prefix weight from the proven LR recipe."""
    if (
        config["prefix_loss_weights"] != [0.25, 0.5]
        or config["learning_rate"] != 2e-5
        or config["epochs"] != 1
        or config["seed"] != 0
        or config["sampling_epoch"] != 0
        or config["strict_ood_consulted"] is not False
        or config["parent_rows"] != 8688
        or paired_manifest["parent_rows"] != 8688
        or paired_manifest["seed"] != 0
        or paired_manifest["epoch"] != 0
    ):
        raise ValueError("prefix training design drift")
    root = Path(config["result_dir"])
    base = next(
        job
        for job in make_lr_jobs(DEFAULT_DATA_DIR, root)
        if job["learning_rate"] == config["learning_rate"]
    )
    jobs = []
    for weight in config["prefix_loss_weights"]:
        name = f"prefix-w{int(weight * 100):03d}-lr2em05-seed0"
        output = root / "runs" / name
        jobs.append(
            {
                **base,
                "job_name": name,
                "design_role": "paired_prefix_candidate",
                "prefix_loss_weight": weight,
                "student_rows": config["paired_rows"],
                "student_rows_sha256": paired_manifest["output_sha256"],
                "soft_targets": config["full_soft_targets"],
                "prefix_cache_contract_sha256": paired_manifest[
                    "cache_contract_sha256"
                ],
                "expected_parents_with_prefix": paired_manifest["parents_with_prefix"],
                "output_dir": str(output),
                "causal_adapter_dir": str(output / "causal_adapter"),
                "model_dir": str(output / "model"),
            }
        )
    return jobs


def prepare() -> None:
    """Prepare only after annotation completion; never launch GPU work here."""
    with initialize_config_dir(
        version_base=None, config_dir=str(Path(__file__).parent.resolve())
    ):
        config = OmegaConf.to_container(compose(config_name="training"), resolve=True)
    root = Path(config["result_dir"])
    if (root / "manifest.json").exists():
        raise FileExistsError(root / "manifest.json")
    audit, selection = inspect_training_inputs(
        DEFAULT_DATA_DIR / "student_rows.jsonl", Path(config["full_soft_targets"])
    )
    heldout = validate_id_separation(DEFAULT_ID_INPUT, audit.pop("trajectory_hashes"))
    paired = prepare_training(
        Path(config["cache_dir"]),
        Path(config["references"]),
        Path(config["paired_rows"]),
        seed=config["seed"],
        numerical_exception=config.get("numerical_exception"),
    )
    jobs = make_jobs(config, paired)
    atomic_write_jsonl(root / "jobs.jsonl", jobs)
    atomic_write_jsonl(root / "selections/preflight-longest-32.jsonl", selection)
    evaluation = json.loads(Path(config["evaluation_template"]).read_text())
    evaluation.update(
        campaign_id=config["campaign_id"],
        hypothesis="Parent-normalized prefix supervision",
    )
    evaluation["scope"]["selection_rule"] = (
        "Two final prefix-weight endpoints on ID only; no intermediate selection."
    )
    evaluation["model_groups"]["4b"].update(
        jobs=str(root / "jobs.jsonl"),
        jobs_sha256=sha256_file(root / "jobs.jsonl"),
        expected_jobs=[job["job_name"] for job in jobs],
        parity_job=jobs[1]["job_name"],
    )
    atomic_write_json(root / "id_benchmark.json", evaluation)
    atomic_write_json(root / "resolved_config.json", config)
    paths = [
        root / name
        for name in (
            "jobs.jsonl",
            "id_benchmark.json",
            "resolved_config.json",
            "selections/preflight-longest-32.jsonl",
        )
    ]
    paths.extend([Path(config["baseline_result"]), Path(config["paired_rows"])])
    paired_path = Path(config["paired_rows"])
    paths.append(paired_path.with_suffix(paired_path.suffix + ".manifest.json"))
    atomic_write_json(
        root / "manifest.json",
        {
            "training": audit,
            "paired_training": paired,
            "held_out_id": heldout,
            "authoring_sha256": sha256_file(Path(__file__).with_name("training.yaml")),
            "files": {str(path): sha256_file(path) for path in paths},
        },
    )


def validate_completed(job: dict, *, preflight: bool = False) -> None:
    from experiments.monitoring_lr_sweep.core import validate_training_metadata

    metadata = validate_training_metadata(
        Path(job["causal_adapter_dir"]) / "training_metadata.json",
        job["learning_rate"],
        expected_steps=1 if preflight else 272,
        require_canary=preflight,
    )
    losses = metadata["losses"]
    if (
        losses.get("prefix_weight") != job["prefix_loss_weight"]
        or losses.get("prefix_normalization") != "parent_mean_one_sample_v1"
        or losses.get("accumulation_policy") != "explicit_microbatch_mean_v1"
        or losses.get("soft_weight") != 1
        or losses.get("soft_type") != "bce"
        or any(
            losses.get(key, 0) != 0
            for key in (
                "completion_weight",
                "direct_weight",
                "pairwise_weight",
                "mil_weight",
            )
        )
        or not math.isfinite(float(metadata["train_metrics"]["train_loss"]))
    ):
        raise ValueError("paired objective metadata drift")
    expected = job["expected_parents_with_prefix"]
    count = losses.get("parents_with_prefix", 0)
    if (preflight and count <= 0) or (not preflight and count != expected):
        raise ValueError("paired target population drift")


def reconstruct_jobs(config: dict) -> list[dict]:
    paired_path = Path(config["paired_rows"])
    paired = json.loads(
        paired_path.with_suffix(paired_path.suffix + ".manifest.json").read_text()
    )
    if sha256_file(paired_path) != paired["output_sha256"]:
        raise ValueError("paired data drift")
    return make_jobs(config, paired)


def execute(root: Path, revision: str | None) -> None:
    """Reuse the two-lane lifecycle after checking idle GPUs and training stack."""
    from experiments.monitoring_duration.run import execute as execute_two_lanes

    if not importlib.metadata.version("torch").startswith("2.11.0"):
        raise ValueError("use the preserved training environment, not the vLLM upgrade")
    memory = [
        int(v)
        for v in subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            text=True,
        ).splitlines()
    ]
    if len(memory) != 2 or max(memory) > 1024:
        raise RuntimeError("both GPUs must be idle; do not interrupt annotation")
    execute_two_lanes(
        root,
        revision,
        job_factory=reconstruct_jobs,
        completed_validator=validate_completed,
        logs_root=Path("logs/lambda/monitoring_prefix_training"),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action", choices=("prepare", "run"), nargs="?", default="prepare"
    )
    parser.add_argument(
        "--root", type=Path, default=Path("results/monitoring_prefix_training")
    )
    parser.add_argument("--revision")
    args = parser.parse_args()
    if args.action == "prepare":
        prepare()
    else:
        execute(args.root, args.revision)
