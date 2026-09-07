"""Freeze nine subset-duration endpoints with an explicit evaluation barrier."""

from __future__ import annotations

import argparse
import fcntl
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from experiments.adapter_capacity_scaling.run_lambda import runtime_environment
from experiments.monitoring_duration.run import summarize
from experiments.monitoring_lr_sweep.core import validate_training_metadata
from experiments.monitoring_lr_sweep.prepare import (
    DEFAULT_ID_INPUT,
    inspect_training_inputs,
    validate_id_separation,
)
from experiments.monitoring_lr_sweep.run_lambda import gpu_training_environment
from experiments.monitoring_objective_ablation.run_lambda import run_serving_parity
from experiments.tool_trajectory_monitoring.benchmark_distilled_ood import (
    validate_config,
    validate_inputs,
    validate_jobs,
)
from experiments.tool_trajectory_monitoring.run_distillation_lambda import (
    verify_job_inputs,
)
from gleipnir.campaign_status import CampaignStatus
from gleipnir.monitoring_systems_screen import (
    atomic_write_json,
    atomic_write_jsonl,
    read_jsonl,
    sha256_file,
)
from gleipnir.qwen35_fast_training import ensure_qwen35_long_trajectory_kernels
from gleipnir.staged_lanes import run_staged_lanes

ROOT = Path("results/monitoring_subset_duration")
LOGS = Path("logs/lambda/monitoring_subset_duration")


def make_jobs(config: dict[str, Any]) -> list[dict[str, Any]]:
    """Reuse the exact selected recipes and subset; initialize every cell fresh."""
    if (
        config["epochs"],
        config["train_rows"],
        config["learning_rate"],
        config["seed"],
        config["strict_ood_consulted"],
    ) != ([2, 3, 5], 1738, 2e-5, 0, False):
        raise ValueError("authorized duration design drift")
    if config["prefix_stage_after"] != "all_six_stage1_evaluations":
        raise ValueError("prefix evaluation barrier drift")
    standard = next(
        j
        for j in read_jsonl(Path(config["standard_jobs"]))
        if j["job_name"] == config["standard_job"]
    )
    mil = next(
        j
        for j in read_jsonl(Path(config["mil_jobs"]))
        if j["job_name"] == config["mil_job"]
    )
    if (
        mil["mil_pooling"],
        mil["mil_loss_weight"],
        mil["mil_max_instances"],
        mil["mil_temperature"],
    ) != ("logmeanexp", 0.25, 8, 1.0):
        raise ValueError("historical best MIL identity drift")
    if standard["train_rows"] != 1738 or standard["num_train_epochs"] != 1:
        raise ValueError("historical 20% baseline identity drift")
    prefix = next(
        j
        for j in read_jsonl(Path(config["prefix_jobs"]))
        if j["job_name"] == config["prefix_job"]
    )
    if prefix["prefix_loss_weight"] != 0.1:
        raise ValueError("best prefix weight drift")
    root = Path(config["result_dir"])
    jobs = []
    for epoch in config["epochs"]:
        for objective, source in (("soft", standard), ("mil", mil)):
            name = f"{objective}-pct020-lr2em05-epochs{epoch}-seed0"
            output = root / "runs" / name
            jobs.append(
                {
                    **source,
                    "job_name": name,
                    "objective": objective,
                    "design_role": "subset_duration",
                    "train_rows": 1738,
                    "data_fraction": 0.2,
                    "num_train_epochs": float(epoch),
                    "expected_steps": 55 * epoch,
                    "save_steps": 55 * epoch,
                    "max_steps": -1,
                    "completion_loss_weight": 0.0,
                    "prefix_loss_weight": 0.0,
                    "selection_manifest": standard["selection_manifest"],
                    "selection_sha256": standard["selection_sha256"],
                    "output_dir": str(output),
                    "causal_adapter_dir": str(output / "causal_adapter"),
                    "model_dir": str(output / "model"),
                }
            )
    for epoch in config["epochs"]:
        name = f"prefix-pct020-w010-lr2em05-epochs{epoch}-seed0"
        output = root / "runs" / name
        jobs.append(
            {
                **prefix,
                "job_name": name,
                "objective": "prefix",
                "train_rows": 1738,
                "data_fraction": 0.2,
                "num_train_epochs": float(epoch),
                "expected_steps": 55 * epoch,
                "save_steps": 55 * epoch,
                "max_steps": -1,
                "completion_loss_weight": 0.0,
                "mil_loss_weight": 0.0,
                "selection_manifest": standard["selection_manifest"],
                "selection_sha256": standard["selection_sha256"],
                "expected_parents_with_prefix": config["subset_parents_with_prefix"],
                "output_dir": str(output),
                "causal_adapter_dir": str(output / "causal_adapter"),
                "model_dir": str(output / "model"),
            }
        )
    return jobs


def lane_plan(
    jobs: list[dict[str, Any]],
) -> list[list[tuple[str, list[dict[str, Any]]]]]:
    """Return stage-one objective lanes and balanced stage-two prefix lanes."""
    return [
        [
            (kind, [j for j in jobs if j["objective"] == kind])
            for kind in ("soft", "mil")
        ],
        [
            (
                "prefix5",
                [
                    j
                    for j in jobs
                    if j["objective"] == "prefix" and j["num_train_epochs"] == 5
                ],
            ),
            (
                "prefix23",
                [
                    j
                    for j in jobs
                    if j["objective"] == "prefix" and j["num_train_epochs"] != 5
                ],
            ),
        ],
    ]


def prepare() -> None:
    with initialize_config_dir(
        version_base=None, config_dir=str(Path(__file__).parent.resolve())
    ):
        config = OmegaConf.to_container(compose(config_name="config"), resolve=True)
    root = Path(config["result_dir"])
    if root.exists():
        raise FileExistsError(root)
    standard = next(
        j
        for j in read_jsonl(Path(config["standard_jobs"]))
        if j["job_name"] == config["standard_job"]
    )
    prefix = next(
        j
        for j in read_jsonl(Path(config["prefix_jobs"]))
        if j["job_name"] == config["prefix_job"]
    )
    selected_ids = {
        r["index"] for r in read_jsonl(Path(standard["selection_manifest"]))
    }
    paired_path = Path(prefix["student_rows"])
    paired_sidecar = paired_path.with_suffix(paired_path.suffix + ".manifest.json")
    paired = json.loads(paired_sidecar.read_text())
    if (
        paired["output_sha256"] != sha256_file(paired_path)
        or paired["cache_contract_sha256"] != prefix["prefix_cache_contract_sha256"]
    ):
        raise ValueError("paired provenance drift")
    config["subset_parents_with_prefix"] = sum(
        bool(r.get("prefix_student_prompt"))
        for r in read_jsonl(paired_path)
        if r["index"] in selected_ids
    )
    if config["subset_parents_with_prefix"] <= 0:
        raise ValueError("missing subset prefix targets")
    jobs = make_jobs(config)
    verify_job_inputs(jobs)
    audit, _ = inspect_training_inputs(
        Path(jobs[0]["student_rows"]), Path(jobs[0]["soft_targets"])
    )
    heldout = validate_id_separation(DEFAULT_ID_INPUT, audit.pop("trajectory_hashes"))
    selected = read_jsonl(Path(jobs[0]["selection_manifest"]))
    keys = {(r["dataset"], r["index"]) for r in selected}
    if len(selected) != 1738 or len(keys) != 1738:
        raise ValueError("20% selection cardinality drift")
    direct = {}
    for job in (jobs[0], jobs[1], jobs[6]):
        rows = {
            (r["dataset"], r["index"]): r
            for r in read_jsonl(Path(job["student_rows"]))
            if (r["dataset"], r["index"]) in keys
        }
        if set(rows) != keys:
            raise ValueError("selected parents missing")
        direct[job["objective"]] = {
            k: {
                field: r[field]
                for field in (
                    "student_prompt",
                    "student_prompt_sha256",
                    "trajectory_sha256",
                    "label",
                )
            }
            for k, r in rows.items()
        }
    if direct["soft"] != direct["mil"] or direct["soft"] != direct["prefix"]:
        raise ValueError("standard/MIL/prefix direct inputs differ")
    atomic_write_jsonl(root / "jobs.jsonl", jobs)
    atomic_write_jsonl(
        root / "selections/preflight-longest-32.jsonl",
        sorted(selected, key=lambda r: r["student_direct_tokens"], reverse=True)[:32],
    )
    paths = [root / "jobs.jsonl", root / "selections/preflight-longest-32.jsonl"]
    for objective, lane in [entry for stage in lane_plan(jobs) for entry in stage]:
        evaluation = json.loads(Path(config["evaluation_template"]).read_text())
        evaluation.update(
            campaign_id=config["campaign_id"] + "-" + objective,
            hypothesis="Duration and MIL on the frozen 20% subset",
        )
        evaluation["scope"]["selection_rule"] = (
            "Nine predeclared final endpoints in two stages; no OOD."
        )
        evaluation["model_groups"]["4b"].update(
            jobs=str(root / "jobs.jsonl"),
            jobs_sha256=sha256_file(root / "jobs.jsonl"),
            jobs_manifest_expected_jobs=[j["job_name"] for j in jobs],
            expected_jobs=[j["job_name"] for j in lane],
            expected_target=lane[0]["target"],
            parity_job=lane[0]["job_name"],
        )
        path = root / f"id_{objective}.json"
        atomic_write_json(path, evaluation)
        validate_config(evaluation)
        validate_inputs(evaluation)
        validate_jobs(evaluation, "4b")
        paths.append(path)
    atomic_write_json(root / "resolved_config.json", config)
    paths.extend(
        [
            root / "resolved_config.json",
            Path(config["baseline_result"]),
            Path(config["standard_jobs"]),
            Path(config["mil_jobs"]),
            Path(config["prefix_jobs"]),
            paired_sidecar,
            Path(__file__).with_name("config.yaml"),
            Path(__file__),
        ]
    )
    atomic_write_json(
        root / "manifest.json",
        {
            "training": audit,
            "held_out_id": heldout,
            "selected_direct_inputs_equal": True,
            "paired_training": paired,
            "subset_parents_with_prefix": config["subset_parents_with_prefix"],
            "files": {str(p): sha256_file(p) for p in paths},
        },
    )


def validate_completed(job: dict[str, Any], *, preflight: bool = False) -> None:
    """Check kernels, precision, systems policy, objective and completed updates."""
    metadata = validate_training_metadata(
        Path(job["causal_adapter_dir"]) / "training_metadata.json",
        job["learning_rate"],
        expected_steps=1 if preflight else job["expected_steps"],
        require_canary=preflight,
        checkpointing_policy=job["gradient_checkpointing_policy"],
        compilation_policy=job["selective_torch_compile_policy"],
    )
    losses = metadata["losses"]
    expected = {
        "soft_weight": 1.0,
        "soft_type": "bce",
        "completion_weight": 0.0,
        "direct_weight": 0.0,
        "pairwise_weight": 0.0,
        "prefix_weight": job["prefix_loss_weight"],
        "mil_weight": job["mil_loss_weight"],
        "accumulation_policy": "explicit_microbatch_mean_v1",
        "model_accepts_loss_kwargs": False,
    }
    if job["objective"] == "mil":
        expected.update(
            mil_pooling="logmeanexp",
            mil_temperature=1.0,
            mil_top_k=3,
            mil_max_instances=8,
        )
        if metadata["sdpa"] != {
            "flash_required": True,
            "flash_enabled": True,
            "math_enabled": False,
            "memory_efficient_enabled": False,
            "cudnn_enabled": False,
        }:
            raise ValueError("MIL flash SDPA policy drift")
    if job["objective"] == "prefix":
        expected["prefix_normalization"] = "parent_mean_one_sample_v1"
        count = losses.get("parents_with_prefix", 0)
        if (preflight and count <= 0) or (
            not preflight and count != job["expected_parents_with_prefix"]
        ):
            raise ValueError("prefix parent count drift")
    for key, value in expected.items():
        if losses.get(key) != value:
            raise ValueError(f"objective metadata drift: {key}")
    if not math.isfinite(float(metadata["train_metrics"]["train_loss"])):
        raise ValueError("nonfinite training loss")


def train(
    path: Path, job: dict[str, Any], environment: dict[str, str], log_path: Path
) -> None:
    with log_path.open("a") as log:
        subprocess.run(
            [
                sys.executable,
                "-u",
                "-m",
                "experiments.tool_trajectory_monitoring.run_distillation_train",
                "--jobs",
                str(path),
                "--job-name",
                job["job_name"],
                "--allow-non-scaling-job",
            ],
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
        )


def execute(root: Path, revision: str | None) -> None:
    """Own the reserved GPUs with independent short-first objective lanes."""
    manifest = json.loads((root / "manifest.json").read_text())
    for path, checksum in manifest["files"].items():
        if sha256_file(Path(path)) != checksum:
            raise ValueError(f"frozen contract drift: {path}")
    config = json.loads((root / "resolved_config.json").read_text())
    jobs = read_jsonl(root / "jobs.jsonl")
    if jobs != make_jobs(config):
        raise ValueError("job reconstruction drift")
    verify_job_inputs(jobs)
    if subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
        text=True,
    ).strip():
        raise RuntimeError("GPUs are not idle")
    LOGS.mkdir(parents=True, exist_ok=True)
    os.environ.update(runtime_environment("0"))
    fast = ensure_qwen35_long_trajectory_kernels()
    if revision:
        fast["GLEIPNIR_COMMIT"] = revision
    atomic_write_json(
        root / "hardware.json",
        {
            "probe": subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=name,uuid,memory.total,driver_version",
                    "--format=csv",
                ],
                text=True,
            ),
            "started_at_unix": time.time(),
            "revision": revision,
        },
    )

    def lane(gpu: int, entry: tuple[str, list[dict[str, Any]]]) -> None:
        objective, lane_jobs = entry
        status = CampaignStatus(root / f"status_{objective}.json", lane_jobs, revision)
        environment = gpu_training_environment(fast, gpu)
        serving = runtime_environment(str(gpu))
        if revision:
            serving["GLEIPNIR_COMMIT"] = revision
        try:
            output = root / "preflight" / objective
            selection = root / "selections/preflight-longest-32.jsonl"
            preflight = {
                **lane_jobs[0],
                "job_name": "preflight-" + objective,
                "train_rows": 32,
                "max_steps": 1,
                "num_train_epochs": -1,
                "save_steps": 1,
                "selection_manifest": str(selection),
                "selection_sha256": sha256_file(selection),
                "selective_torch_compile_canary_tokens": 2048,
                "output_dir": str(output),
                "causal_adapter_dir": str(output / "causal_adapter"),
                "model_dir": str(output / "model"),
            }
            atomic_write_jsonl(output / "jobs.jsonl", [preflight])
            status.update(phase="preflight", preflight="running")
            train(
                output / "jobs.jsonl",
                preflight,
                environment,
                LOGS / f"preflight_{objective}.log",
            )
            validate_completed(preflight, preflight=True)
            status.update(phase="training", preflight="passed")
            for job in lane_jobs:
                with status.job(job["job_name"], gpu=gpu):
                    train(
                        root / "jobs.jsonl",
                        job,
                        environment,
                        LOGS / f"{job['job_name']}.log",
                    )
                    validate_completed(job)
            evaluation = root / f"id_{objective}.json"
            status.update(phase="serving_parity")
            run_serving_parity(evaluation, root / objective, environment, serving)
            status.update(phase="id_evaluation")
            with (LOGS / f"evaluation_{objective}.log").open("a") as log:
                subprocess.run(
                    [
                        sys.executable,
                        "-u",
                        "-m",
                        "experiments.tool_trajectory_monitoring.benchmark_distilled_ood",
                        "--config",
                        str(evaluation),
                        "--model-size",
                        "4b",
                        "--output-root",
                        str(root / "id_evaluation"),
                    ],
                    env=serving,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=True,
                )
            status.update(
                state="complete", phase="complete", completed_at_unix=time.time()
            )
        except BaseException as error:
            status.update(state="failed", error=repr(error))
            raise

    def stage_complete(stage: int) -> None:
        if stage == 0:
            summarize(root, jobs[:6], config)
            atomic_write_json(
                root / "stage1_summary.json",
                json.loads((root / "summary.json").read_text()),
            )
            atomic_write_json(
                root / "stage1_complete.json", {"completed_at_unix": time.time()}
            )

    run_staged_lanes(lane_plan(jobs), lane, stage_complete)
    summarize(root, jobs, config)
    atomic_write_json(
        root / "completion.json",
        {"state": "complete", "completed_at_unix": time.time()},
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "run"))
    parser.add_argument("--revision", default=os.environ.get("GLEIPNIR_COMMIT"))
    args = parser.parse_args()
    if args.phase == "prepare":
        prepare()
    else:
        with (ROOT / "runner.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            execute(ROOT, args.revision)


if __name__ == "__main__":
    main()
