"""Prepare and execute one mixed-deception, monitoring-only-MIL experiment."""

import argparse
import fcntl
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from experiments.adapter_capacity_scaling.run_lambda import runtime_environment
from experiments.monitoring_duration.run import summarize
from experiments.monitoring_lr_sweep.prepare import (
    DEFAULT_ID_INPUT,
    validate_id_separation,
)
from experiments.monitoring_lr_sweep.run_lambda import gpu_training_environment
from experiments.monitoring_objective_ablation.run_lambda import run_serving_parity
from experiments.monitoring_subset_duration.run import (
    train,
)
from experiments.monitoring_subset_duration.run import (
    validate_completed as validate_mil,
)
from experiments.tool_trajectory_monitoring.benchmark_distilled_ood import (
    validate_config,
    validate_inputs,
    validate_jobs,
)
from experiments.tool_trajectory_monitoring.prepare_mixed_4b_distillation import (
    MIXED_SOFT_SHA256,
    MIXED_STUDENT_SHA256,
    verify_aligned_soft_rows,
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

ROOT = Path("results/monitoring_mil_mixture")
LOGS = Path("logs/lambda/monitoring_mil_mixture")


def identity(row: dict) -> tuple[str, str]:
    return str(row["dataset"]), str(row["index"])


def select_mixture(
    students: list[dict], targets: list[dict], selection: list[dict]
) -> tuple[list[dict], list[dict]]:
    """Preserve original aligned records, filtering only monitoring membership."""
    wanted = {identity(r): r for r in selection}
    if len(wanted) != len(selection):
        raise ValueError("duplicate monitoring selection")
    rows, soft, seen, monitoring_seen = [], [], set(), set()
    for student, target in zip(students, targets, strict=True):
        key = identity(student)
        if key != identity(target) or student["label"] != target["label"]:
            raise ValueError("student/teacher alignment drift")
        if key in seen:
            raise ValueError("duplicate source identity")
        seen.add(key)
        is_monitoring = key[0].startswith("tool_trajectory/")
        if is_monitoring and key not in wanted:
            continue
        if is_monitoring:
            if any(
                student.get(k) != wanted[key].get(k)
                for k in (
                    "label",
                    "trajectory_sha256",
                    "lineage_group",
                    "student_direct_tokens",
                )
            ):
                raise ValueError("selected monitoring provenance drift")
            monitoring_seen.add(key)
        rows.append({**student, "mil_enabled": is_monitoring})
        soft.append(target)
    if monitoring_seen != set(wanted):
        raise ValueError("missing selected monitoring parents")
    return rows, soft


def make_job(config: dict) -> dict:
    if (
        config["monitoring_rows"],
        config["deception_rows"],
        config["epochs"],
        config["learning_rate"],
        config["seed"],
        config["strict_ood_consulted"],
    ) != (1738, 13149, 3, 2e-5, 0, False):
        raise ValueError("authorized mixture design drift")
    source = next(
        j
        for j in read_jsonl(Path(config["source_jobs"]))
        if j["job_name"] == config["source_job"]
    )
    if (source["mil_loss_weight"], source["mil_pooling"]) != (0.25, "logmeanexp"):
        raise ValueError("selected MIL recipe drift")
    root, data = Path(config["result_dir"]), Path(config["data_dir"])
    name = "mil20-plus-deception-lr2em05-epochs3-seed0"
    systems = {}
    if "world_size" in config:
        world = int(config["world_size"])
        if world not in {1, 2}:
            raise ValueError("unsupported mixture world size")
        systems = {
            "world_size": world,
            "gradient_accumulation_steps": 32 // world,
            "nonreentrant_checkpointing": config["nonreentrant_checkpointing"],
            "selective_torch_compile_policy": config["selective_torch_compile_policy"],
        }
        name += f"-world{world}"
    output = root / "runs" / name
    return {
        **source,
        **systems,
        "job_name": name,
        "design_role": "mil_deception_mixture",
        "data_scope": "monitoring20_plus_all_deception",
        "train_rows": 14887,
        "learning_rate": float(config["learning_rate"]),
        "num_train_epochs": int(config["epochs"]),
        "seed": int(config["seed"]),
        "deception_rows": 13149,
        "monitoring_rows": 1738,
        "data_fraction": None,
        "selection_manifest": None,
        "selection_sha256": None,
        "expected_steps": 1398,
        "save_steps": 1398,
        "student_rows": str(data / "student_rows.jsonl"),
        "student_rows_sha256": sha256_file(data / "student_rows.jsonl"),
        "soft_targets": str(data / "soft_targets.jsonl"),
        "soft_targets_sha256": sha256_file(data / "soft_targets.jsonl"),
        "output_dir": str(output),
        "causal_adapter_dir": str(output / "causal_adapter"),
        "model_dir": str(output / "model"),
    }


def prepare() -> None:
    with initialize_config_dir(
        version_base=None, config_dir=str(Path(__file__).parent.resolve())
    ):
        config = OmegaConf.to_container(compose(config_name="config"), resolve=True)
    root, data, source = map(
        Path, (config["result_dir"], config["data_dir"], config["source_data_dir"])
    )
    if root.exists() or data.exists():
        raise FileExistsError("reuse frozen artifacts rather than preparing again")
    inputs = [source / "mixed_student_rows.jsonl", source / "mixed_soft_targets.jsonl"]
    for p, digest in zip(
        inputs, (MIXED_STUDENT_SHA256, MIXED_SOFT_SHA256), strict=True
    ):
        if sha256_file(p) != digest:
            raise ValueError(f"historical input drift: {p}")
    verify_aligned_soft_rows(*inputs)
    selection = read_jsonl(Path(config["monitoring_selection"]))
    rows, targets = select_mixture(
        read_jsonl(inputs[0]), read_jsonl(inputs[1]), selection
    )
    populations = Counter(r["mil_enabled"] for r in rows)
    if populations != {True: 1738, False: 13149}:
        raise ValueError(f"mixture population drift: {populations}")
    original = {identity(r): r for r in read_jsonl(source / "student_rows.jsonl")}
    for row in rows:
        if (
            row["mil_enabled"]
            and row["student_prompt"] != original[identity(row)]["student_prompt"]
        ):
            raise ValueError("monitoring direct prompt differs from baseline")
    heldout = validate_id_separation(
        DEFAULT_ID_INPUT, {r["trajectory_sha256"] for r in selection}
    )
    atomic_write_jsonl(data / "student_rows.jsonl", rows)
    atomic_write_jsonl(data / "soft_targets.jsonl", targets)
    longest_mon = sorted(
        (r for r in rows if r["mil_enabled"]),
        key=lambda r: r["student_direct_tokens"],
        reverse=True,
    )[:16]
    longest_dec = sorted(
        (r for r in rows if not r["mil_enabled"]),
        key=lambda r: len(r["student_prompt"].encode()),
        reverse=True,
    )[:16]
    preflight = [r for pair in zip(longest_mon, longest_dec, strict=True) for r in pair]
    atomic_write_jsonl(
        root / "preflight_selection.jsonl",
        [{k: r[k] for k in ("dataset", "index", "label")} for r in preflight],
    )
    job = make_job(config)
    atomic_write_jsonl(root / "jobs.jsonl", [job])
    evaluation = json.loads(Path(config["evaluation_template"]).read_text())
    evaluation.update(
        campaign_id=config["campaign_id"],
        hypothesis="Deception transfer into monitoring-only MIL",
    )
    evaluation["scope"]["selection_rule"] = (
        "One frozen final three-epoch endpoint; ID only, no OOD."
    )
    evaluation["model_groups"]["4b"].update(
        jobs=str(root / "jobs.jsonl"),
        jobs_sha256=sha256_file(root / "jobs.jsonl"),
        expected_jobs=[job["job_name"]],
        expected_target="kimi_soft_plus_auxiliary",
        parity_job=job["job_name"],
    )
    atomic_write_json(root / "id_benchmark.json", evaluation)
    atomic_write_json(root / "resolved_config.json", config)
    validate_config(evaluation)
    validate_inputs(evaluation)
    validate_jobs(evaluation, "4b")
    paths = [
        root / n
        for n in (
            "jobs.jsonl",
            "id_benchmark.json",
            "resolved_config.json",
            "preflight_selection.jsonl",
        )
    ]
    paths += [
        Path(config["baseline_result"]),
        Path(config["source_jobs"]),
        Path(config["monitoring_selection"]),
        Path("experiments/deception_distillation/train_student_sft.py"),
        Path("src/gleipnir/mil.py"),
        Path("experiments/monitoring_mil_mixture/run.py"),
        Path("experiments/monitoring_mil_mixture/config.yaml"),
    ]
    atomic_write_json(
        root / "manifest.json",
        {
            "source_artifacts": {str(p): sha256_file(p) for p in inputs},
            "held_out_id": heldout,
            "monitoring_rows": 1738,
            "deception_rows": 13149,
            "source_counts": dict(Counter(r["dataset"] for r in rows)),
            "files": {str(p): sha256_file(p) for p in paths},
        },
    )


def validate_completed(job: dict, *, preflight: bool = False) -> None:
    validate_mil(job, preflight=preflight)
    metadata = json.loads(
        (Path(job["causal_adapter_dir"]) / "training_metadata.json").read_text()
    )
    expected = {
        "enabled_rows": 16 if preflight else 1738,
        "disabled_rows": 16 if preflight else 13149,
        "reduction": "eligible_bag_sum_over_all_parents",
    }
    if metadata["mil_population"] != expected:
        raise ValueError("MIL eligibility population drift")
    if not math.isfinite(metadata["train_metrics"]["train_loss"]):
        raise ValueError("nonfinite loss")


def execute(root: Path = ROOT) -> None:
    logs = LOGS if root == ROOT else Path("logs/lambda") / root.name
    manifest = json.loads((root / "manifest.json").read_text())
    for path, digest in manifest["files"].items():
        if sha256_file(Path(path)) != digest:
            raise ValueError(f"frozen contract drift: {path}")
    config = json.loads((root / "resolved_config.json").read_text())
    (job,) = read_jsonl(root / "jobs.jsonl")
    if job != make_job(config):
        raise ValueError("job reconstruction drift")
    verify_job_inputs([job])
    if subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True
    ).strip():
        raise RuntimeError("GPUs not idle; do not interrupt unrelated work")
    logs.mkdir(parents=True, exist_ok=True)
    serving = runtime_environment("0")
    os.environ.update(serving)
    env = gpu_training_environment(ensure_qwen35_long_trajectory_kernels(), 0)
    world = int(job.get("world_size", 1))
    env["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, range(world)))
    if world > 1:
        env["OMP_NUM_THREADS"] = "1"
    status = CampaignStatus(
        root / "status.json", [job], os.environ.get("GLEIPNIR_COMMIT")
    )
    status.update(training_gpus=list(range(world)))
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
            )
        },
    )
    try:
        out = root / "preflight"
        selection = root / "preflight_selection.jsonl"
        preflight = {
            **job,
            "job_name": "mixed-mil-preflight",
            "train_rows": 32,
            "max_steps": 1,
            "num_train_epochs": -1,
            "save_steps": 1,
            "selection_manifest": str(selection),
            "selection_sha256": sha256_file(selection),
            "train_sampling_strategy": "sequential",
            "selective_torch_compile_canary_tokens": 2048,
            "output_dir": str(out),
            "causal_adapter_dir": str(out / "causal_adapter"),
            "model_dir": str(out / "model"),
        }
        atomic_write_jsonl(root / "preflight_jobs.jsonl", [preflight])
        status.update(phase="preflight", preflight="running")
        if validated := config.get("validated_preflight_jobs"):
            prior = next(
                j for j in read_jsonl(Path(validated)) if j["expected_steps"] == 1
            )
            validate_completed(prior, preflight=True)
        else:
            train(root / "preflight_jobs.jsonl", preflight, env, logs / "preflight.log")
            validate_completed(preflight, preflight=True)
        status.update(phase="training", preflight="passed")
        with status.job(job["job_name"], gpu=0):
            train(root / "jobs.jsonl", job, env, logs / "training.log")
            validate_completed(job)
        status.update(phase="serving_parity")
        run_serving_parity(root / "id_benchmark.json", root, env, serving)
        status.update(phase="id_evaluation")
        subprocess.run(
            [
                sys.executable,
                "-u",
                "-m",
                "experiments.tool_trajectory_monitoring.benchmark_distilled_ood",
                "--config",
                str(root / "id_benchmark.json"),
                "--model-size",
                "4b",
                "--output-root",
                str(root / "id_evaluation"),
            ],
            env=serving,
            check=True,
        )
        summarize(root, [job], config)
        status.update(state="complete", phase="complete", completed_at_unix=time.time())
    except BaseException as error:
        status.update(state="failed", error=repr(error))
        raise


def distributed_screen(attempt: int = 1, eager_ddp: bool = False) -> None:
    """Bounded matched screen; never automatically promote uninspected results."""
    suffix = "ddp_screen" if attempt == 1 else f"ddp_screen_attempt{attempt}"
    root = ROOT / suffix
    logs = LOGS / suffix
    if root.exists():
        raise FileExistsError("screen already exists; inspect before retrying")
    logs.mkdir(parents=True, exist_ok=True)
    (source,) = read_jsonl(ROOT / "jobs.jsonl")
    verify_job_inputs([source])
    if subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True
    ).strip():
        raise RuntimeError("GPUs not idle")
    rows = read_jsonl(Path(source["student_rows"]))
    stable = sorted(
        rows, key=lambda r: hashlib.sha256(str(identity(r)).encode()).hexdigest()
    )
    selected = [r for r in stable if r["mil_enabled"]][:30]
    selected += [r for r in stable if not r["mil_enabled"]][:226]
    selected.sort(key=lambda r: hashlib.sha256(str(identity(r)).encode()).hexdigest())
    selection = root / "selection.jsonl"
    atomic_write_jsonl(
        selection, [{k: r[k] for k in ("dataset", "index", "label")} for r in selected]
    )
    # Membership selection preserves source order; materialize screen order.
    screen_rows = root / "student_rows.jsonl"
    atomic_write_jsonl(screen_rows, selected)
    longest_keys = read_jsonl(ROOT / "preflight_selection.jsonl")
    by_id = {identity(r): r for r in rows}
    longest_rows = root / "longest_rows.jsonl"
    atomic_write_jsonl(longest_rows, [by_id[identity(r)] for r in longest_keys])
    jobs = []
    for name, world, longest in (
        ("single", 1, False),
        ("ddp_longest", 2, True),
        ("ddp", 2, False),
    ):
        output = root / name
        chosen = ROOT / "preflight_selection.jsonl" if longest else selection
        jobs.append(
            {
                **source,
                "job_name": f"mixed-mil-screen-{name}",
                "world_size": world,
                "student_rows": str(longest_rows if longest else screen_rows),
                "student_rows_sha256": sha256_file(
                    longest_rows if longest else screen_rows
                ),
                "gradient_accumulation_steps": 32 // world,
                "nonreentrant_checkpointing": True,
                "train_rows": 32 if longest else 256,
                "expected_steps": 1 if longest else 8,
                "max_steps": 1 if longest else 8,
                "num_train_epochs": -1,
                "save_steps": 1 if longest else 8,
                "selection_manifest": str(chosen),
                "selection_sha256": sha256_file(chosen),
                "train_sampling_strategy": "sequential",
                "selective_torch_compile_canary_tokens": 2048,
                "selective_torch_compile_policy": (
                    "none"
                    if eager_ddp and world == 2
                    else source["selective_torch_compile_policy"]
                ),
                "output_dir": str(output),
                "causal_adapter_dir": str(output / "causal_adapter"),
                "model_dir": str(output / "model"),
            }
        )
    atomic_write_jsonl(root / "jobs.jsonl", jobs)
    code = [
        Path(__file__),
        Path("experiments/deception_distillation/train_student_sft.py"),
        Path("src/gleipnir/distributed_training.py"),
        Path("src/gleipnir/mil.py"),
        Path("experiments/tool_trajectory_monitoring/run_distillation_train.py"),
    ]
    atomic_write_json(
        root / "manifest.json",
        {
            "source_manifest_sha256": sha256_file(ROOT / "manifest.json"),
            "intervention": (
                "DDP world2 vs world1; batch32; nonreentrant in both; "
                f"eager_ddp={eager_ddp}; materialized interleaved screen order"
            ),
            "selection_rule": (
                "finite, synchronized replicas, longest preflight, "
                ">1% steady speedup; inspect before promotion"
            ),
            "files": {
                str(p): sha256_file(p)
                for p in [
                    *code,
                    selection,
                    screen_rows,
                    longest_rows,
                    root / "jobs.jsonl",
                ]
            },
        },
    )
    os.environ.update(runtime_environment("0,1"))
    env = gpu_training_environment(ensure_qwen35_long_trajectory_kernels(), 0)
    env["OMP_NUM_THREADS"] = "1"
    status = CampaignStatus(
        root / "status.json", jobs, os.environ.get("GLEIPNIR_COMMIT")
    )
    for job in jobs:
        env["CUDA_VISIBLE_DEVICES"] = "0,1" if job["world_size"] == 2 else "0"
        with status.job(job["job_name"], gpu=list(range(job["world_size"]))):
            train(root / "jobs.jsonl", job, env, logs / f"{job['job_name']}.log")
            validate_mil(job, preflight=job["expected_steps"] == 1)
    status.update(state="complete", phase="awaiting_screen_analysis")


def promote_distributed_screen(screen: Path, root: Path) -> None:
    """Freeze a fresh full run only after the measured DDP screen passes."""
    if root.exists():
        raise FileExistsError("refuse to overwrite a frozen production run")
    if json.loads((screen / "status.json").read_text())["state"] != "complete":
        raise ValueError("screen is not complete")
    jobs = read_jsonl(screen / "jobs.jsonl")
    (single,) = [j for j in jobs if j["world_size"] == 1]
    (ddp,) = [j for j in jobs if j["world_size"] == 2 and j["expected_steps"] == 8]
    paths = []
    metadata = {}
    for candidate in jobs:
        validate_mil(candidate, preflight=candidate["expected_steps"] == 1)
        path = Path(candidate["causal_adapter_dir"]) / "training_metadata.json"
        paths.append(path)
        current = json.loads(path.read_text())
        population = current["mil_population"]
        expected = (16, 16) if candidate["expected_steps"] == 1 else (30, 226)
        if (population["enabled_rows"], population["disabled_rows"]) != expected:
            raise ValueError("screen population drift")
        if not math.isfinite(current["train_metrics"]["train_loss"]):
            raise ValueError("nonfinite screen loss")
        if (
            candidate["world_size"] == 2
            and current["distributed_training"].get("mil_projection_autocast")
            is not False
        ):
            raise ValueError("DDP selected-head precision not preserved")
        metadata[candidate["job_name"]] = current
    if single["student_rows_sha256"] != ddp["student_rows_sha256"]:
        raise ValueError("unmatched throughput samples")
    times = [
        metadata[j["job_name"]]["optimizer_step_timing"]["steady_mean_seconds"]
        for j in (single, ddp)
    ]
    speedup = times[0] / times[1]
    if not speedup > 1.01:
        raise ValueError(f"DDP did not clear the practical speed gate: {speedup:.3f}x")
    config = json.loads((ROOT / "resolved_config.json").read_text())
    config.update(
        result_dir=str(root),
        campaign_id=config["campaign_id"] + "-ddp2",
        world_size=2,
        nonreentrant_checkpointing=True,
        selective_torch_compile_policy=ddp["selective_torch_compile_policy"],
        validated_preflight_jobs=str(screen / "jobs.jsonl"),
        distributed_sampler={
            "even_batches": True,
            "padding_rows_per_epoch": 1,
            "effective_batch": 32,
            "final_update_parents": 8,
        },
    )
    job = make_job(config)
    verify_job_inputs([job])
    atomic_write_jsonl(root / "jobs.jsonl", [job])
    atomic_write_json(root / "resolved_config.json", config)
    atomic_write_jsonl(
        root / "preflight_selection.jsonl",
        read_jsonl(ROOT / "preflight_selection.jsonl"),
    )
    evaluation = json.loads((ROOT / "id_benchmark.json").read_text())
    evaluation["campaign_id"] = config["campaign_id"]
    evaluation["model_groups"]["4b"].update(
        jobs=str(root / "jobs.jsonl"),
        jobs_sha256=sha256_file(root / "jobs.jsonl"),
        expected_jobs=[job["job_name"]],
        parity_job=job["job_name"],
    )
    atomic_write_json(root / "id_benchmark.json", evaluation)
    validate_config(evaluation)
    validate_inputs(evaluation)
    validate_jobs(evaluation, "4b")
    paths += [screen / "jobs.jsonl", screen / "manifest.json", ROOT / "manifest.json"]
    paths += list(root.glob("*.json")) + [
        root / "jobs.jsonl",
        root / "preflight_selection.jsonl",
    ]
    paths += [
        Path(__file__),
        Path("experiments/deception_distillation/train_student_sft.py"),
        Path("experiments/tool_trajectory_monitoring/run_distillation_train.py"),
        Path("experiments/monitoring_lr_sweep/core.py"),
        Path("experiments/monitoring_subset_duration/run.py"),
        Path("src/gleipnir/distributed_training.py"),
        Path("src/gleipnir/mil.py"),
    ]
    atomic_write_json(
        root / "manifest.json",
        {
            "authorized_transition": (
                "single-GPU stopped before training; promote measured DDP"
            ),
            "screen": str(screen),
            "single_seconds_per_step": times[0],
            "ddp_seconds_per_step": times[1],
            "speedup": speedup,
            "files": {str(p): sha256_file(p) for p in paths},
        },
    )
    print(
        json.dumps(
            {
                "prepared": str(root),
                "speedup": speedup,
                "training_hours_estimate": 1398 * times[1] / 3600,
            }
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "phase", choices=("prepare", "run", "ddp-screen", "promote-ddp")
    )
    parser.add_argument("--screen", type=Path)
    parser.add_argument("--result-dir", type=Path, default=ROOT)
    parser.add_argument("--attempt", type=int, default=1)
    parser.add_argument("--eager-ddp", action="store_true")
    arguments = parser.parse_args()
    phase = arguments.phase
    if phase == "prepare":
        prepare()
    elif phase == "ddp-screen":
        distributed_screen(arguments.attempt, arguments.eager_ddp)
    elif phase == "promote-ddp":
        if arguments.screen is None or arguments.result_dir == ROOT:
            parser.error("promotion requires --screen and a fresh --result-dir")
        promote_distributed_screen(arguments.screen, arguments.result_dir)
    else:
        with (arguments.result_dir / "runner.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            execute(arguments.result_dir)
