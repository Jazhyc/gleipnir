"""Config-driven matched systems screens for long-trajectory monitoring.

This module owns the repeated preparation, validation, preflight, parallel
execution, status, and throughput-summary machinery used by systems-only
ablations. Individual hypotheses remain auditable JSON contracts and READMEs;
they do not need four nearly identical Python entrypoints.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
SCHEMA_VERSION = 1


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of one file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read non-empty JSONL records."""
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def atomic_write_json(path: Path, value: Any) -> None:
    """Atomically write formatted JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def atomic_write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    """Atomically write JSONL records."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    temporary.replace(path)


def stable_stratified_selection(
    records: list[dict[str, Any]], count: int = 640, seed: int = 0
) -> list[dict[str, Any]]:
    """Select a stable proportional sample across dataset/label strata."""
    if not 0 < count <= len(records):
        raise ValueError("selection count must be positive and no larger than input")
    strata: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        strata[(str(record["dataset"]), int(record["label"]))].append(record)
    exact = {key: count * len(rows) / len(records) for key, rows in strata.items()}
    quotas = {key: math.floor(value) for key, value in exact.items()}
    remainder = count - sum(quotas.values())
    order = sorted(strata, key=lambda key: (-(exact[key] - quotas[key]), key))
    for key in order[:remainder]:
        quotas[key] += 1
    chosen: set[tuple[str, Any]] = set()
    for key, rows in strata.items():
        ranked = sorted(
            rows,
            key=lambda row: hashlib.sha256(
                f"{seed}\0{row['dataset']}\0{row['index']}".encode()
            ).digest(),
        )
        chosen.update(
            (str(row["dataset"]), row["index"]) for row in ranked[: quotas[key]]
        )
    selected = [
        row
        for row in records
        if (str(row["dataset"]), row["index"]) in chosen
    ]
    if len(selected) != count:
        raise AssertionError(f"selected {len(selected)} rows instead of {count}")
    return selected


def selection_manifest_row(row: dict[str, Any]) -> dict[str, Any]:
    """Retain only identity, provenance, label, and length in a selection."""
    return {
        key: row[key]
        for key in (
            "dataset",
            "index",
            "label",
            "source_dataset",
            "student_direct_tokens",
            "trajectory_sha256",
        )
    }


@dataclass(frozen=True)
class ScreenPaths:
    """Resolved inputs and artifact paths for one screen invocation."""

    data_dir: Path
    result_dir: Path
    student_rows: Path
    soft_targets: Path
    selection: Path
    preflight_selection: Path
    jobs: Path
    status: Path


def load_config(path: Path) -> dict[str, Any]:
    """Load and validate the structural parts of a systems-screen contract."""
    config = json.loads(path.read_text())
    if config.get("systems_screen_schema") != SCHEMA_VERSION:
        raise ValueError("unsupported systems-screen schema")
    conditions = config.get("conditions")
    if not isinstance(conditions, list) or len(conditions) < 2:
        raise ValueError("a matched screen requires at least two conditions")
    names = [str(condition.get("job_name", "")) for condition in conditions]
    if not all(names) or len(names) != len(set(names)):
        raise ValueError("condition names must be present and unique")
    comparison = config.get("comparison", {})
    if (
        comparison.get("control") not in names
        or comparison.get("candidate") not in names
    ):
        raise ValueError("comparison jobs must name configured conditions")
    preflight = config.get("preflight", {})
    if preflight.get("condition") not in names:
        raise ValueError("preflight condition must name a configured condition")
    if int(config.get("gpus", 0)) < 1:
        raise ValueError("systems screen requires at least one GPU")
    return config


def resolve_paths(
    config: dict[str, Any],
    *,
    data_dir: Path | None = None,
    result_dir: Path | None = None,
) -> ScreenPaths:
    """Resolve config-relative data and artifact paths against the repository."""
    data = config["data"]
    artifacts = config["artifacts"]
    resolved_data = (data_dir or ROOT / data["directory"]).resolve()
    resolved_result = (result_dir or ROOT / artifacts["result_dir"]).resolve()
    return ScreenPaths(
        data_dir=resolved_data,
        result_dir=resolved_result,
        student_rows=resolved_data / data["student_rows"],
        soft_targets=resolved_data / data["soft_targets"],
        selection=resolved_result / "selections" / "matched.jsonl",
        preflight_selection=(
            resolved_result / "selections" / "preflight-longest.jsonl"
        ),
        jobs=resolved_result / "jobs.jsonl",
        status=resolved_result / "status.json",
    )


def make_jobs(
    config: dict[str, Any], paths: ScreenPaths, selection_sha256: str
) -> list[dict[str, Any]]:
    """Materialize exact jobs from a shared recipe and condition overrides."""
    jobs = make_jobs_unchecked(config, paths, selection_sha256)
    validate_jobs(config, jobs, paths, selection_sha256)
    return jobs


def validate_jobs(
    config: dict[str, Any],
    jobs: list[dict[str, Any]],
    paths: ScreenPaths,
    selection_sha256: str,
) -> None:
    """Reject any job-manifest drift from the declarative contract."""
    expected = make_jobs_unchecked(config, paths, selection_sha256)
    if jobs != expected:
        raise ValueError("job manifest differs from the systems-screen contract")


def make_jobs_unchecked(
    config: dict[str, Any], paths: ScreenPaths, selection_sha256: str
) -> list[dict[str, Any]]:
    """Build expected jobs without recursively validating them."""
    data = config["data"]
    base = {
        **config["recipe"],
        "train_rows": int(config["selection"]["rows"]),
        "selection_manifest": paths.selection.as_posix(),
        "selection_sha256": selection_sha256,
        "student_rows": paths.student_rows.as_posix(),
        "soft_targets": paths.soft_targets.as_posix(),
        "student_rows_sha256": data["student_rows_sha256"],
        "soft_targets_sha256": data["soft_targets_sha256"],
    }
    jobs = []
    for condition in config["conditions"]:
        name = str(condition["job_name"])
        output = paths.result_dir / "runs" / name
        jobs.append(
            {
                **base,
                **condition.get("overrides", {}),
                "job_name": name,
                "design_role": condition["design_role"],
                "output_dir": output.as_posix(),
                "causal_adapter_dir": (output / "causal_adapter").as_posix(),
                "model_dir": (output / "model").as_posix(),
            }
        )
    return jobs


def prepare_screen(
    config_path: Path,
    *,
    data_dir: Path | None = None,
    result_dir: Path | None = None,
) -> dict[str, Any]:
    """Audit inputs and write selections, jobs, and a reconstruction manifest."""
    config = load_config(config_path)
    paths = resolve_paths(config, data_dir=data_dir, result_dir=result_dir)
    data = config["data"]
    if sha256_file(paths.student_rows) != data["student_rows_sha256"]:
        raise ValueError("student-row checksum drifted")
    if sha256_file(paths.soft_targets) != data["soft_targets_sha256"]:
        raise ValueError("soft-target checksum drifted")
    records = read_jsonl(paths.student_rows)
    if len(records) != int(data["rows"]):
        raise ValueError("student-row count drifted")
    selection_config = config["selection"]
    selected = stable_stratified_selection(
        records, int(selection_config["rows"]), int(selection_config["seed"])
    )
    longest = sorted(
        records,
        key=lambda row: (int(row["student_direct_tokens"]), str(row["index"])),
        reverse=True,
    )[: int(selection_config["preflight_rows"])]
    atomic_write_jsonl(
        paths.selection, [selection_manifest_row(row) for row in selected]
    )
    atomic_write_jsonl(
        paths.preflight_selection,
        [selection_manifest_row(row) for row in longest],
    )
    selection_sha = sha256_file(paths.selection)
    jobs = make_jobs(config, paths, selection_sha)
    atomic_write_jsonl(paths.jobs, jobs)
    manifest = {
        "campaign_id": config["campaign_id"],
        "config": config_path.resolve().as_posix(),
        "config_sha256": sha256_file(config_path),
        "jobs": [job["job_name"] for job in jobs],
        "jobs_sha256": sha256_file(paths.jobs),
        "selection_sha256": selection_sha,
        "preflight_selection_sha256": sha256_file(paths.preflight_selection),
    }
    atomic_write_json(paths.result_dir / "manifest.json", manifest)
    return manifest


def validate_prepared_artifacts(
    config_path: Path, config: dict[str, Any], paths: ScreenPaths
) -> dict[str, Any]:
    """Verify every prepared input and manifest identity before GPU work."""
    data = config["data"]
    if sha256_file(paths.student_rows) != data["student_rows_sha256"]:
        raise ValueError("student-row checksum drifted")
    if sha256_file(paths.soft_targets) != data["soft_targets_sha256"]:
        raise ValueError("soft-target checksum drifted")
    manifest = json.loads((paths.result_dir / "manifest.json").read_text())
    expected = {
        "campaign_id": config["campaign_id"],
        "config_sha256": sha256_file(config_path),
        "jobs_sha256": sha256_file(paths.jobs),
        "selection_sha256": sha256_file(paths.selection),
        "preflight_selection_sha256": sha256_file(paths.preflight_selection),
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise ValueError(f"prepared artifact drift for {key}")
    return manifest


def nested_value(value: dict[str, Any], dotted_path: str) -> Any:
    """Read one dotted mapping path, returning a sentinel-like missing value."""
    current: Any = value
    for component in dotted_path.split("."):
        if not isinstance(current, dict) or component not in current:
            return _MISSING
        current = current[component]
    return current


_MISSING = object()


def condition_by_name(config: dict[str, Any], name: str) -> dict[str, Any]:
    """Return one configured condition by stable job name."""
    return next(
        condition for condition in config["conditions"] if condition["job_name"] == name
    )


def validate_training_metadata(
    metadata: dict[str, Any],
    config: dict[str, Any],
    job: dict[str, Any],
    *,
    expected_steps: int,
    require_canary: bool = False,
) -> None:
    """Validate common kernels, batch, explicit expectations, and completion."""
    expected_batch = {
        "micro_batch_size": job["micro_batch_size"],
        "gradient_accumulation_steps": job["gradient_accumulation_steps"],
        "effective_batch_size": job["effective_batch_size"],
    }
    if metadata.get("training_batch") != expected_batch:
        raise ValueError("training batch metadata drifted")
    kernels = config["kernels"]
    fla = metadata.get("flash_linear_attention", {})
    causal = metadata.get("causal_conv1d", {})
    if (
        fla.get("available") is not True
        or fla.get("required") is not True
        or fla.get("version") != kernels["flash_linear_attention"]
        or fla.get("backend_dispatch_disabled") is not True
        or fla.get("triton_version") != kernels["triton"]
        or not metadata.get("gated_delta_kernel_modules")
        or causal.get("available") is not True
        or causal.get("required") is not True
        or causal.get("version") != kernels["causal_conv1d"]
        or not causal.get("kernel_modules")
    ):
        raise ValueError("fast-kernel metadata is incomplete")
    condition = condition_by_name(config, str(job["job_name"]))
    expectations = {
        **config.get("metadata_expectations", {}),
        **condition.get("metadata_expectations", {}),
    }
    for path, expected in expectations.items():
        actual = nested_value(metadata, path)
        if actual is _MISSING or actual != expected:
            raise ValueError(
                f"training metadata drift for {path}: {actual!r} != {expected!r}"
            )
    completed = int(metadata.get("training_state", {}).get("global_step", -1))
    if completed != expected_steps:
        raise ValueError(f"completed {completed} steps instead of {expected_steps}")
    timing = metadata.get("optimizer_step_timing", {})
    if int(timing.get("recorded_steps", -1)) != expected_steps:
        raise ValueError("optimizer-step timing is incomplete")
    rate = float(
        metadata.get("train_metrics", {}).get("train_samples_per_second", math.nan)
    )
    if not math.isfinite(rate) or rate <= 0:
        raise ValueError("invalid training throughput")
    compiled = metadata.get("selective_torch_compile", {})
    unique_graphs = int(
        compiled.get("dynamo_counters", {}).get("stats", {}).get("unique_graphs", 0)
    )
    if not 1 <= unique_graphs <= int(config["maximum_unique_graphs"]):
        raise ValueError(f"unexpected Dynamo graph count: {unique_graphs}")
    canary = compiled.get("canary") or {}
    if require_canary and canary.get("passed") is not True:
        raise ValueError("same-weights compile canary did not pass")


def summarize_screen(
    config: dict[str, Any], jobs: list[dict[str, Any]]
) -> dict[str, Any]:
    """Validate conditions, extract common metrics, and apply the gain gate."""
    conditions: dict[str, dict[str, Any]] = {}
    expected_steps = int(config["recipe"]["max_steps"])
    for job in jobs:
        metadata = job["training_metadata"]
        validate_training_metadata(
            metadata, config, job, expected_steps=expected_steps
        )
        timing = metadata["optimizer_step_timing"]
        conditions[job["job_name"]] = {
            "samples_per_second": float(
                metadata["train_metrics"]["train_samples_per_second"]
            ),
            "train_runtime_seconds": float(metadata["train_metrics"]["train_runtime"]),
            "steady_mean_step_seconds": float(timing["steady_mean_seconds"]),
            "peak_cuda_memory_gib": float(
                metadata["peak_cuda_memory_allocated_bytes"]
            )
            / 2**30,
            "unique_graphs": int(
                metadata["selective_torch_compile"]["dynamo_counters"]["stats"][
                    "unique_graphs"
                ]
            ),
        }
    comparison = config["comparison"]
    control_name = comparison["control"]
    candidate_name = comparison["candidate"]
    improvement = (
        conditions[candidate_name]["samples_per_second"]
        / conditions[control_name]["samples_per_second"]
        - 1.0
    )
    selected_name = (
        candidate_name
        if improvement >= float(comparison["minimum_relative_improvement"])
        else control_name
    )
    selected_job = next(job for job in jobs if job["job_name"] == selected_name)
    return {
        "campaign_id": config["campaign_id"],
        "conditions": conditions,
        "relative_throughput_improvement": improvement,
        "selected_job": selected_name,
        "selected_value": selected_job[comparison["selection_job_field"]],
    }


class CampaignStatus:
    """Thread-safe atomic status for a matched systems screen."""

    def __init__(self, path: Path, jobs: list[dict[str, Any]], revision: str | None):
        self.path = path
        self.lock = threading.Lock()
        self.value: dict[str, Any] = {
            "state": "running",
            "phase": "initializing",
            "started_at_unix": time.time(),
            "revision": revision,
            "planned_jobs": [job["job_name"] for job in jobs],
            "active_jobs": [],
            "completed_jobs": [],
            "preflight": "pending",
        }
        self.update()

    def update(self, **values: Any) -> None:
        with self.lock:
            self.value.update(values)
            atomic_write_json(self.path, self.value)

    def start(self, name: str) -> None:
        with self.lock:
            self.value["active_jobs"].append(name)
            atomic_write_json(self.path, self.value)

    def finish(self, name: str) -> None:
        with self.lock:
            self.value["active_jobs"].remove(name)
            self.value["completed_jobs"].append(name)
            atomic_write_json(self.path, self.value)


def gpu_environment(base: dict[str, str], gpu: int, cache: Path) -> dict[str, str]:
    """Return an isolated per-condition kernel and compiler environment."""
    environment = dict(base)
    environment.update(
        CUDA_VISIBLE_DEVICES=str(gpu),
        FLA_DISABLE_BACKEND_DISPATCH="1",
        TILELANG_CACHE_DIR=(cache / "tilelang").as_posix(),
        TORCHINDUCTOR_CACHE_DIR=(cache / "torchinductor").as_posix(),
        TRITON_CACHE_DIR=(cache / "triton").as_posix(),
        TVM_CACHE_DIR=(cache / "tvm").as_posix(),
        TORCH_LOGS="recompiles",
    )
    return environment


def round_robin_lanes(
    jobs: list[dict[str, Any]], gpu_count: int
) -> list[list[dict[str, Any]]]:
    """Assign jobs to stable serial lanes without concurrent GPU reuse."""
    if gpu_count < 1:
        raise ValueError("GPU count must be positive")
    lanes: list[list[dict[str, Any]]] = [[] for _ in range(gpu_count)]
    for index, job in enumerate(jobs):
        lanes[index % gpu_count].append(job)
    return lanes


def run_training_job(
    jobs_path: Path,
    name: str,
    environment: dict[str, str],
    *,
    preflight: bool = False,
) -> None:
    """Invoke the existing distillation worker through its stable CLI."""
    command = [
        sys.executable,
        "-m",
        "experiments.tool_trajectory_monitoring.run_distillation_train",
        "--jobs",
        jobs_path.as_posix(),
        "--job-name",
        name,
        "--allow-preflight-job" if preflight else "--allow-non-scaling-job",
    ]
    subprocess.run(command, cwd=ROOT, env=environment, check=True)


def run_screen(config_path: Path, *, revision: str | None = None) -> None:
    """Preflight, execute conditions in parallel lanes, and summarize."""
    from gleipnir.qwen35_fast_training import (
        DEFAULT_CAUSAL_CONV1D_TARGET,
        DEFAULT_FLA_TARGET,
        DEFAULT_TRITON_TARGET,
        ensure_qwen35_long_trajectory_kernels,
    )

    config = load_config(config_path)
    paths = resolve_paths(config)
    validate_prepared_artifacts(config_path, config, paths)
    selection_sha = sha256_file(paths.selection)
    jobs = read_jsonl(paths.jobs)
    validate_jobs(config, jobs, paths, selection_sha)
    for job in jobs:
        for field in ("student_rows", "soft_targets"):
            if not Path(job[field]).is_file():
                raise FileNotFoundError(job[field])
    status = CampaignStatus(paths.status, jobs, revision)
    try:
        status.update(phase="kernel_preflight")
        environment = ensure_qwen35_long_trajectory_kernels(
            DEFAULT_FLA_TARGET, DEFAULT_CAUSAL_CONV1D_TARGET, DEFAULT_TRITON_TARGET
        )
        if revision:
            environment["GLEIPNIR_COMMIT"] = revision
        preflight_config = config["preflight"]
        selected = next(
            job for job in jobs if job["job_name"] == preflight_config["condition"]
        )
        preflight_dir = paths.result_dir / "preflight" / selected["job_name"]
        preflight = {
            **selected,
            "job_name": f"preflight-{selected['job_name']}",
            "design_role": "longest_trajectory_systems_preflight",
            "train_rows": int(config["selection"]["preflight_rows"]),
            "max_steps": int(preflight_config["max_steps"]),
            "selection_manifest": paths.preflight_selection.as_posix(),
            "selection_sha256": sha256_file(paths.preflight_selection),
            "save_steps": int(preflight_config["max_steps"]),
            "selective_torch_compile_canary_tokens": int(
                preflight_config["same_weights_logit_canary_tokens"]
            ),
            "output_dir": preflight_dir.as_posix(),
            "causal_adapter_dir": (preflight_dir / "causal_adapter").as_posix(),
            "model_dir": (preflight_dir / "model").as_posix(),
        }
        preflight_jobs = paths.result_dir / "preflight_jobs.jsonl"
        atomic_write_jsonl(preflight_jobs, [preflight])
        status.update(phase="longest_trajectory_preflight", preflight="running")
        run_training_job(
            preflight_jobs,
            preflight["job_name"],
            gpu_environment(
                environment, 0, paths.result_dir / "compile_cache" / "preflight"
            ),
            preflight=True,
        )
        metadata = json.loads(
            (preflight_dir / "causal_adapter" / "training_metadata.json").read_text()
        )
        validation_job = {**selected, "job_name": selected["job_name"]}
        validate_training_metadata(
            metadata,
            config,
            validation_job,
            expected_steps=int(preflight_config["max_steps"]),
            require_canary=True,
        )
        status.update(phase="benchmarking", preflight="passed")

        def run_condition(index: int, job: dict[str, Any]) -> dict[str, Any]:
            name = str(job["job_name"])
            status.start(name)
            run_training_job(
                paths.jobs,
                name,
                gpu_environment(
                    environment,
                    index % int(config["gpus"]),
                    paths.result_dir / "compile_cache" / name,
                ),
            )
            condition_metadata = json.loads(
                (Path(job["causal_adapter_dir"]) / "training_metadata.json").read_text()
            )
            validate_training_metadata(
                condition_metadata,
                config,
                job,
                expected_steps=int(config["recipe"]["max_steps"]),
            )
            status.finish(name)
            return {**job, "training_metadata": condition_metadata}

        gpu_count = int(config["gpus"])
        lanes = round_robin_lanes(jobs, gpu_count)

        def run_lane(gpu: int, lane: list[dict[str, Any]]) -> list[dict[str, Any]]:
            return [run_condition(gpu, job) for job in lane]

        with ThreadPoolExecutor(max_workers=gpu_count) as executor:
            futures = [
                executor.submit(run_lane, gpu, lane)
                for gpu, lane in enumerate(lanes)
                if lane
            ]
            by_name = {
                job["job_name"]: job
                for future in futures
                for job in future.result()
            }
        completed = [by_name[job["job_name"]] for job in jobs]
        status.update(phase="summarizing", active_jobs=[])
        summary = summarize_screen(config, completed)
        atomic_write_json(paths.result_dir / "summary.json", summary)
        status.update(state="complete", phase="complete", completed_at_unix=time.time())
    except BaseException as error:
        status.update(
            state="failed",
            phase="failed",
            active_jobs=[],
            error=repr(error),
            failed_at_unix=time.time(),
        )
        raise


def parse_args() -> argparse.Namespace:
    """Parse the shared prepare/run/summarize command."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "run", "summarize"))
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--result-dir", type=Path)
    parser.add_argument("--revision", default=os.environ.get("GLEIPNIR_COMMIT"))
    return parser.parse_args()


def main() -> None:
    """Run a config-driven systems-screen action."""
    args = parse_args()
    if args.action == "prepare":
        print(
            json.dumps(
                prepare_screen(
                    args.config,
                    data_dir=args.data_dir,
                    result_dir=args.result_dir,
                ),
                indent=2,
                sort_keys=True,
            )
        )
    elif args.action == "run":
        if args.data_dir is not None or args.result_dir is not None:
            raise ValueError("run uses the frozen config paths")
        run_screen(args.config, revision=args.revision)
    else:
        config = load_config(args.config)
        paths = resolve_paths(
            config, data_dir=args.data_dir, result_dir=args.result_dir
        )
        selection_sha = sha256_file(paths.selection)
        jobs = read_jsonl(paths.jobs)
        validate_jobs(config, jobs, paths, selection_sha)
        enriched = []
        for job in jobs:
            metadata = json.loads(
                (Path(job["causal_adapter_dir"]) / "training_metadata.json").read_text()
            )
            enriched.append({**job, "training_metadata": metadata})
        summary = summarize_screen(config, enriched)
        atomic_write_json(paths.result_dir / "summary.json", summary)
        print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
