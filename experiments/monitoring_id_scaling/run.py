"""Prepare a nested ID data curve and queue the shared two-GPU experiment runner."""

import argparse
import json
import math
import os
import selectors
import subprocess
import time
from collections import Counter
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from experiments.monitoring_duration.run import execute, validate_completed
from experiments.monitoring_lr_sweep.prepare import (
    DEFAULT_ID_INPUT,
    inspect_training_inputs,
    validate_id_separation,
)
from experiments.monitoring_lr_sweep.prepare import (
    make_jobs as make_lr_jobs,
)
from experiments.tool_trajectory_monitoring.benchmark_distilled_ood import (
    validate_config,
    validate_inputs,
    validate_jobs,
)
from gleipnir.monitoring_systems_screen import (
    atomic_write_json,
    atomic_write_jsonl,
    read_jsonl,
    sha256_file,
)
from gleipnir.nested_subsets import nested_subsets

ROOT = Path("results/monitoring_id_scaling")


def make_jobs(config: dict) -> list[dict]:
    """Keep the frozen one-epoch LR recipe; change only selections and endpoints."""
    if (
        config["epochs"] != 1
        or config["learning_rate"] != 2e-5
        or config["seed"] != 0
        or config["counts"] != [434, 869, 1738, 4344]
        or config["fractions"] != [0.05, 0.1, 0.2, 0.5]
        or config["strict_ood_consulted"] is not False
    ):
        raise ValueError("one-epoch ID scaling design drift")
    root = Path(config["result_dir"])
    base = next(
        j
        for j in make_lr_jobs(Path(config["data_dir"]), root)
        if j["learning_rate"] == 2e-5
    )
    jobs = []
    for fraction, count in zip(config["fractions"], config["counts"], strict=True):
        name = f"soft-pct{round(fraction * 100):03d}-lr2em05-epoch1-seed0"
        selection = root / "selections" / f"n{count}.jsonl"
        output = root / "runs" / name
        jobs.append(
            {
                **base,
                "job_name": name,
                "design_role": "nested_id_data_scaling",
                "train_rows": count,
                "data_fraction": fraction,
                "selection_manifest": str(selection),
                "selection_sha256": sha256_file(selection),
                "expected_steps": math.ceil(count / 32),
                "save_steps": math.ceil(count / 32),
                "completion_loss_weight": 0.0,
                "mil_loss_weight": 0.0,
                "prefix_loss_weight": 0.0,
                "output_dir": str(output),
                "causal_adapter_dir": str(output / "causal_adapter"),
                "model_dir": str(output / "model"),
            }
        )
    return jobs


def prepare() -> None:
    with initialize_config_dir(
        version_base=None, config_dir=str(Path(__file__).parent.resolve())
    ):
        config = OmegaConf.to_container(compose(config_name="config"), resolve=True)
    root = Path(config["result_dir"])
    if root.exists():
        raise FileExistsError(root)
    data = Path(config["data_dir"])
    audit, longest = inspect_training_inputs(
        data / "student_rows.jsonl", data / "soft_targets.jsonl"
    )
    heldout = validate_id_separation(DEFAULT_ID_INPUT, audit.pop("trajectory_hashes"))
    rows = read_jsonl(data / "student_rows.jsonl")
    subsets = nested_subsets(rows, config["counts"], config["seed"])
    selections = []
    for count, selected in subsets.items():
        path = root / "selections" / f"n{count}.jsonl"
        atomic_write_jsonl(
            path,
            [
                {
                    k: row[k]
                    for k in (
                        "dataset",
                        "index",
                        "label",
                        "lineage_group",
                        "student_direct_tokens",
                        "trajectory_sha256",
                    )
                }
                for row in selected
            ],
        )
        selections.append(
            {
                "rows": count,
                "fraction_actual": count / len(rows),
                "path": str(path),
                "sha256": sha256_file(path),
                "tokens": sum(r["student_direct_tokens"] for r in selected),
                "source_label_counts": dict(
                    Counter(f"{r['source_dataset']}:{r['label']}" for r in selected)
                ),
            }
        )
    atomic_write_jsonl(root / "selections/preflight-longest-32.jsonl", longest)
    jobs = make_jobs(config)
    atomic_write_jsonl(root / "jobs.jsonl", jobs)
    evaluation = json.loads(Path(config["evaluation_template"]).read_text())
    evaluation.update(
        campaign_id=config["campaign_id"], hypothesis="One-epoch nested ID data scaling"
    )
    evaluation["scope"]["selection_rule"] = (
        "Four frozen subset endpoints plus historical matched 100%; "
        "no intermediate selection or OOD."
    )
    evaluation["model_groups"]["4b"].update(
        jobs=str(root / "jobs.jsonl"),
        jobs_sha256=sha256_file(root / "jobs.jsonl"),
        expected_jobs=[j["job_name"] for j in jobs],
        parity_job=jobs[-1]["job_name"],
    )
    validate_config(evaluation)
    validate_inputs(evaluation)
    validate_jobs(evaluation, "4b")
    atomic_write_json(root / "id_benchmark.json", evaluation)
    atomic_write_json(root / "resolved_config.json", config)
    baseline = json.loads(Path(config["baseline_result"]).read_text())
    job = baseline["training_job"]
    if (
        baseline["rows"] != 3012
        or job["num_train_epochs"] != 1
        or job["learning_rate"] != 2e-5
        or job["train_rows"] != 8688
    ):
        raise ValueError("historical 100% point is not the matching one-epoch recipe")
    paths = (
        [
            root / name
            for name in (
                "jobs.jsonl",
                "id_benchmark.json",
                "resolved_config.json",
                "selections/preflight-longest-32.jsonl",
            )
        ]
        + [Path(s["path"]) for s in selections]
        + [Path(config["baseline_result"])]
    )
    paths.extend(
        [
            Path(__file__),
            Path(__file__).with_name("config.yaml"),
            Path("src/gleipnir/nested_subsets.py"),
            Path("experiments/monitoring_duration/run.py"),
        ]
    )
    atomic_write_json(
        root / "manifest.json",
        {
            "training": audit,
            "held_out_id": heldout,
            "selections": selections,
            "recorded_lineages": len({r["lineage_group"] for r in rows}),
            "files": {str(p): sha256_file(p) for p in paths},
        },
    )


def dependency_ready(status: dict) -> bool:
    """A failed/cancelled/dead predecessor is never permission to start."""
    return status.get("state") == "complete" and status.get("phase") == "complete"


def plot_curve(curve: list[dict], destination: Path) -> None:
    """Plot the endpoints without smoothing or uncertainty claims."""
    from gleipnir.plotting import plt, save_figure, set_plot_style

    set_plot_style()
    figure, axes = plt.subplots(1, 2, figsize=(14, 5))
    x = [100 * point["fraction"] for point in curve]
    axes[0].plot(
        x, [p["metrics"]["macro"]["pauroc_at_20"] for p in curve], "o-", label="Macro"
    )
    for source in ("gloom_exfiltration", "test_stride"):
        values = [
            next(
                g["pauroc_at_20"]
                for g in p["metrics"]["groups"]
                if g["group"] == source
            )
            for p in curve
        ]
        axes[0].plot(x, values, "o--", label=source)
    axes[0].set_ylabel("ID pAUROC@20 (higher is better)")
    axes[0].legend(fontsize=10)
    axes[1].plot(x, [p["metrics"]["macro"]["brier"] for p in curve], "o-")
    axes[1].set_ylabel("Macro Brier (lower is better)")
    for axis in axes:
        axis.set_xlabel("Training data (%)")
        axis.set_xticks(x)
    figure.suptitle("One epoch, LR 2e-5, seed 0; historical 100% endpoint")
    save_figure(figure, destination)


def write_curve(root: Path, config: dict) -> None:
    # Include the historical 100% result explicitly in the curve artifact.
    summary = json.loads((root / "summary.json").read_text())
    jobs = make_jobs(config)
    curve = [
        {
            "fraction": j["data_fraction"],
            "train_rows": j["train_rows"],
            "expected_steps": j["expected_steps"],
            **r,
        }
        for j, r in zip(jobs, summary["rows"], strict=True)
    ]
    curve.append(
        {
            "fraction": 1.0,
            "train_rows": 8688,
            "expected_steps": 272,
            "historical": True,
            "result": config["baseline_result"],
            "metrics": summary["baseline"],
        }
    )
    selections = json.loads((root / "manifest.json").read_text())["selections"]
    for point, job in zip(curve[:-1], jobs, strict=True):
        result_path = (
            root / "id_evaluation/4b/adapters" / job["job_name"] / "result.json"
        )
        metadata = json.loads(result_path.read_text())["training_metadata"]
        point["train_runtime_seconds"] = metadata["train_metrics"]["train_runtime"]
        point["training_tokens_per_epoch"] = next(
            s["tokens"] for s in selections if s["rows"] == job["train_rows"]
        )
    baseline = json.loads(Path(config["baseline_result"]).read_text())[
        "training_metadata"
    ]
    curve[-1]["train_runtime_seconds"] = baseline["train_metrics"]["train_runtime"]
    curve[-1]["training_tokens_per_epoch"] = baseline["batching"]["padding"][
        "direct_tokens"
    ]
    atomic_write_json(
        root / "scaling_curve.json",
        {"epochs": 1, "seed": 0, "points": curve, "strict_ood_consulted": False},
    )
    plot_curve(curve, root / "scaling_curve.svg")


def wait_and_run(root: Path, revision: str | None) -> None:
    config = json.loads((root / "resolved_config.json").read_text())
    predecessor = config["predecessor"]
    status_path = Path(predecessor["root"]) / "status.json"
    queue = {"state": "waiting", "predecessor": predecessor, "queued_at": time.time()}
    atomic_write_json(root / "queue_status.json", queue)
    try:
        if not dependency_ready(json.loads(status_path.read_text())):
            pid = predecessor["pid"]
            handle = os.pidfd_open(pid)
            try:
                command = (
                    Path(f"/proc/{pid}/cmdline")
                    .read_bytes()
                    .decode()
                    .replace("\0", " ")
                )
                if (
                    predecessor["command_contains"] not in command
                    or predecessor["root"] not in command
                ):
                    raise ValueError("predecessor process identity drift")
                print("Waiting for predecessor pipeline exit", pid, flush=True)
                with selectors.DefaultSelector() as selector:
                    selector.register(handle, selectors.EVENT_READ)
                    selector.select()
            finally:
                os.close(handle)
        if not dependency_ready(json.loads(status_path.read_text())):
            raise RuntimeError(
                "predecessor did not complete successfully; scaling not launched"
            )
        processes = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-compute-apps=pid",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        ).strip()
        if processes:
            raise RuntimeError(f"GPUs still have compute processes: {processes}")
        queue.update(state="running", started_at=time.time())
        atomic_write_json(root / "queue_status.json", queue)
        execute(
            root,
            revision,
            job_factory=make_jobs,
            completed_validator=validate_completed,
            logs_root=Path("logs/lambda/monitoring_id_scaling"),
        )
        write_curve(root, config)
        queue.update(state="complete", completed_at=time.time())
    except BaseException as error:
        queue.update(state="failed", error=repr(error))
        raise
    finally:
        atomic_write_json(root / "queue_status.json", queue)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "queue"))
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--revision", default=os.environ.get("GLEIPNIR_COMMIT"))
    args = parser.parse_args()
    if args.phase == "prepare":
        prepare()
    else:
        wait_and_run(args.root, args.revision)


if __name__ == "__main__":
    main()
