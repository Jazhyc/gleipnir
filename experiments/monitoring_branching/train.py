"""Freeze and run the authorized all-prefix experiment, with final ID evaluation."""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from gleipnir.monitoring_systems_screen import atomic_write_json, sha256_file


def prepare() -> Path:
    """Freeze provenance and successful hard gates before allocating the model."""
    with initialize_config_dir(
        version_base=None, config_dir=str(Path(__file__).parent.resolve())
    ):
        config = OmegaConf.to_container(compose(config_name="training"), resolve=True)
    root = Path(config["result_dir"])
    if root.exists():
        raise FileExistsError(root)
    exception = config["numerical_exception"]
    if (
        sha256_file(Path(exception["failed_diagnostic"]))
        != exception["failed_diagnostic_sha256"]
    ):
        raise ValueError("authorized numerical diagnostic drift")
    original = Path("results/monitoring_branching/hybrid_preflight_v1")
    remaining = Path("results/monitoring_branching/hybrid_preflight_authorized_v2")
    gates = [original / f"parity-{p}.json" for p in (0, 487)] + [
        remaining / name
        for name in (
            "parity-2608.json",
            "parity-132.json",
            "memory-8142.json",
            "memory-7706.json",
        )
    ]
    for path in gates:
        result = json.loads(path.read_text())
        if not result["passed"] or any(
            result.get(k) != v
            for k, v in {
                "alignment": 64,
                "checkpoint_segments": True,
                "fp32_head": True,
                "independent_endpoint": True,
                "two_gpu_model_parallel": True,
            }.items()
        ):
            raise ValueError(f"unapproved hard gate failure or recipe drift: {path}")
    from experiments.monitoring_lr_sweep.prepare import (
        DEFAULT_DATA_DIR,
        DEFAULT_ID_INPUT,
        inspect_training_inputs,
        validate_id_separation,
    )

    audit, _ = inspect_training_inputs(
        DEFAULT_DATA_DIR / "student_rows.jsonl", DEFAULT_DATA_DIR / "soft_targets.jsonl"
    )
    holdout = validate_id_separation(DEFAULT_ID_INPUT, audit.pop("trajectory_hashes"))
    prior = Path("results/monitoring_prefix_training/manifest.json")
    prior_manifest = json.loads(prior.read_text())
    if (
        prior_manifest["paired_training"]["numerical_exception"]
        != config["cache_exception"]
    ):
        raise ValueError("cache authorization drift")
    # Reuse the frozen model/data/evaluation descriptor, not its sampled inputs.
    jobs_path = Path("results/monitoring_prefix_training/jobs.jsonl")
    if sha256_file(jobs_path) != prior_manifest["files"][str(jobs_path)]:
        raise ValueError("prior jobs drift")
    job = json.loads(jobs_path.read_text().splitlines()[0])
    output = root / "runs" / config["job_name"]
    job.update(
        job_name=config["job_name"],
        output_dir=str(output),
        causal_adapter_dir=str(output / "causal_adapter"),
        model_dir=str(output / "model"),
        student_rows=str(DEFAULT_DATA_DIR / "student_rows.jsonl"),
        student_rows_sha256=sha256_file(DEFAULT_DATA_DIR / "student_rows.jsonl"),
        prefix_loss_weight=0.1,
        prefix_sampling="all",
        torch_compile=False,
        selective_torch_compile_policy="none",
        gradient_checkpointing_policy="functional_branch_segments_and_full_endpoint_layers",
        gradient_accumulation_steps=32,
        micro_batch_size=1,
        save_steps=32,
        training_implementation="gleipnir.branch_trainer",
        effective_batch_size=32,
        parallelism="two_gpu_layer_split",
        resolved_branch_config=str(root / "resolved_config.json"),
    )
    root.mkdir(parents=True)
    (root / "jobs.jsonl").write_text(json.dumps(job) + "\n")
    evaluation = json.loads(Path(config["evaluation_template"]).read_text())
    evaluation.update(
        campaign_id=config["campaign_id"],
        hypothesis="All versus one sampled prefix per parent",
    )
    evaluation["scope"]["selection_rule"] = (
        "One final ID endpoint; compare full-only and sampled w0.1; "
        "no OOD or intermediate selection."
    )
    evaluation["model_groups"]["4b"].update(
        jobs=str(root / "jobs.jsonl"),
        jobs_sha256=sha256_file(root / "jobs.jsonl"),
        expected_jobs=[job["job_name"]],
        parity_job=job["job_name"],
    )
    atomic_write_json(root / "id_benchmark.json", evaluation)
    atomic_write_json(root / "resolved_config.json", config)
    paths = gates + [
        Path(exception["failed_diagnostic"]),
        prior,
        jobs_path,
        root / "resolved_config.json",
        root / "jobs.jsonl",
        root / "id_benchmark.json",
        Path(__file__),
        Path(__file__).with_name("training.yaml"),
    ]
    paths.extend(
        Path("src/gleipnir") / name
        for name in (
            "branch_training.py",
            "branch_trainer.py",
            "branch_model.py",
            "branch_data.py",
            "prefix_loss.py",
        )
    )
    atomic_write_json(
        root / "manifest.json",
        {
            "files": {str(p): sha256_file(p) for p in paths},
            "training": audit,
            "heldout": holdout,
            "prefix_provenance": prior_manifest["paired_training"],
            "numerical_exception": exception,
            "cache_exception": config["cache_exception"],
        },
    )
    return root


def verify(root: Path) -> dict:
    manifest = json.loads((root / "manifest.json").read_text())
    for path, checksum in manifest["files"].items():
        if sha256_file(Path(path)) != checksum:
            raise ValueError(f"frozen campaign drift: {path}")
    return json.loads((root / "resolved_config.json").read_text())


def training(root: Path, *, smoke: bool, resume: Path | None) -> None:
    config = verify(root)
    from transformers import AutoTokenizer

    from gleipnir.branch_data import BranchDataset
    from gleipnir.branch_model import MODEL_ID, MODEL_REVISION, load_branch_model
    from gleipnir.branch_trainer import train

    destination = root / "smoke" if smoke else root / "runs" / config["job_name"]
    if resume is None:
        destination.mkdir(parents=True, exist_ok=False)
    elif not (resume / "complete.json").is_file():
        raise ValueError("incomplete resume checkpoint")
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
        if [tokenizer.encode(s, add_special_tokens=False) for s in ("0", "1")] != [
            [15],
            [16],
        ]:
            raise ValueError("decision token drift")
        dataset = BranchDataset(tokenizer)
        model, metadata = load_branch_model(
            seed=config["seed"], adapter=resume / "adapter" if resume else None
        )
        metadata.update(
            config=config,
            provenance=dataset.provenance,
            code_revision=os.environ.get("GLEIPNIR_COMMIT"),
        )
        atomic_write_json(destination / "initialization.json", metadata)
        result = train(
            model,
            dataset,
            config,
            destination,
            resume=resume,
            smoke_parents=8 if smoke else None,
        )
        if not smoke:
            adapter = destination / "causal_adapter"
            model.save_pretrained(adapter, safe_serialization=True)
            tokenizer.save_pretrained(adapter)
            atomic_write_json(
                adapter / "training_metadata.json", {**metadata, **result}
            )
            from gleipnir.qwen35_adapter_rebase import rebase_adapter

            rebase_adapter(adapter, destination / "model")
        atomic_write_json(destination / "status.json", {"state": "complete", **result})
    except BaseException as error:
        atomic_write_json(destination / "failure.json", {"error": repr(error)})
        raise


def pipeline(root: Path) -> None:
    """Training exits before existing serving parity and persistent ID inference."""
    config = verify(root)
    smoke = json.loads((root / "smoke/status.json").read_text())
    if smoke.get("state") != "complete":
        raise ValueError("fresh-adapter integration smoke has not passed")
    from experiments.monitoring_lr_sweep.run_lambda import gpu_training_environment
    from experiments.monitoring_objective_ablation.run_lambda import run_serving_parity
    from gleipnir.qwen35_fast_training import ensure_qwen35_long_trajectory_kernels

    fast = gpu_training_environment(ensure_qwen35_long_trajectory_kernels(), 0)
    fast["CUDA_VISIBLE_DEVICES"] = "0,1"
    status = {"state": "running", "phase": "training"}
    try:
        atomic_write_json(root / "status.json", status)
        subprocess.run(
            [
                sys.executable,
                "-u",
                "-m",
                "experiments.monitoring_branching.train",
                "train",
                "--root",
                str(root),
            ],
            env=fast,
            check=True,
        )
        status["phase"] = "serving_parity"
        atomic_write_json(root / "status.json", status)
        from experiments.monitoring_objective_ablation.run_lambda import (
            runtime_environment,
        )

        serving = runtime_environment("0")
        fast["CUDA_VISIBLE_DEVICES"] = "0"
        run_serving_parity(root / "id_benchmark.json", root, fast, serving)
        status["phase"] = "id_evaluation"
        atomic_write_json(root / "status.json", status)
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
                "--only-job",
                config["job_name"],
            ],
            env=serving,
            check=True,
        )
        status.update(state="complete", phase="complete")
    except BaseException as error:
        status.update(state="failed", error=repr(error))
        raise
    finally:
        atomic_write_json(root / "status.json", status)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "smoke", "train", "pipeline"))
    parser.add_argument(
        "--root", type=Path, default=Path("results/monitoring_branching_training")
    )
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    if args.phase == "prepare":
        print(prepare())
    elif args.phase == "pipeline":
        pipeline(args.root)
    else:
        training(args.root, smoke=args.phase == "smoke", resume=args.resume)


if __name__ == "__main__":
    main()
