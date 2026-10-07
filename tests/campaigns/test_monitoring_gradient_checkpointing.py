import json
from pathlib import Path

from experiments.monitoring_gradient_checkpointing.core import (
    JOB_NAME,
    summarize,
    validate_jobs,
)
from experiments.monitoring_gradient_checkpointing.prepare import make_job
from experiments.tool_trajectory_monitoring.run_distillation_train import (
    training_command,
)


def test_job_disables_checkpointing_without_changing_batch(tmp_path: Path) -> None:
    selection = tmp_path / "selection.jsonl"
    selection.write_text("{}\n")
    job = make_job(tmp_path / "data", tmp_path / "results", selection)
    validate_jobs([job])
    assert job["job_name"] == JOB_NAME
    assert job["micro_batch_size"] == 1
    assert job["gradient_accumulation_steps"] == 32
    assert job["gradient_checkpointing"] is False
    assert "student.training.gradient_checkpointing=false" in training_command(job)


def test_summary_requires_five_percent_throughput_gain() -> None:
    baseline = {
        "training_batch": {
            "micro_batch_size": 1,
            "gradient_accumulation_steps": 32,
            "effective_batch_size": 32,
        },
        "train_metrics": {"train_samples_per_second": 1.0},
        "peak_cuda_memory_allocated_bytes": 10 * 2**30,
    }
    candidate = {
        "gradient_checkpointing": False,
        "training_batch": baseline["training_batch"],
        "flash_linear_attention": {
            "available": True,
            "required": True,
            "version": "0.5.2",
            "backend_dispatch_disabled": True,
            "triton_version": "3.7.1",
        },
        "gated_delta_kernel_modules": ["fla.ops.gated_delta_rule"],
        "causal_conv1d": {
            "available": True,
            "required": True,
            "version": "1.6.2.post1",
            "kernel_modules": ["causal_conv1d.causal_conv1d_interface"],
        },
        "training_state": {"global_step": 20},
        "train_metrics": {
            "train_samples_per_second": 1.049,
            "train_runtime": 10.0,
        },
        "batching": {"padding": {"direct_tokens": 1_000}},
        "optimizer_step_timing": {"steady_mean_seconds": 1.0},
        "peak_cuda_memory_allocated_bytes": 20 * 2**30,
    }
    assert summarize(baseline, candidate)["selected_gradient_checkpointing"] is True
    candidate["train_metrics"]["train_samples_per_second"] = 1.05
    assert summarize(baseline, candidate)["selected_gradient_checkpointing"] is False


def test_config_matches_intervention() -> None:
    config = json.loads(
        Path("experiments/monitoring_gradient_checkpointing/config.json").read_text()
    )
    assert config["intervention"]["name"] == JOB_NAME
    assert config["intervention"]["gradient_checkpointing"] is False
