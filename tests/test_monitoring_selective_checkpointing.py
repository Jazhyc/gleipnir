import json
from pathlib import Path

from experiments.monitoring_selective_checkpointing.core import (
    EXPECTED_CHECKPOINTED_LAYERS,
    JOB_NAME,
    POLICY,
    summarize,
    validate_jobs,
)
from experiments.monitoring_selective_checkpointing.prepare import make_job
from experiments.tool_trajectory_monitoring.run_distillation_train import (
    training_command,
)


def test_job_changes_only_checkpointing_policy(tmp_path: Path) -> None:
    selection = tmp_path / "selection.jsonl"
    selection.write_text("{}\n")
    job = make_job(tmp_path / "data", tmp_path / "results", selection)
    validate_jobs([job])
    assert job["job_name"] == JOB_NAME
    assert job["gradient_checkpointing_policy"] == POLICY
    assert len(EXPECTED_CHECKPOINTED_LAYERS) == 24
    assert (
        "student.training.gradient_checkpointing_policy=linear_attention_only"
        in training_command(job)
    )


def test_summary_requires_five_percent_gain() -> None:
    baseline = {"train_metrics": {"train_samples_per_second": 1.0}}
    candidate = {
        "gradient_checkpointing": True,
        "gradient_checkpointing_policy": POLICY,
        "checkpointed_layer_indices": EXPECTED_CHECKPOINTED_LAYERS,
        "materialized_training_inputs": {"completion": False, "direct": True},
        "training_batch": {
            "micro_batch_size": 1,
            "gradient_accumulation_steps": 32,
            "effective_batch_size": 32,
        },
        "flash_linear_attention": {
            "version": "0.5.2",
            "backend_dispatch_disabled": True,
            "triton_version": "3.7.1",
        },
        "gated_delta_kernel_modules": ["fla.ops.gated_delta_rule.chunk"],
        "causal_conv1d": {
            "version": "1.6.2.post1",
            "kernel_modules": ["causal_conv1d.causal_conv1d_interface"],
        },
        "training_state": {"global_step": 20},
        "train_metrics": {"train_samples_per_second": 1.049},
        "peak_cuda_memory_allocated_bytes": 40 * 2**30,
    }
    assert summarize(baseline, candidate)["selected_policy"] == "all"
    candidate["train_metrics"]["train_samples_per_second"] = 1.05
    assert summarize(baseline, candidate)["selected_policy"] == POLICY


def test_config_matches_policy() -> None:
    config = json.loads(
        Path("experiments/monitoring_selective_checkpointing/config.json").read_text()
    )
    assert config["gradient_checkpointing_policy"] == POLICY
