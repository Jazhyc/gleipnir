import json
from pathlib import Path

from experiments.monitoring_selective_compile.core import (
    COMPILE_POLICY,
    EXPECTED_COMPILED_LAYERS,
    JOB_NAME,
    summarize,
    validate_jobs,
)
from experiments.monitoring_selective_compile.prepare import make_job
from experiments.tool_trajectory_monitoring.run_distillation_train import (
    training_command,
)


def candidate_metadata(rate: float) -> dict:
    return {
        "gradient_checkpointing_policy": "linear_attention_only",
        "checkpointed_layer_indices": [
            index for index in range(32) if (index + 1) % 4 != 0
        ],
        "selective_torch_compile": {
            "policy": COMPILE_POLICY,
            "compiled_layer_indices": EXPECTED_COMPILED_LAYERS,
            "backend": "inductor",
            "mode": "default",
            "dynamic": True,
            "global_torch_compile": False,
            "dynamo_counters": {"stats": {"unique_graphs": 8}},
        },
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
        "train_metrics": {"train_samples_per_second": rate},
        "peak_cuda_memory_allocated_bytes": 40 * 2**30,
    }


def test_job_changes_only_selective_compilation(tmp_path: Path) -> None:
    selection = tmp_path / "selection.jsonl"
    selection.write_text("{}\n")
    job = make_job(tmp_path / "data", tmp_path / "results", selection)
    validate_jobs([job])
    assert job["job_name"] == JOB_NAME
    assert job["selective_torch_compile_policy"] == COMPILE_POLICY
    command = training_command(job)
    assert (
        "student.training.selective_torch_compile_policy="
        "uncheckpointed_full_attention" in command
    )
    assert "student.training.selective_torch_compile_dynamic=true" in command


def test_summary_requires_five_percent_gain() -> None:
    baseline = {"train_metrics": {"train_samples_per_second": 1.0}}
    all_layer_baseline = {"train_metrics": {"train_samples_per_second": 1.0}}
    candidate = candidate_metadata(1.049)
    assert (
        summarize(baseline, all_layer_baseline, candidate)["selected_policy"]
        == "none"
    )
    candidate["train_metrics"]["train_samples_per_second"] = 1.05
    assert (
        summarize(baseline, all_layer_baseline, candidate)["selected_policy"]
        == COMPILE_POLICY
    )


def test_config_matches_policy() -> None:
    config = json.loads(
        Path("experiments/monitoring_selective_compile/config.json").read_text()
    )
    assert config["selective_torch_compile_policy"] == COMPILE_POLICY
