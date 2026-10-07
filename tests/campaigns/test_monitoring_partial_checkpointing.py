import json
from pathlib import Path

from experiments.monitoring_partial_checkpointing.core import (
    CANDIDATE_CHECKPOINT_POLICY,
    CANDIDATE_CHECKPOINTED_LAYERS,
    CANDIDATE_NAME,
    COMPILE_POLICY,
    CONTROL_CHECKPOINT_POLICY,
    CONTROL_NAME,
    EXPECTED_ALL_LAYERS,
    OPTIMIZER_STEPS,
    summarize,
    validate_jobs,
    validate_training_metadata,
)
from experiments.monitoring_partial_checkpointing.prepare import make_jobs
from experiments.monitoring_selective_checkpointing.core import (
    EXPECTED_CHECKPOINTED_LAYERS,
)
from experiments.tool_trajectory_monitoring.run_distillation_train import (
    training_command,
)


def metadata(policy: str, rate: float, peak_gib: float = 60.0) -> dict:
    candidate = policy == CANDIDATE_CHECKPOINT_POLICY
    checkpointed = (
        CANDIDATE_CHECKPOINTED_LAYERS
        if candidate
        else EXPECTED_CHECKPOINTED_LAYERS
    )
    return {
        "gradient_checkpointing_policy": policy,
        "gradient_checkpointing_layer_indices": checkpointed if candidate else None,
        "checkpointed_layer_indices": checkpointed,
        "selective_torch_compile": {
            "policy": COMPILE_POLICY,
            "compiled_layer_indices": EXPECTED_ALL_LAYERS,
            "disabled_linear_attention_layer_indices": EXPECTED_CHECKPOINTED_LAYERS,
            "backend": "inductor",
            "mode": "default",
            "dynamic": True,
            "global_torch_compile": False,
            "dynamo_counters": {"stats": {"unique_graphs": 4}},
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
        "training_state": {"global_step": OPTIMIZER_STEPS},
        "train_metrics": {
            "train_samples_per_second": rate,
            "train_runtime": 256 / rate,
        },
        "optimizer_step_timing": {"steady_mean_seconds": 30.0},
        "peak_cuda_memory_allocated_bytes": peak_gib * 2**30,
    }


def test_jobs_freeze_partial_checkpointing_and_eight_steps(tmp_path: Path) -> None:
    selection = tmp_path / "selection.jsonl"
    selection.write_text("{}\n")
    jobs = make_jobs(tmp_path / "data", tmp_path / "results", selection)
    validate_jobs(jobs)
    assert [job["job_name"] for job in jobs] == [CONTROL_NAME, CANDIDATE_NAME]
    assert {job["max_steps"] for job in jobs} == {8}
    assert jobs[1]["gradient_checkpointing_layer_indices"] == [2, 10, 18, 26]
    command = training_command(jobs[1])
    assert "student.training.gradient_checkpointing_policy=explicit" in command
    assert (
        "student.training.gradient_checkpointing_layer_indices=[2,10,18,26]"
        in command
    )


def test_preflight_rejects_candidate_above_memory_ceiling() -> None:
    value = metadata(CANDIDATE_CHECKPOINT_POLICY, 1.0, peak_gib=72.1)
    value["training_state"]["global_step"] = 1
    try:
        validate_training_metadata(
            value,
            checkpoint_policy=CANDIDATE_CHECKPOINT_POLICY,
            expected_steps=1,
            enforce_memory_ceiling=True,
        )
    except ValueError as error:
        assert "above 72.00 GiB ceiling" in str(error)
    else:
        raise AssertionError("unsafe preflight was accepted")


def test_summary_rejects_sub_one_percent_gain(tmp_path: Path) -> None:
    selection = tmp_path / "selection.jsonl"
    selection.write_text("{}\n")
    jobs = make_jobs(tmp_path / "data", tmp_path / "results", selection)
    jobs[0]["training_metadata"] = metadata(CONTROL_CHECKPOINT_POLICY, 1.0)
    jobs[1]["training_metadata"] = metadata(CANDIDATE_CHECKPOINT_POLICY, 1.009)
    result = summarize(jobs)
    assert result["selected_checkpointing_policy"] == CONTROL_CHECKPOINT_POLICY
    jobs[1]["training_metadata"]["train_metrics"][
        "train_samples_per_second"
    ] = 1.01
    result = summarize(jobs)
    assert result["selected_checkpointing_policy"] == CANDIDATE_CHECKPOINT_POLICY


def test_config_freezes_eight_steps_and_memory_ceiling() -> None:
    config = json.loads(
        Path("experiments/monitoring_partial_checkpointing/config.json").read_text()
    )
    assert config["optimizer_steps"] == 8
    assert config["maximum_preflight_allocated_gib"] == 72.0
