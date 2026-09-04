import json
from pathlib import Path

from experiments.monitoring_inner_mixer_compile.core import (
    CANDIDATE_NAME,
    CANDIDATE_POLICY,
    CONTROL_NAME,
    CONTROL_POLICY,
    EXPECTED_ALL_LAYERS,
    OPTIMIZER_STEPS,
    summarize,
    validate_jobs,
)
from experiments.monitoring_inner_mixer_compile.prepare import make_jobs
from experiments.monitoring_selective_checkpointing.core import (
    EXPECTED_CHECKPOINTED_LAYERS,
)
from experiments.tool_trajectory_monitoring.run_distillation_train import (
    training_command,
)


def metadata(policy: str, rate: float) -> dict:
    candidate = policy == CANDIDATE_POLICY
    return {
        "gradient_checkpointing_policy": "linear_attention_only",
        "gradient_checkpointing_layer_indices": None,
        "checkpointed_layer_indices": EXPECTED_CHECKPOINTED_LAYERS,
        "selective_torch_compile": {
            "policy": policy,
            "compiled_layer_indices": EXPECTED_ALL_LAYERS,
            "disabled_linear_attention_layer_indices": (
                [] if candidate else EXPECTED_CHECKPOINTED_LAYERS
            ),
            "disabled_linear_attention_kernel_layer_indices": (
                EXPECTED_CHECKPOINTED_LAYERS if candidate else []
            ),
            "disabled_linear_attention_kernel_components": (
                ["causal_conv1d_fn", "chunk_gated_delta_rule", "norm.forward"]
                if candidate
                else []
            ),
            "backend": "inductor",
            "mode": "default",
            "dynamic": True,
            "global_torch_compile": False,
            "dynamo_counters": {"stats": {"unique_graphs": 7 if candidate else 3}},
            "canary": None,
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
        "peak_cuda_memory_allocated_bytes": 28 * 2**30,
    }


def test_jobs_are_matched_eight_step_conditions(tmp_path: Path) -> None:
    selection = tmp_path / "selection.jsonl"
    selection.write_text("{}\n")
    jobs = make_jobs(tmp_path / "data", tmp_path / "results", selection)
    validate_jobs(jobs)
    assert [job["job_name"] for job in jobs] == [CONTROL_NAME, CANDIDATE_NAME]
    assert {job["max_steps"] for job in jobs} == {8}
    assert {
        job["selective_torch_compile_policy"] for job in jobs
    } == {CONTROL_POLICY, CANDIDATE_POLICY}
    assert (
        "student.training.selective_torch_compile_policy="
        "full_attention_and_linear_mixer_segments" in training_command(jobs[1])
    )


def test_summary_rejects_sub_one_percent_gain(tmp_path: Path) -> None:
    selection = tmp_path / "selection.jsonl"
    selection.write_text("{}\n")
    jobs = make_jobs(tmp_path / "data", tmp_path / "results", selection)
    jobs[0]["training_metadata"] = metadata(CONTROL_POLICY, 1.0)
    jobs[1]["training_metadata"] = metadata(CANDIDATE_POLICY, 1.009)
    assert summarize(jobs)["selected_policy"] == CONTROL_POLICY
    jobs[1]["training_metadata"]["train_metrics"]["train_samples_per_second"] = 1.01
    assert summarize(jobs)["selected_policy"] == CANDIDATE_POLICY


def test_config_freezes_eight_steps() -> None:
    config = json.loads(
        Path("experiments/monitoring_inner_mixer_compile/config.json").read_text()
    )
    assert config["optimizer_steps"] == 8
    assert config["minimum_relative_improvement"] == 0.01
