"""Audit the short B200 screen without GPU execution or paid operations."""

from pathlib import Path

from experiments.tool_trajectory_monitoring.run_distillation_train import (
    training_command,
)
from gleipnir.monitoring_systems_screen import load_config, make_jobs, resolve_paths

CONFIG = Path("experiments/b200_training_throughput/config.yaml")
WARM_CONFIG = CONFIG.with_name("warm_config.yaml")
FA4_CONFIG = CONFIG.with_name("fa4_config.yaml")


def test_batched_repeat_requires_its_own_longest_row_preflight() -> None:
    config = load_config(CONFIG.with_name("batching_repeat_config.yaml"))
    original = load_config(CONFIG)
    assert config["selection"] == original["selection"]
    assert config["data"] == original["data"]
    assert config["preflight"]["condition"] == "length-grouped-b2"
    jobs = make_jobs(config, resolve_paths(config), "fixed-selection")
    for job in jobs:
        assert job["attn_implementation"] == "sdpa"
        assert job["gradient_checkpointing_policy"] == "linear_attention_only"
        assert job["max_steps"] == 10 and job["train_rows"] == 320
        assert job["micro_batch_size"] * job["gradient_accumulation_steps"] == 32


def test_fa4_screen_changes_only_full_attention_backend() -> None:
    config = load_config(FA4_CONFIG)
    original = load_config(CONFIG)
    assert config["selection"] == original["selection"]
    assert config["data"] == original["data"]
    jobs = make_jobs(config, resolve_paths(config), "fixed-selection")
    assert [j["attn_implementation"] for j in jobs] == ["sdpa", "flash_attention_4"]
    for job in jobs:
        assert job["micro_batch_size"] == 1
        assert job["gradient_accumulation_steps"] == 32
        assert job["gradient_checkpointing_policy"] == "linear_attention_only"
        assert job["max_steps"] == 10 and job["train_rows"] == 320
        assert (
            f"student.attn_implementation={job['attn_implementation']}"
            in training_command(job)
        )
    candidate = jobs[1]
    assert candidate["attention_backend_version"] == "4.0.0b33"
    assert candidate["attention_backend_canary_reference"] == "sdpa"
    assert "student.training.attention_backend_version=4.0.0b33" in training_command(
        candidate
    )
    assert (
        "student.training.attention_backend_canary_reference=sdpa"
        in training_command(candidate)
    )
    assert config["preflight"]["condition"] == candidate["job_name"]


def test_warmed_continuation_preserves_cohort_and_bounds_checkpoint_removal() -> None:
    original = load_config(CONFIG)
    config = load_config(WARM_CONFIG)
    assert config["selection"] == original["selection"]
    assert config["data"] == original["data"]
    assert config["artifacts"]["result_dir"] != original["artifacts"]["result_dir"]
    jobs = make_jobs(config, resolve_paths(config), "fixed-selection")
    assert config["preflight"]["condition"] == "half-checkpoint-b1"
    half = next(j for j in jobs if j["job_name"] == "half-checkpoint-b1")
    indices = half["gradient_checkpointing_layer_indices"]
    assert len(indices) == 12
    assert set(indices) <= set(
        original["metadata_expectations"]["checkpointed_layer_indices"]
    )
    for job in jobs:
        assert job["max_steps"] == 10 and job["train_rows"] == 320
        assert job["micro_batch_size"] * job["gradient_accumulation_steps"] == 32
        assert job["rank"] == 128 and job["gradient_checkpointing"] is True


def test_all_conditions_use_ten_updates_and_the_same_effective_batch() -> None:
    config = load_config(CONFIG)
    jobs = make_jobs(config, resolve_paths(config), "fixed-selection")
    assert config["gpus"] == 1
    assert config["selection"]["rows"] == 320
    assert config["data"]["rows"] == 21837
    for job in jobs:
        assert job["max_steps"] == 10
        assert job["train_rows"] == 320
        assert job["micro_batch_size"] * job["gradient_accumulation_steps"] == 32
        assert job["selection_sha256"] == "fixed-selection"
        assert job["rank"] == 128
        assert job["soft_loss_weight"] == 1.0
        assert job["direct_loss_weight"] == 0.0
        assert (
            job["selective_torch_compile_policy"] == "full_attention_and_linear_shell"
        )
    no_checkpoint = next(j for j in jobs if j["job_name"] == "no-checkpoint-b1")
    assert no_checkpoint["gradient_checkpointing"] is False
    assert no_checkpoint["gradient_checkpointing_policy"] == "all"


def test_remote_training_logs_are_unique_and_use_the_runpod_tree() -> None:
    config = load_config(CONFIG)
    jobs = make_jobs(config, resolve_paths(config), "fixed-selection")
    log_arguments = []
    for job in jobs:
        args = training_command(job)
        logs = [arg for arg in args if arg.startswith("hydra.run.dir=")]
        assert logs == [
            f"hydra.run.dir=logs/runpod/b200_training_throughput/hydra/{job['job_name']}"
        ]
        log_arguments.extend(logs)
    assert len(log_arguments) == len(set(log_arguments))
