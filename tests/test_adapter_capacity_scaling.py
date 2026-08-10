import pytest

from experiments.adapter_capacity_scaling.core import (
    DEFAULT_RANKS,
    balanced_lora_lanes,
    split_jobs,
    validate_ranks,
)
from experiments.adapter_capacity_scaling.run_train import training_command


def lora_job(seed: int, rank: int) -> dict[str, object]:
    return {
        "job_name": f"seed{seed}-r{rank:03d}",
        "capacity_kind": "lora",
        "seed": seed,
        "rank": rank,
        "lora_alpha": 2 * rank,
        "student_rows": "/data/student.jsonl",
        "soft_targets": "/data/soft.jsonl",
        "output_dir": "/results/run",
        "model_dir": "/results/run/model",
    }


def full_job(seed: int) -> dict[str, object]:
    return {
        "job_name": f"seed{seed}-full",
        "capacity_kind": "full",
        "seed": seed,
        "rank": None,
        "lora_alpha": None,
        "student_rows": "/data/student.jsonl",
        "soft_targets": "/data/soft.jsonl",
        "output_dir": "/results/full",
        "model_dir": "/results/full/model",
    }


def test_default_ranks_are_powers_of_two_through_256() -> None:
    assert validate_ranks(DEFAULT_RANKS) == [1, 2, 4, 8, 16, 32, 64, 128, 256]
    with pytest.raises(ValueError, match="powers of two"):
        validate_ranks([1, 3, 4])


def test_capacity_jobs_split_and_balance_without_loss() -> None:
    lora = [lora_job(seed, rank) for seed in range(3) for rank in DEFAULT_RANKS]
    full = [full_job(seed) for seed in range(3)]
    selected_lora, selected_full = split_jobs(lora + full)
    lanes = balanced_lora_lanes(selected_lora, 2)

    assert selected_full == full
    assert len(selected_lora) == 27
    assert sorted(job["job_name"] for lane in lanes for job in lane) == sorted(
        job["job_name"] for job in lora
    )
    loads = [sum(256 + int(job["rank"]) for job in lane) for lane in lanes]
    assert abs(loads[0] - loads[1]) <= 512


def test_training_commands_keep_alpha_ratio_and_use_fsdp_for_full() -> None:
    lora_command = training_command(lora_job(1, 128), distributed_processes=2)
    full_command = training_command(full_job(2), distributed_processes=2)

    assert "student.lora.r=128" in lora_command
    assert "student.lora.alpha=256" in lora_command
    assert "student.finetuning_mode=lora" in lora_command
    assert "accelerate.commands.launch" not in lora_command
    assert "accelerate.commands.launch" in full_command
    assert "student.finetuning_mode=full" in full_command
    assert "student.training.fsdp.enabled=true" in full_command
    assert "student.training.per_device_train_batch_size=1" in full_command


def test_full_training_rejects_one_process() -> None:
    with pytest.raises(ValueError, match="at least two processes"):
        training_command(full_job(0), distributed_processes=1)
