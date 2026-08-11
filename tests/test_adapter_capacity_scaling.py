import json
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from experiments.adapter_capacity_scaling.core import (
    DEFAULT_RANKS,
    balanced_lora_lanes,
    split_jobs,
    validate_ranks,
)
from experiments.adapter_capacity_scaling.run_lambda import Status
from experiments.adapter_capacity_scaling.run_train import training_command
from gleipnir.qwen35_fast_training import (
    FLA_PACKAGES,
    fla_environment,
    fla_install_command,
)


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
        "causal_adapter_dir": "/results/run/causal_adapter",
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
    assert "student.model_loader=causal_lm" in lora_command
    assert "student.quantization.enabled=true" in lora_command
    assert "student.training.per_device_train_batch_size=8" in lora_command
    assert "student.training.gradient_accumulation_steps=4" in lora_command
    assert "student.output_dir=/results/run/causal_adapter" in lora_command
    assert "accelerate.commands.launch" not in lora_command
    assert "accelerate.commands.launch" in full_command
    assert "student.finetuning_mode=full" in full_command
    assert "student.model_loader=image_text_to_text" in full_command
    assert "student.quantization.enabled=false" in full_command
    assert "student.training.fsdp.enabled=true" in full_command
    assert "student.training.per_device_train_batch_size=1" in full_command


def test_full_training_rejects_one_process() -> None:
    with pytest.raises(ValueError, match="at least two processes"):
        training_command(full_job(0), distributed_processes=1)


def test_status_distinguishes_planned_and_cancelled_jobs(tmp_path) -> None:
    status_path = tmp_path / "status.json"
    Status(status_path, [lora_job(0, 16)], [full_job(0)])

    status = json.loads(status_path.read_text())
    assert status["planned_jobs"] == ["seed0-r016"]
    assert status["cancelled_jobs"] == ["seed0-full"]


def test_capacity_config_uses_standard_qlora_and_fast_batch() -> None:
    cfg = OmegaConf.load("experiments/adapter_capacity_scaling/config.yaml")

    assert cfg.student.quantization.enabled is True
    assert cfg.student.quantization.bnb_4bit_quant_type == "nf4"
    assert cfg.student.quantization.bnb_4bit_use_double_quant is True
    assert cfg.student.quantization.bnb_4bit_compute_dtype == "bfloat16"
    assert cfg.student.training.per_device_train_batch_size == 8
    assert cfg.student.training.gradient_accumulation_steps == 4
    assert cfg.student.training.require_flash_linear_attention is True
    assert cfg.student.training.torch_compile is False


def test_fla_install_is_pinned_and_isolated() -> None:
    target = Path("/tmp/test-fla")
    command = fla_install_command(
        target,
        python=Path("/opt/venv/bin/python"),
        uv_executable="/usr/bin/uv",
    )
    environment = fla_environment(target, {"PYTHONPATH": "/existing"})

    assert command[:3] == ["/usr/bin/uv", "pip", "install"]
    assert list(FLA_PACKAGES) == [
        "flash-linear-attention==0.5.2",
        "fla-core==0.5.2",
    ]
    assert command[-1] == "--no-deps"
    assert environment["PYTHONPATH"] == "/tmp/test-fla:/existing"
    assert environment["GLEIPNIR_FLA_VERSION"] == "0.5.2"
