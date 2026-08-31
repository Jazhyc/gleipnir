from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from scripts import prepare_hf_adapter_release as prepare
from scripts import upload_hf_adapter_release as upload


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")


def adapter_directory(path: Path, base_model: str, key: str) -> dict:
    path.mkdir(parents=True)
    write_json(
        path / "adapter_config.json",
        {
            "base_model_name_or_path": base_model,
            "peft_type": "LORA",
            "r": 128,
        },
    )
    save_file({key: torch.ones(2, 2, dtype=torch.float32)}, path / prepare.WEIGHTS_NAME)
    return prepare.inspect_adapter(path / prepare.WEIGHTS_NAME)


def release_spec(tmp_path: Path) -> tuple[dict, Path, dict]:
    base_model = "Qwen/Qwen3.5-Test"
    source_dir = tmp_path / "source"
    serving_dir = tmp_path / "serving"
    source = adapter_directory(source_dir, base_model, "base_model.model.layer.weight")
    serving = adapter_directory(
        serving_dir, base_model, "base_model.model.language_model.layer.weight"
    )
    card = tmp_path / "README.md"
    card.write_text("# Gleipnir Test\n", encoding="utf-8")
    instruction = tmp_path / "student.txt"
    instruction.write_text("Classify this trajectory.\n", encoding="utf-8")
    common = {
        "format_version": 1,
        "training": {"data_rows": 2},
        "evaluation": {"rows": 4},
    }
    model = {
        "base_model": base_model,
        "base_revision": "abc123",
        "display_name": "Gleipnir Test",
        "evaluation": {"mean_ood_auroc": 0.75},
        "job_name": "test-job",
        "repo_name": "Gleipnir-Test",
    }
    spec = {
        "name": "test",
        "model": model,
        "source_dir": source_dir,
        "serving_dir": serving_dir,
        "card_path": card,
        "source": source,
        "serving": serving,
        "rebase": {
            "destination_prefix": "base_model.model.language_model.",
            "destination_sha256": serving["sha256"],
            "format": "test-rebase-v1",
            "source_prefix": "base_model.model.",
            "source_sha256": source["sha256"],
            "tensor_count": 1,
        },
        "metadata": {
            "commit": "def456",
            "peak_cuda_memory_allocated_bytes": 1024,
            "train_metrics": {"train_runtime": 12.5},
            "trainable_parameters": 4,
        },
        "common": common,
    }
    contract = {
        "prompt_set_id": "test-v1",
        "student_instruction_file": "student_prompt.txt",
    }
    return spec, instruction, contract


def test_stage_model_builds_checksum_complete_release(tmp_path: Path) -> None:
    spec, instruction, contract = release_spec(tmp_path)
    output = tmp_path / "output"
    output.mkdir()

    destination = prepare.stage_model(spec, output, instruction, contract)
    manifest = upload.load_manifest(destination)

    assert destination.name == "Gleipnir-Test"
    assert manifest["base_revision"] == "abc123"
    assert manifest["files"]["adapter_model.safetensors"]["sha256"] == spec[
        "source"
    ]["sha256"]
    assert manifest["files"]["vllm/adapter_model.safetensors"]["sha256"] == spec[
        "serving"
    ]["sha256"]
    summary = json.loads((destination / "training_summary.json").read_text())
    assert summary["training"]["runtime_seconds"] == 12.5


def test_stage_model_refuses_to_overwrite_release(tmp_path: Path) -> None:
    spec, instruction, contract = release_spec(tmp_path)
    output = tmp_path / "output"
    output.mkdir()
    prepare.stage_model(spec, output, instruction, contract)

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        prepare.stage_model(spec, output, instruction, contract)


def test_upload_validation_rejects_changed_staged_file(tmp_path: Path) -> None:
    spec, instruction, contract = release_spec(tmp_path)
    output = tmp_path / "output"
    output.mkdir()
    destination = prepare.stage_model(spec, output, instruction, contract)
    (destination / "student_prompt.txt").write_text("changed\n", encoding="utf-8")

    with pytest.raises(ValueError, match="size drifted|checksum drifted"):
        upload.load_manifest(destination)


def test_validate_adapter_directory_rejects_non_fp32_weights(tmp_path: Path) -> None:
    path = tmp_path / "adapter"
    path.mkdir()
    write_json(
        path / "adapter_config.json",
        {
            "base_model_name_or_path": "Qwen/Qwen3.5-Test",
            "peft_type": "LORA",
            "r": 128,
        },
    )
    weights = path / prepare.WEIGHTS_NAME
    save_file({"layer.weight": torch.ones(1, dtype=torch.bfloat16)}, weights)

    with pytest.raises(ValueError, match="must remain FP32"):
        prepare.validate_adapter_directory(
            path,
            {"base_model": "Qwen/Qwen3.5-Test"},
            prepare.sha256_file(weights),
        )


def test_upload_script_runs_as_a_file() -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(upload.ROOT / "scripts/upload_hf_adapter_release.py"),
            "--help",
        ],
        cwd=upload.ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
