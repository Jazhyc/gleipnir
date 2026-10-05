"""Run a frozen packed monitoring job with validated startup reuse and export."""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path
from typing import Any

from gleipnir.monitoring_campaign_data import file_hash, read_rows, write_json
from gleipnir.monitoring_campaign_runtime import training_environment
from gleipnir.monitoring_training_command import training_command
from gleipnir.qwen35_adapter_rebase import rebase_adapter


def run_training(
    root: Path,
    job: dict[str, Any],
    config: dict[str, Any],
    *,
    manifest_path: Path,
    config_path: Path,
    source_paths: tuple[str, ...],
) -> None:
    """Audit exact inputs, complete one epoch and preserve FP32 adapter layouts."""
    from experiments.monitoring_hard_labels.train import validate_training_metadata

    started = time.monotonic()
    output = Path(job["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    path = output / "job.json"
    if path.exists() and json.loads(path.read_text()) != job:
        raise ValueError("existing training job drift")
    write_json(path, job)
    master = Path(job["causal_adapter_dir"])
    if not (master / "training_metadata.json").exists():
        from transformers import AutoTokenizer

        write_json(output / "status.json", {"stage": "token_audit"})
        tokenizer = AutoTokenizer.from_pretrained(
            job["model"], revision=job["model_revision"]
        )
        lengths = []
        rows = read_rows(Path(job["student_rows"]))
        for n, row in enumerate(rows, 1):
            rendered = (
                tokenizer.apply_chat_template(
                    [{"role": "user", "content": row["student_prompt"]}],
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
                + "Prediction:"
            )
            lengths.append(len(tokenizer.encode(rendered, add_special_tokens=False)))
            if n % 500 == 0:
                print(f"token_audit_progress {n}/{config['training_rows']}", flush=True)
        audit = {
            "rows": len(lengths),
            "maximum": max(lengths),
            "total": sum(lengths),
            "configured_cap": job["max_length"],
            "truncated": 0,
        }
        write_json(output / "token_audit.json", audit)
        if len(lengths) != config["training_rows"] or max(lengths) > job["max_length"]:
            raise ValueError("training zero-truncation/coverage gate failed")
        command = training_command(job)
        command.append(
            f"student.init_adapter={root / config['model']['initial_adapter']}"
        )
        write_json(
            output / "execution_contract.json",
            {
                "job": job,
                "command": command,
                "manifest_sha256": file_hash(manifest_path),
                "source_sha256": {p: file_hash(root / p) for p in source_paths},
            },
        )
        write_json(
            output / "status.json",
            {
                "stage": "training",
                "validation": "reuse_recorded_recipe",
                "reference_sha256": config["startup_validation_reference_sha256"],
            },
        )
        subprocess.run(
            command,
            cwd=root,
            env=training_environment(
                root,
                config["campaign_id"],
                native_fp4_mlp=job.get("native_fp4_mlp", False),
            ),
            check=True,
        )
    metadata = json.loads((master / "training_metadata.json").read_text())
    validate_training_metadata(
        metadata,
        job,
        config["model"]["initial_master_sha256"],
        expected_steps=config["expected_steps"],
    )
    serving = Path(job["model_dir"])
    rebase_adapter(master, serving)
    write_json(
        output / "complete.json",
        {
            "status": "trained",
            "epochs": 1,
            "steps": config["expected_steps"],
            "training_rows": config["training_rows"],
            "config_sha256": file_hash(config_path),
            "manifest_sha256": file_hash(manifest_path),
            "teacher_targets_sha256": file_hash(Path(job["soft_targets"])),
            "master_sha256": file_hash(master / "adapter_model.safetensors"),
            "serving_sha256": file_hash(serving / "adapter_model.safetensors"),
        },
    )
    write_json(
        output / "status.json",
        {
            "stage": "trained",
            "steps": config["expected_steps"],
            "seconds_this_invocation": time.monotonic() - started,
        },
    )
    print("training_complete 4b filtered", flush=True)
