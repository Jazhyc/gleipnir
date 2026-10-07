"""Train the approved 4B prompt with the unchanged selected packed recipe."""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

from experiments.monitoring_hard_labels.train import validate_training_metadata
from experiments.monitoring_injection_disentanglement.prepare import (
    CONFIG,
    DATA,
    OUTPUT,
    ROOT,
    configuration,
    file_hash,
    read_rows,
    resolved_profile,
    verify_preparation,
    write_json,
)
from gleipnir.monitoring_campaign_runtime import training_environment
from gleipnir.monitoring_training_command import training_command
from gleipnir.qwen35_adapter_rebase import rebase_adapter


def make_job(config: dict, recipe: dict) -> dict:
    output = OUTPUT / "4b/conservative"
    return {
        **recipe,
        "job_name": "4b-conservative",
        "model": config["model"]["id"],
        "model_revision": config["model"]["revision"],
        "seed": config["seed"],
        "student_rows": str(DATA / "train/student_rows.jsonl"),
        "soft_targets": str(DATA / "soft_targets.jsonl"),
        "selection_manifest": None,
        "num_train_epochs": 1.0,
        "max_steps": -1,
        "learning_rate": config["learning_rate"],
        "soft_loss_weight": 1.0,
        "direct_loss_weight": 0.0,
        "completion_loss_weight": 0.0,
        "startup_validation_reference": str(
            ROOT / config["startup_validation_reference"]
        ),
        "save_steps": 1000000,
        "gradient_checkpointing": False,
        "output_dir": str(output),
        "causal_adapter_dir": str(output / "causal_adapter"),
        "model_dir": str(output / "model"),
        "hydra_log_dir": str(
            ROOT / "logs/runpod/monitoring_injection_disentanglement/hydra"
        ),
    }


def main() -> None:
    started = time.monotonic()
    config = configuration()
    manifest = verify_preparation()
    job = make_job(config, resolved_profile()["recipe"])
    output = Path(job["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    if (output / "job.json").exists() and json.loads(
        (output / "job.json").read_text()
    ) != job:
        raise ValueError("existing job contract drift")
    write_json(output / "job.json", job)
    master = Path(job["causal_adapter_dir"])
    if not (master / "training_metadata.json").exists():
        from transformers import AutoTokenizer

        write_json(output / "status.json", {"stage": "token_audit"})
        tokenizer = AutoTokenizer.from_pretrained(
            job["model"], revision=job["model_revision"]
        )
        lengths = []
        for n, row in enumerate(read_rows(Path(job["student_rows"])), 1):
            p = (
                tokenizer.apply_chat_template(
                    [{"role": "user", "content": row["student_prompt"]}],
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
                + "Prediction:"
            )
            lengths.append(len(tokenizer.encode(p, add_special_tokens=False)))
            if n % 500 == 0:
                print(f"token_audit_progress {n}/8688", flush=True)
        audit = {
            "rows": len(lengths),
            "maximum": max(lengths),
            "total": sum(lengths),
            "configured_cap": job["max_length"],
            "truncated": 0,
        }
        write_json(output / "token_audit.json", audit)
        if len(lengths) != 8688 or max(lengths) > job["max_length"]:
            raise ValueError("training zero-truncation gate failed")
        command = training_command(job)
        command.append(
            f"student.init_adapter={ROOT / config['model']['initial_adapter']}"
        )
        write_json(
            output / "execution_contract.json",
            {
                "job": job,
                "command": command,
                "manifest_sha256": file_hash(DATA / "manifest.json"),
                "source_sha256": {
                    p: file_hash(ROOT / p)
                    for p in (
                        "experiments/monitoring_injection_disentanglement/train.py",
                        "experiments/deception_distillation/train_student_sft.py",
                        "src/gleipnir/campaigns/training_command.py",
                        "src/gleipnir/__init__.py",
                        "src/gleipnir/_compat.py",
                        "src/gleipnir/training/packed.py",
                        "src/gleipnir/packed_sequences.py",
                    )
                },
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
            cwd=ROOT,
            env=training_environment(ROOT, "monitoring_injection_disentanglement_v1"),
            check=True,
        )
    metadata = json.loads((master / "training_metadata.json").read_text())
    validate_training_metadata(metadata, job, config["model"]["initial_master_sha256"])
    rebase_adapter(master, Path(job["model_dir"]))
    write_json(
        output / "complete.json",
        {
            "status": "trained",
            "epochs": 1,
            "steps": 272,
            "config_sha256": file_hash(CONFIG),
            "template_sha256": manifest["template_sha256"]["conservative"],
            "teacher_targets_sha256": file_hash(DATA / "soft_targets.jsonl"),
            "master_sha256": file_hash(master / "adapter_model.safetensors"),
            "serving_sha256": file_hash(
                Path(job["model_dir"]) / "adapter_model.safetensors"
            ),
        },
    )
    write_json(
        output / "status.json",
        {
            "stage": "trained",
            "steps": 272,
            "seconds_this_invocation": time.monotonic() - started,
        },
    )
    print("training_complete 4b conservative", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        write_json(
            OUTPUT / f"failure_train_{time.time_ns()}.json",
            {"type": type(exc).__name__, "message": str(exc)},
        )
        raise
