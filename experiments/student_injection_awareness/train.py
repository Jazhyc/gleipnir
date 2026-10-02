"""Train one matched packed BF16 student and checksum-rebase its FP32 adapter."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

import yaml
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from experiments.student_injection_awareness.prepare import (
    CONFIG,
    DATA,
    OUTPUT,
    ROOT,
    VARIANTS,
    file_hash,
    read_rows,
    templates,
    write_json,
)
from experiments.tool_trajectory_monitoring.run_distillation_train import (
    training_command,
)
from gleipnir.flashqla_training import flashqla_environment
from gleipnir.monitoring_systems_screen import gpu_environment
from gleipnir.qwen35_adapter_rebase import rebase_adapter
from gleipnir.qwen35_fast_training import (
    DEFAULT_CAUSAL_CONV1D_TARGET,
    DEFAULT_FLA_TARGET,
    DEFAULT_TRITON_TARGET,
    causal_conv1d_environment,
    fla_environment,
    triton_environment,
)


def make_job(config: dict, recipe: dict, size: str, variant: str) -> dict:
    model = config["models"][size]
    output = OUTPUT / size / variant
    return {
        **recipe,
        "job_name": f"{size}-{variant}",
        "model": model["id"],
        "model_revision": model["revision"],
        "seed": config["seed"],
        "student_rows": str(DATA / variant / "student_rows.jsonl"),
        "soft_targets": str(DATA / "soft_targets.jsonl"),
        "selection_manifest": None,
        "num_train_epochs": 1.0,
        "max_steps": -1,
        "learning_rate": 5e-5,
        "save_steps": 1000000,
        "gradient_checkpointing": model["checkpointing"],
        "gradient_checkpointing_policy": "all",
        "gradient_checkpointing_layer_indices": None,
        "output_dir": str(output),
        "causal_adapter_dir": str(output / "causal_adapter"),
        "model_dir": str(output / "model"),
        "hydra_log_dir": str(ROOT / "logs/runpod/student_injection_awareness/hydra"),
    }


def environment() -> dict[str, str]:
    env = flashqla_environment(
        triton_environment(
            DEFAULT_TRITON_TARGET,
            causal_conv1d_environment(
                DEFAULT_CAUSAL_CONV1D_TARGET,
                fla_environment(DEFAULT_FLA_TARGET, dict(os.environ)),
            ),
        )
    )
    env = gpu_environment(env, 0, ROOT / ".cache/training/student_injection_awareness")
    env.update(
        FLA_DISABLE_BACKEND_DISPATCH="1",
        OMP_NUM_THREADS="4",
        WANDB_MODE="disabled",
        PYTHONUNBUFFERED="1",
    )
    return env


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", choices=("4b", "9b"), required=True)
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    args = parser.parse_args()
    config = yaml.safe_load(CONFIG.read_text())
    manifest = json.loads((DATA / "manifest.json").read_text())
    if manifest["config_sha256"] != file_hash(CONFIG):
        raise ValueError("campaign config drift")
    for relative in [f"{args.variant}/student_rows.jsonl", "soft_targets.jsonl"]:
        if file_hash(DATA / relative) != manifest["files_sha256"][relative]:
            raise ValueError("training input drift")
    if (
        templates()[args.variant].template_sha256
        != manifest["template_sha256"][args.variant]
    ):
        raise ValueError("student instruction drift")
    with initialize_config_dir(
        version_base=None, config_dir=str(ROOT / "src/gleipnir/configs/systems_screen")
    ):
        profile = OmegaConf.to_container(
            compose(config_name=config["profile"]), resolve=True
        )
    job = make_job(config, profile["recipe"], args.size, args.variant)
    output = Path(job["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "job.json", job)
    master = Path(job["causal_adapter_dir"])
    if not (master / "training_metadata.json").exists():
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            job["model"], revision=job["model_revision"]
        )
        rows = read_rows(Path(job["student_rows"]))
        lengths = []
        for number, row in enumerate(rows, start=1):
            prompt = (
                tokenizer.apply_chat_template(
                    [{"role": "user", "content": row["student_prompt"]}],
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
                + "Prediction:"
            )
            lengths.append(len(tokenizer.encode(prompt, add_special_tokens=False)))
            if number % 500 == 0:
                print(f"token_audit_progress {number}/{len(rows)}", flush=True)
        write_json(
            output / "token_audit.json",
            {
                "rows": len(rows),
                "maximum": max(lengths),
                "total": sum(lengths),
                "configured_cap": job["max_length"],
                "truncated": 0,
            },
        )
        if len(rows) != 8688 or max(lengths) > job["max_length"]:
            raise ValueError("training population or zero-truncation gate failed")
        command = training_command(job)
        if initial := config["models"][args.size]["initial_adapter"]:
            command.append(f"student.init_adapter={ROOT / initial}")
        write_json(
            output / "execution_contract.json",
            {
                "job": job,
                "command": command,
                "manifest_sha256": file_hash(DATA / "manifest.json"),
                "source_sha256": {
                    p: file_hash(ROOT / p)
                    for p in [
                        "experiments/student_injection_awareness/train.py",
                        "experiments/deception_distillation/train_student_sft.py",
                        "experiments/tool_trajectory_monitoring/run_distillation_train.py",
                        "src/gleipnir/packed_training.py",
                        "src/gleipnir/packed_sequences.py",
                    ]
                },
            },
        )
        subprocess.run(command, cwd=ROOT, env=environment(), check=True)
    metadata = json.loads((master / "training_metadata.json").read_text())
    packing = metadata["sequence_packing"]
    if metadata["training_state"]["global_step"] != 272 or not all(
        packing[k]["passed"] for k in ["eager_canary", "compiled_canary", "preflight"]
    ):
        raise ValueError("completed training or packing gates failed")
    if packing["initial_master_sha256"] == packing["final_master_sha256"]:
        raise ValueError("master adapter did not change")
    expected = config["models"][args.size].get("initial_master_sha256")
    if expected and expected != packing["initial_master_sha256"]:
        raise ValueError("initial adapter identity drift")
    rebase_adapter(master, Path(job["model_dir"]))
    write_json(
        output / "complete.json",
        {
            "steps": 272,
            "epochs": 1,
            "teacher_targets_sha256": file_hash(DATA / "soft_targets.jsonl"),
            "template_sha256": manifest["template_sha256"][args.variant],
            "status": "trained",
        },
    )
    print(f"training_complete {args.size} {args.variant}", flush=True)


if __name__ == "__main__":
    main()
