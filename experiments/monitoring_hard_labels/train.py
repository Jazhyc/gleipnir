"""Train one matched packed BF16 student and checksum-rebase its FP32 adapter."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from experiments.monitoring_hard_labels.prepare import (
    CONFIG,
    DATA,
    FRACTIONS,
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


def make_job(config: dict, recipe: dict, size: str, variant: str) -> dict:
    model = config["models"][size]
    fraction = config["hard_label_fractions"][variant]
    output = OUTPUT / size / variant
    job = {
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
        "train_rows": config["training_rows"],
        "startup_validation_reference": str(
            ROOT / config["startup_validation_reference"]
        ),
        "learning_rate": config["learning_rate"],
        "soft_loss_weight": 1.0 - fraction,
        "direct_loss_weight": fraction,
        "target": "source_hard_and_kimi_soft",
        "save_steps": 1000000,
        "gradient_checkpointing": False,
        "gradient_checkpointing_policy": "all",
        "gradient_checkpointing_layer_indices": None,
        "output_dir": str(output),
        "causal_adapter_dir": str(output / "causal_adapter"),
        "model_dir": str(output / "model"),
        "hydra_log_dir": str(ROOT / "logs/runpod/monitoring_hard_labels/hydra"),
    }
    return job


def validate_training_metadata(
    metadata: dict, job: dict, initial_hash: str, *, expected_steps: int = 272
) -> None:
    """Require completed updates, exact loss weights and the selected packed recipe."""
    from gleipnir.monitoring_systems_screen import nested_value

    profile = resolved_profile()
    reused = metadata.get("startup_validation")
    skipped_paths = {
        "sequence_packing.eager_canary.passed",
        "sequence_packing.compiled_canary.passed",
        "sequence_packing.preflight.passed",
        "gated_delta_backend.finite",
    }
    if reused:
        from gleipnir.validated_startup import (
            reused_diagnostic_view,
            validation_reference,
        )

        expected_reference = validation_reference(
            Path(job["startup_validation_reference"]),
            packed_attention_backend=job.get("packed_attention_backend", "sdpa"),
            packed_attention_version=job.get("packed_attention_version"),
            learning_gradient_tolerance=job.get("packing_learning_gradient_tolerance"),
            expected_sha256=job.get("startup_validation_reference_sha256"),
            **({"native_fp4_mlp": True} if job.get("native_fp4_mlp", False) else {}),
        )
        if reused["reference_sha256"] != expected_reference["reference_sha256"]:
            raise ValueError("reused validation identity drift")
        metadata = reused_diagnostic_view(
            {**metadata, "startup_validation": expected_reference}
        )
    for path, expected in profile["metadata_expectations"].items():
        if reused and path in skipped_paths:
            continue
        if nested_value(metadata, path) != expected:
            raise ValueError(f"systems metadata drift: {path}")
    packing = metadata["sequence_packing"]
    if metadata["training_state"]["global_step"] != expected_steps or (
        not reused
        and not all(
            packing[k]["passed"]
            for k in ("eager_canary", "compiled_canary", "preflight")
        )
    ):
        raise ValueError("completed training or packing gates failed")
    if packing["initial_master_sha256"] != initial_hash:
        raise ValueError("initial adapter identity drift")
    if packing["initial_master_sha256"] == packing["final_master_sha256"]:
        raise ValueError("master adapter did not change")
    losses = metadata["losses"]
    if (
        losses["soft_weight"] != job["soft_loss_weight"]
        or losses["direct_weight"] != job["direct_loss_weight"]
        or losses["completion_weight"] != 0
        or losses["accumulation_policy"] != "sum_per_example_over_logical_batch_v1"
    ):
        raise ValueError("objective or equal-example weighting drift")
    if (
        metadata["gradient_checkpointing"]
        or metadata["quantization"]["enabled"]
        or metadata["direct_logits_mode"] != "selected_positions"
        or metadata["optimization"]["learning_rate"] != job["learning_rate"]
        or metadata["gated_delta_backend"]["backend"] != "flashqla"
    ):
        raise ValueError("selected packed BF16 recipe drift")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", choices=("4b",), default="4b")
    parser.add_argument("--variant", choices=tuple(FRACTIONS), required=True)
    args = parser.parse_args()
    config = configuration()
    manifest = verify_preparation()
    profile = resolved_profile()
    job = make_job(config, profile["recipe"], args.size, args.variant)
    output = Path(job["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    if (output / "job.json").exists():
        if json.loads((output / "job.json").read_text()) != job:
            raise ValueError("existing training job contract drift")
    write_json(output / "job.json", job)
    master = Path(job["causal_adapter_dir"])
    if not (master / "training_metadata.json").exists():
        if (output / "token_audit.json").exists():
            audit = json.loads((output / "token_audit.json").read_text())
            if (
                audit["rows"] != 8688
                or audit["maximum"] > job["max_length"]
                or audit["truncated"]
            ):
                raise ValueError("cached token audit failed")
            print("Reusing unchanged-input token audit", flush=True)
        else:
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
                        "experiments/monitoring_hard_labels/train.py",
                        "experiments/deception_distillation/train_student_sft.py",
                        "src/gleipnir/campaigns/training_command.py",
                        "src/gleipnir/__init__.py",
                        "src/gleipnir/_compat.py",
                        "src/gleipnir/training/packed.py",
                        "src/gleipnir/training/packing.py",
                    ]
                },
            },
        )
        subprocess.run(
            command,
            cwd=ROOT,
            env=training_environment(
                ROOT,
                "monitoring_hard_labels_v1",
                native_fp4_mlp=job.get("native_fp4_mlp", False),
            ),
            check=True,
        )
    metadata = json.loads((master / "training_metadata.json").read_text())
    validate_training_metadata(
        metadata, job, config["models"][args.size]["initial_master_sha256"]
    )
    rebase_adapter(master, Path(job["model_dir"]))
    write_json(
        output / "complete.json",
        {
            "steps": 272,
            "epochs": 1,
            "teacher_targets_sha256": file_hash(DATA / "soft_targets.jsonl"),
            "template_sha256": manifest["template_sha256"][args.variant],
            "status": "trained",
            "hard_label_fraction": FRACTIONS[args.variant],
            "config_sha256": file_hash(CONFIG),
            "master_sha256": file_hash(master / "adapter_model.safetensors"),
        },
    )
    print(f"training_complete {args.size} {args.variant}", flush=True)


if __name__ == "__main__":
    main()
