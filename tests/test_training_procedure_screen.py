import pandas as pd

from experiments.adapter_capacity_scaling.run_train import training_command
from experiments.training_procedure_screen.core import (
    balanced_lanes,
    completed_variant,
    hard_anchor_replication_variants,
    screen_variants,
    validate_hard_anchor_replication_jobs,
    validate_screen_jobs,
)
from experiments.training_procedure_screen.summarize import (
    crossfit_calibration,
    logit_average,
)
from experiments.training_procedure_screen.summarize_hard_anchor_replication import (
    aggregate_replication,
)


def jobs() -> list[dict[str, object]]:
    output = []
    for variant in screen_variants():
        job = {
            **completed_variant(variant),
            "capacity_kind": "lora",
            "rank": 128,
            "lora_alpha": 256,
            "optimizer": "adamw",
            "learning_rate": 5e-5,
            "lr_scheduler_type": "linear",
            "warmup_ratio": 0.03,
            "num_train_epochs": 2.0,
            "lora_dropout": 0.0,
            "weight_decay": 0.0,
            "dataset_sampling": "proportional",
            "soft_target_logit_scale": 1.0,
            "save_strategy": "steps",
            "save_steps": 175,
            "save_total_limit": 6,
            "save_only_model": True,
            "train_rows": 11177,
            "selection_sha256": "selection",
            "selection_manifest": "/data/selection.jsonl",
            "student_rows": "/data/student.jsonl",
            "soft_targets": "/data/soft.jsonl",
            "output_dir": "/results/run",
            "causal_adapter_dir": "/results/run/causal_adapter",
            "model_dir": "/results/run/model",
            "effective_batch_size": int(completed_variant(variant)["micro_batch_size"])
            * int(completed_variant(variant)["gradient_accumulation_steps"]),
            "config_path": "../training_procedure_screen",
            "config_name": "config",
        }
        output.append(job)
    return output


def test_frozen_training_procedure_design() -> None:
    prepared = jobs()
    validate_screen_jobs(prepared)
    assert len(prepared) == 20
    assert [job["seed"] for job in prepared[:3]] == [0, 1, 2]
    lanes = balanced_lanes(prepared)
    assert sorted(job["job_name"] for lane in lanes for job in lane) == sorted(
        job["job_name"] for job in prepared
    )


def test_training_command_carries_structural_overrides() -> None:
    job = next(job for job in jobs() if job["job_name"] == "targets-with-lm-head")
    command = training_command(job, distributed_processes=1)
    assert "--config-path" in command
    assert "../training_procedure_screen" in command
    assert (
        "student.lora.target_modules="
        "[q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj,lm_head]"
    ) in command
    assert "student.training.dataset_loss_weighting=mean" in command

    dora = next(job for job in jobs() if job["job_name"] == "adapter-dora")
    dora_command = training_command(dora, distributed_processes=1)
    assert "student.lora.use_dora=true" in dora_command

    head = next(job for job in jobs() if job["job_name"] == "decision-binary-head")
    head_command = training_command(head, distributed_processes=1)
    assert "student.training.decision_head_mode=binary_head" in head_command
    assert "student.training.decision_head_init=random" in head_command


def test_frozen_hard_anchor_replication_design() -> None:
    source = {str(job["job_name"]): job for job in jobs()}
    replication = []
    for variant in hard_anchor_replication_variants():
        replication.append({**source[str(variant["source_job_name"])], **variant})
    validate_hard_anchor_replication_jobs(replication)
    assert [job["job_name"] for job in replication] == [
        "hard-anchor-010-seed1",
        "hard-anchor-010-seed2",
        "hard-anchor-025-seed1",
        "hard-anchor-025-seed2",
    ]


def test_hard_anchor_replication_uses_paired_seed_differences() -> None:
    rows = []
    for weight, improvement in ((0.1, 0.01), (0.25, 0.02)):
        for seed in (0, 1, 2):
            rows.append(
                {
                    "direct_loss_weight": weight,
                    "seed": seed,
                    "macro_auroc": 0.9 + improvement,
                    "macro_balanced_accuracy": 0.8,
                    "macro_brier": 0.1 - improvement,
                    "paired_difference_macro_auroc": improvement,
                    "paired_difference_macro_balanced_accuracy": 0.0,
                    "paired_difference_macro_brier": -improvement,
                }
            )
    summary = aggregate_replication(pd.DataFrame(rows))
    assert [row["direct_loss_weight"] for row in summary] == [0.1, 0.25]
    assert summary[1]["mean_paired_difference_macro_auroc"] == 0.02
    assert summary[1]["passes_mean_metric_constraints"] is True


def test_logit_ensemble_and_crossfit_calibration() -> None:
    records = []
    for dataset in ("a", "b"):
        for index in range(20):
            label = index % 2
            records.append(
                {
                    "dataset": dataset,
                    "index": index,
                    "label": label,
                    "score": 0.8 if label else 0.2,
                }
            )
    first = (
        pd.DataFrame(records)
        .sort_values(["dataset", "index"])
        .reset_index(drop=True)
    )
    second = first.assign(score=lambda frame: frame["score"] * 0.9 + 0.05)
    ensemble = logit_average([first, second])
    assert len(ensemble) == len(first)
    assert 0.2 < ensemble.loc[0, "score"] < 0.23
    calibration = crossfit_calibration(ensemble)
    assert calibration["threshold_macro_balanced_accuracy"] == 1.0
    assert calibration["platt_macro_brier"] < 0.01
