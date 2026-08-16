"""Frozen design and validation for the structural training-procedure screen."""

from __future__ import annotations

from collections import Counter
from typing import Any

BASELINE_RECIPE: dict[str, Any] = {
    "quantization_enabled": True,
    "lora_init": "default",
    "lora_use_dora": False,
    "lora_target_modules": [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    "soft_loss_weight": 1.0,
    "direct_loss_weight": 0.0,
    "soft_loss_type": "bce",
    "soft_huber_delta": 1.0,
    "dataset_loss_weighting": "mean",
    "group_dro_eta": 2.0,
    "group_dro_ema": 0.9,
    "micro_batch_size": 8,
    "gradient_accumulation_steps": 4,
}


def screen_variants() -> list[dict[str, Any]]:
    """Return the frozen single-factor cells, including three baseline seeds."""
    variants: list[dict[str, Any]] = [
        {
            "job_name": f"baseline-seed{seed}",
            "intervention_family": "baseline",
            "seed": seed,
        }
        for seed in (0, 1, 2)
    ]
    variants.extend(
        [
            {
                "job_name": "base-bf16-lora",
                "intervention_family": "base_precision",
                "quantization_enabled": False,
            },
            {
                "job_name": "init-loftq",
                "intervention_family": "adapter_initialization",
                "lora_init": "loftq",
                "evaluation_base": "nf4",
                "promotion_eligible": False,
            },
            {
                "job_name": "adapter-dora",
                "intervention_family": "adapter_parameterization",
                "lora_use_dora": True,
                "serving_backend": "transformers_only",
            },
            {
                "job_name": "init-eva",
                "intervention_family": "adapter_initialization",
                "lora_init": "eva",
                "eva_rho": 2.0,
                "eva_tau": 0.99,
                "eva_rows": 256,
            },
            {
                "job_name": "targets-with-lm-head",
                "intervention_family": "adapter_targets",
                "lora_target_modules": [
                    *BASELINE_RECIPE["lora_target_modules"],
                    "lm_head",
                ],
                "serving_backend": "transformers_canary_required",
            },
            {
                "job_name": "targets-attention-only",
                "intervention_family": "adapter_targets",
                "lora_target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
            },
            {
                "job_name": "targets-mlp-only",
                "intervention_family": "adapter_targets",
                "lora_target_modules": ["gate_proj", "up_proj", "down_proj"],
            },
            {
                "job_name": "decision-token-rows",
                "intervention_family": "decision_head",
                "decision_head_mode": "binary_head",
                "decision_head_init": "token_rows",
                "serving_backend": "transformers_only",
            },
            {
                "job_name": "decision-binary-head",
                "intervention_family": "decision_head",
                "decision_head_mode": "binary_head",
                "decision_head_init": "random",
                "serving_backend": "transformers_only",
            },
            {
                "job_name": "hard-anchor-010",
                "intervention_family": "loss",
                "direct_loss_weight": 0.1,
            },
            {
                "job_name": "hard-anchor-025",
                "intervention_family": "loss",
                "direct_loss_weight": 0.25,
            },
            {
                "job_name": "huber-delta-050",
                "intervention_family": "loss",
                "soft_loss_type": "huber",
                "soft_huber_delta": 0.5,
            },
            {
                "job_name": "huber-delta-100",
                "intervention_family": "loss",
                "soft_loss_type": "huber",
                "soft_huber_delta": 1.0,
            },
            {
                "job_name": "group-dro",
                "intervention_family": "loss_weighting",
                "dataset_loss_weighting": "group_dro",
                "group_dro_eta": 2.0,
                "group_dro_ema": 0.9,
            },
            {
                "job_name": "effective-batch-8",
                "intervention_family": "effective_batch",
                "gradient_accumulation_steps": 1,
            },
            {
                "job_name": "effective-batch-16",
                "intervention_family": "effective_batch",
                "gradient_accumulation_steps": 2,
            },
            {
                "job_name": "effective-batch-64",
                "intervention_family": "effective_batch",
                "gradient_accumulation_steps": 8,
            },
        ]
    )
    return variants


def completed_variant(variant: dict[str, Any]) -> dict[str, Any]:
    """Overlay a variant on the frozen baseline defaults."""
    return {**BASELINE_RECIPE, "seed": 0, "promotion_eligible": True, **variant}


def hard_anchor_replication_variants() -> list[dict[str, Any]]:
    """Return the frozen seed replications for the two hard-anchor candidates."""
    return [
        {
            "job_name": f"hard-anchor-{weight_name}-seed{seed}",
            "intervention_family": "hard_anchor_replication",
            "seed": seed,
            "direct_loss_weight": weight,
            "source_job_name": f"hard-anchor-{weight_name}",
        }
        for weight_name, weight in (("010", 0.1), ("025", 0.25))
        for seed in (1, 2)
    ]


def validate_hard_anchor_replication_jobs(jobs: list[dict[str, Any]]) -> None:
    """Fail closed if the hard-anchor seed replication drifts."""
    expected_variants = hard_anchor_replication_variants()
    expected_names = [variant["job_name"] for variant in expected_variants]
    names = [str(job["job_name"]) for job in jobs]
    if names != expected_names or len(set(names)) != 4:
        raise ValueError("hard-anchor replication jobs differ from the frozen design")
    expected_by_name = {
        str(variant["job_name"]): completed_variant(variant)
        for variant in expected_variants
    }
    for job in jobs:
        expected = expected_by_name[str(job["job_name"])]
        for key in (
            "seed",
            "direct_loss_weight",
            "source_job_name",
            "micro_batch_size",
            "gradient_accumulation_steps",
            "quantization_enabled",
            "lora_init",
            "lora_use_dora",
            "lora_target_modules",
            "soft_loss_type",
            "dataset_loss_weighting",
        ):
            if job[key] != expected[key]:
                raise ValueError(
                    f"hard-anchor replication drift for {job['job_name']}: {key}"
                )
        if int(job["effective_batch_size"]) != 32:
            raise ValueError("hard-anchor replication requires effective batch 32")
    invariants = (
        "rank",
        "lora_alpha",
        "optimizer",
        "learning_rate",
        "lr_scheduler_type",
        "warmup_ratio",
        "num_train_epochs",
        "train_rows",
        "selection_sha256",
    )
    for key in invariants:
        if len({str(job[key]) for job in jobs}) != 1:
            raise ValueError(f"replication invariant differs across jobs: {key}")


def validate_screen_jobs(jobs: list[dict[str, Any]]) -> None:
    """Fail closed if the prepared jobs drift from the frozen design."""
    expected = [variant["job_name"] for variant in screen_variants()]
    names = [str(job["job_name"]) for job in jobs]
    if names != expected:
        raise ValueError("training-procedure jobs differ from the frozen design")
    if len(names) != 20 or len(set(names)) != len(names):
        raise ValueError("expected 20 unique training-procedure cells")
    baseline_seeds = sorted(
        int(job["seed"])
        for job in jobs
        if job["intervention_family"] == "baseline"
    )
    if baseline_seeds != [0, 1, 2]:
        raise ValueError("baseline cells must use seeds 0, 1, and 2")
    invariants = (
        "rank",
        "lora_alpha",
        "optimizer",
        "learning_rate",
        "lr_scheduler_type",
        "warmup_ratio",
        "num_train_epochs",
        "train_rows",
        "selection_sha256",
    )
    for key in invariants:
        if len({str(job[key]) for job in jobs}) != 1:
            raise ValueError(f"screen invariant differs across jobs: {key}")
    families = Counter(str(job["intervention_family"]) for job in jobs)
    if families != Counter(
        {
            "baseline": 3,
            "base_precision": 1,
            "adapter_initialization": 2,
            "adapter_parameterization": 1,
            "adapter_targets": 3,
            "decision_head": 2,
            "loss": 4,
            "loss_weighting": 1,
            "effective_batch": 3,
        }
    ):
        raise ValueError(f"unexpected intervention families: {families}")
    for job in jobs:
        expected_effective = int(job["micro_batch_size"]) * int(
            job["gradient_accumulation_steps"]
        )
        if int(job["effective_batch_size"]) != expected_effective:
            raise ValueError(f"effective batch mismatch for {job['job_name']}")
        if job["lora_init"] not in {"default", "loftq", "eva"}:
            raise ValueError(f"unsupported LoRA initialization: {job['lora_init']}")
        if job["dataset_loss_weighting"] not in {"mean", "group_dro"}:
            raise ValueError("unsupported dataset loss weighting")


def balanced_lanes(
    jobs: list[dict[str, Any]], lanes: int = 2
) -> list[list[dict[str, Any]]]:
    """Balance estimated training load while separating baseline seeds."""
    if lanes != 2:
        raise ValueError("the frozen campaign requires two lanes")

    def load(job: dict[str, Any]) -> float:
        multiplier = 1.0
        if not bool(job["quantization_enabled"]):
            multiplier = 1.25
        elif job["lora_init"] in {"loftq", "eva"}:
            multiplier = 1.15
        return float(job["train_rows"]) * float(job["num_train_epochs"]) * multiplier

    output: list[list[dict[str, Any]]] = [[], []]
    totals = [0.0, 0.0]
    for job in sorted(jobs, key=load, reverse=True):
        lane = min(range(lanes), key=lambda index: totals[index])
        output[lane].append(job)
        totals[lane] += load(job)
    return output
