"""Verify ordinary packed Trainer updates and scoped startup settings."""

from copy import deepcopy

import pytest
import torch
from transformers import Trainer, TrainerCallback, TrainingArguments

from gleipnir.adaptive_microbatching import (
    AdaptiveMicrobatchTrainerMixin,
    MicrobatchPolicy,
    gradient_partition_canary,
)
from gleipnir.packed_sequences import collate_packed_monitoring, packed_partition
from gleipnir.packed_training import (
    packed_memory_preflight,
    packed_training_runtime,
    validate_packed_training_config,
)
from gleipnir.training import configure_mean_loss_accumulation


def student_config():
    return {
        "quantization": {
            "enabled": False,
            "full_bf16_lora": True,
            "mlp_precision": "bf16",
        },
        "model_loader": "causal_lm",
        "finetuning_mode": "lora",
        "attn_implementation": "sdpa",
        "lora": {"dropout": 0},
        "training": {
            "sequence_packing": True,
            "gated_delta_backend": "flashqla",
            "adaptive_microbatching": {"enabled": True, "max_padded_tokens": 256},
            "selective_torch_compile_policy": "full_attention_and_linear_shell",
            "selective_torch_compile_canary_tokens": 256,
        },
    }


@pytest.mark.parametrize("fail", [False, True])
def test_runtime_restores_attention_compiler_and_matmul_settings(fail):
    from torch._inductor import config as inductor
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    original = (
        ALL_ATTENTION_FUNCTIONS["sdpa"],
        torch._dynamo.config.recompile_limit,
        torch._dynamo.config.fail_on_recompile_limit_hit,
        inductor.emulate_precision_casts,
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction_split_k,
        torch.backends.cuda.preferred_blas_library(),
    )
    try:
        with packed_training_runtime(student_config()) as metadata:
            assert metadata["enabled"] and metadata["max_packed_tokens"] == 256
            assert metadata["bf16_matmul"] == {
                "blas_library": "cublaslt",
                "allow_reduced_precision": False,
                "allow_split_k": False,
            }
            assert torch._dynamo.config.recompile_limit == 64
            assert torch._dynamo.config.fail_on_recompile_limit_hit
            assert inductor.emulate_precision_casts
            assert ALL_ATTENTION_FUNCTIONS["sdpa"] is not original[0]
            if fail:
                raise RuntimeError("startup failure")
    except RuntimeError:
        assert fail
    assert (
        ALL_ATTENTION_FUNCTIONS["sdpa"],
        torch._dynamo.config.recompile_limit,
        torch._dynamo.config.fail_on_recompile_limit_hit,
        inductor.emulate_precision_casts,
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction_split_k,
        torch.backends.cuda.preferred_blas_library(),
    ) == original


@pytest.mark.parametrize(
    "field,value",
    [
        ("quantization.enabled", True),
        ("quantization.full_bf16_lora", False),
        ("training.gated_delta_backend", "fla"),
        ("attn_implementation", "eager"),
        ("training.selective_torch_compile_canary_tokens", 0),
        ("lora.dropout", 0.1),
        ("training.mil_loss_weight", 1),
        ("training.packing_compile_cache_limit", 129),
    ],
)
def test_unsupported_packing_stops_before_model_loading(field, value):
    config = student_config()
    target = config
    components = field.split(".")
    for part in components[:-1]:
        target = target[part]
    target[components[-1]] = value
    with pytest.raises(ValueError):
        validate_packed_training_config(config)


class ToyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.2))


def loss_for(model, batch):
    return (
        (model.weight * batch["soft_targets"] - batch["binary_labels"]).square().mean()
    )


class MeanTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        loss = loss_for(model, inputs)
        return (loss, {}) if return_outputs else loss


class PackedTrainer(AdaptiveMicrobatchTrainerMixin, MeanTrainer):
    pass


class Capture(TrainerCallback):
    def __init__(self):
        self.gradients = []

    def on_pre_optimizer_step(self, args, state, control, model=None, **kwargs):
        self.gradients.append(float(model.weight.grad))


def features(count):
    return [
        {
            "direct_input_ids": [1] * (300 if i % 9 == 0 else 10),
            "binary_label": i % 2,
            "dataset_id": 0,
            "soft_target": (i + 1) / count,
        }
        for i in range(count)
    ]


def run(tmp_path, rows, packed):
    capture = Capture()
    trainer = PackedTrainer(
        model=ToyModel(),
        train_dataset=rows,
        data_collator=collate_packed_monitoring,
        callbacks=[capture],
        args=TrainingArguments(
            output_dir=str(tmp_path),
            use_cpu=True,
            per_device_train_batch_size=32,
            gradient_accumulation_steps=1,
            num_train_epochs=1,
            learning_rate=0,
            max_grad_norm=0,
            report_to="none",
            save_strategy="no",
            disable_tqdm=True,
            remove_unused_columns=False,
            seed=0,
            data_seed=0,
        ),
    )
    configure_mean_loss_accumulation(trainer)
    trainer.enable_adaptive_microbatching(
        MicrobatchPolicy(256, 1),
        collate_packed_monitoring,
        profile=False,
        require_finite_gradients=True,
        sequence_packing=packed,
    )
    trainer.train()
    return capture.gradients, trainer.adaptive_microbatch_metadata()


@pytest.mark.parametrize("count", [32, 45])
def test_packed_optimizer_matches_singletons_including_partial_tail(tmp_path, count):
    rows = features(count)
    reference, _ = run(tmp_path / "singletons", deepcopy(rows), False)
    actual, metadata = run(tmp_path / "packed", deepcopy(rows), True)
    assert actual == pytest.approx(reference, abs=1e-6)
    assert metadata["logical_batch_sizes"] == ([32] if count == 32 else [32, 13])
    assert metadata["sequence_packing"]
    assert any(r["examples"] > 8 for r in metadata["records"])
    assert all(r["padded_tokens"] == r["tokens"] for r in metadata["records"])
    for update, size in enumerate(metadata["logical_batch_sizes"], 1):
        indices = [
            i
            for r in metadata["records"]
            if r["update"] == update
            for i in r["logical_indices"]
        ]
        assert sorted(indices) == list(range(size))


def test_packed_gradient_canary_and_memory_preflight_preserve_masters():
    model = ToyModel().eval()
    rows = features(8)
    initial = model.weight.detach().clone()

    def forward(batch):
        return loss_for(model, batch)

    receipt = gradient_partition_canary(
        model,
        rows,
        collate_packed_monitoring,
        forward,
        MicrobatchPolicy(256, 1),
        candidate_collator=collate_packed_monitoring,
        partition_strategy=lambda lengths: packed_partition(lengths, 256),
    )
    assert receipt["passed"] and receipt["finite"]
    preflight = packed_memory_preflight(
        model, rows, forward, token_budget=256, logical_batch_size=4
    )
    assert (
        preflight["passed"]
        and preflight["examples"] == 4
        and preflight["max_length"] == 300
    )
    assert model.weight.grad is None and not model.training
    assert torch.equal(initial, model.weight)
    with pytest.raises(FloatingPointError):
        packed_memory_preflight(
            model,
            rows,
            lambda batch: forward(batch) * float("nan"),
            token_budget=256,
            logical_batch_size=4,
        )
    assert model.weight.grad is None and not model.training


def test_smoke_reuses_frozen_inputs_and_runs_only_the_selected_recipe():
    from pathlib import Path

    import yaml

    from experiments.monitoring_sequence_packing.default_recipe_smoke import smoke_job

    config = yaml.safe_load(
        Path(
            "experiments/monitoring_sequence_packing/default_recipe_smoke.yaml"
        ).read_text()
    )
    profile = yaml.safe_load(Path(config["profile"]).read_text())
    source = {
        "student_rows": "frozen_rows",
        "selection_sha256": "frozen",
        "soft_targets_sha256": "targets",
        "train_rows": 320,
        "seed": 0,
    }
    original = deepcopy(source)
    job = smoke_job(config, source, profile["recipe"])
    assert source == original
    assert all(job[k] == source[k] for k in source)
    assert job["max_steps"] == 2 and job["sequence_packing"]
    assert job["full_bf16_lora"] and not job["gradient_checkpointing"]
    assert job["adaptive_microbatching"]["profile"] is False
    assert job["adaptive_microbatching"]["max_padded_tokens"] == 16384
    assert job["micro_batch_size"] == 32 and job["gradient_accumulation_steps"] == 1
