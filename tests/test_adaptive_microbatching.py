"""Check optimizer boundaries and unequal microbatch weights with real Trainer."""

from pathlib import Path

import pytest
import torch
from transformers import Trainer, TrainerCallback, TrainingArguments

from experiments.b200_adaptive_microbatching.diagnose import diagnostic_job
from experiments.tool_trajectory_monitoring.run_distillation_train import (
    training_command,
)
from gleipnir.adaptive_microbatching import (
    AdaptiveMicrobatchTrainerMixin,
    MicrobatchPolicy,
    gradient_partition_canary,
)
from gleipnir.monitoring_systems_screen import (
    load_config,
    make_jobs,
    resolve_paths,
    validate_training_metadata,
)
from gleipnir.training import configure_mean_loss_accumulation


def test_partition_covers_examples_and_respects_actual_padding_budget():
    lengths = [30000, 20000, 9000, 8192, 4096, 4000, 2048, 1000, 500, 80, 20]
    policy = MicrobatchPolicy(8192, 4)
    batches = policy.partition(lengths)
    assert sorted(i for batch in batches for i in batch) == list(range(len(lengths)))
    assert [i for batch in batches for i in batch] == sorted(
        range(len(lengths)), key=lambda i: -lengths[i]
    )
    assert len(batches[0]) == len(batches[1]) == len(batches[2]) == 1
    assert any(len(batch) > 1 for batch in batches)
    for batch in batches:
        assert len(batch) in {1, 2, 4}
        assert len(batch) == 1 or len(batch) * max(lengths[i] for i in batch) <= 8192


@pytest.mark.parametrize("budget,size", [(0, 4), (-1, 1), (8, 0), (8, 3)])
def test_invalid_policies_fail_closed(budget, size):
    with pytest.raises(ValueError):
        MicrobatchPolicy(budget, size)


@pytest.mark.parametrize("lengths", [[], [0], [4, -1]])
def test_invalid_lengths_fail_closed(lengths):
    with pytest.raises(ValueError):
        MicrobatchPolicy(8192, 4).partition(lengths)


class ToyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.75))

    def forward(self, **kwargs):
        raise AssertionError("the trainer owns the per-example loss")


def collate(features):
    return {
        "x": torch.tensor([feature["x"] for feature in features]),
        "y": torch.tensor([feature["y"] for feature in features]),
    }


def loss_for(model, batch):
    return (model.weight * batch["x"] - batch["y"]).square().mean()


class MeanTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        loss = loss_for(model, inputs)
        return (loss, {}) if return_outputs else loss


class AdaptiveTrainer(AdaptiveMicrobatchTrainerMixin, MeanTrainer):
    pass


class CaptureGradients(TrainerCallback):
    def __init__(self):
        self.gradients = []

    def on_pre_optimizer_step(self, args, state, control, model=None, **kwargs):
        self.gradients.append(float(model.weight.grad))


def features(count):
    return [
        {
            "direct_input_ids": [1] * (1000 if i % 7 == 0 else 10 + i),
            "x": float(i + 1) / count,
            "y": float(i % 3),
        }
        for i in range(count)
    ]


def run_trainer(tmp_path, rows, maximum_size):
    model = ToyModel()
    capture = CaptureGradients()
    trainer = AdaptiveTrainer(
        model=model,
        train_dataset=rows,
        data_collator=collate,
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
        MicrobatchPolicy(256, maximum_size), collate, profile=False
    )
    output = trainer.train()
    return (
        capture.gradients,
        output.training_loss,
        trainer.adaptive_microbatch_metadata(),
    )


@pytest.mark.parametrize("count", [32, 48])
def test_real_trainer_preserves_gradients_and_partial_optimizer_batches(
    tmp_path, count
):
    rows = features(count)
    singleton, reference_loss, reference = run_trainer(tmp_path / "single", rows, 1)
    adaptive, actual_loss, metadata = run_trainer(tmp_path / "adaptive", rows, 4)
    assert metadata["logical_batch_sizes"] == ([32] if count == 32 else [32, 16])
    assert reference["logical_batch_sizes"] == metadata["logical_batch_sizes"]
    assert len(adaptive) == (1 if count == 32 else 2)
    assert adaptive == pytest.approx(singleton, abs=1e-6)
    assert actual_loss == pytest.approx(reference_loss, abs=1e-6)
    assert {1, 4} <= set(metadata["physical_microbatch_sizes"])
    assert sum(r["examples"] for r in metadata["records"]) == count
    if count == 32:
        model = ToyModel()
        loss_for(model, collate(rows)).backward()
        assert adaptive[0] == pytest.approx(float(model.weight.grad), abs=1e-6)


def test_gradient_canary_clears_gradients_and_restores_eval_mode():
    model = ToyModel().eval()
    result = gradient_partition_canary(
        model,
        features(8),
        collate,
        lambda batch: loss_for(model, batch),
        MicrobatchPolicy(256, 4),
    )
    assert result["passed"] is True
    assert result["reference_gradient_norm"] > 0
    assert result["relative_l2_error"] < 1e-6
    assert result["gradient_cosine_similarity"] == pytest.approx(1)
    assert result["actual_mean_loss"] == pytest.approx(result["reference_mean_loss"])
    assert model.weight.grad is None
    assert model.training is False


def test_eager_diagnostic_preserves_inputs_and_per_example_objective(tmp_path):
    _, original, _ = adaptive_contract()
    diagnostic = diagnostic_job(original, tmp_path, "eager")
    for key in (
        "student_rows",
        "soft_targets",
        "selection_sha256",
        "rank",
        "seed",
        "soft_loss_weight",
        "direct_loss_weight",
        "adaptive_microbatching",
    ):
        assert diagnostic[key] == original[key]
    assert diagnostic["max_steps"] == 1
    assert diagnostic["selective_torch_compile_policy"] == "none"
    assert original["selective_torch_compile_policy"] != "none"


def test_gradient_canary_rejects_nonmean_loss():
    model = ToyModel()
    result = gradient_partition_canary(
        model,
        features(8),
        collate,
        lambda batch: (model.weight * batch["x"] - batch["y"]).square().sum(),
        MicrobatchPolicy(256, 4),
    )
    assert result["passed"] is False
    assert model.weight.grad is None


def test_nonfinite_training_loss_fails_before_optimizer_update(tmp_path):
    rows = features(32)
    rows[0]["y"] = float("nan")
    with pytest.raises(FloatingPointError, match="nonfinite"):
        run_trainer(tmp_path, rows, 4)


def adaptive_contract():
    config = load_config(Path("experiments/b200_adaptive_microbatching/config.yaml"))
    jobs = make_jobs(config, resolve_paths(config), "fixed-selection")
    job = jobs[-1]
    metadata = {
        "training_batch": {
            "micro_batch_size": None,
            "gradient_accumulation_steps": None,
            "effective_batch_size": 32,
            "logical_batch_size": 32,
        },
        "adaptive_microbatching": {
            "policy": {"max_padded_tokens": 16384, "max_micro_batch_size": 8},
            "logical_batch_sizes": [32, 32],
            "records": [
                {
                    "update": update,
                    "examples": 8,
                    "logical_indices": list(range(offset, offset + 8)),
                    "max_length": 2048,
                    "tokens": 14000,
                    "padded_tokens": 16384,
                    "seconds": 1.0,
                }
                for update in (1, 2)
                for offset in range(0, 32, 8)
            ],
        },
    }
    return config, job, metadata


def test_launcher_forwards_bounded_adaptive_policy():
    config, job, _ = adaptive_contract()
    command = training_command(job)
    assert "student.training.adaptive_microbatching.enabled=true" in command
    assert "student.training.adaptive_microbatching.max_padded_tokens=16384" in command
    assert "student.training.per_device_train_batch_size=32" in command
    assert "student.training.gradient_accumulation_steps=1" in command
    assert config["maximum_unique_graphs"] == 24
    assert config["selection"]["rows"] == 320


@pytest.mark.parametrize(
    "fault", ["coverage", "duplicate", "budget", "size", "timing", "parity"]
)
def test_metadata_rejects_invalid_physical_batches(fault):
    config, job, metadata = adaptive_contract()
    record = metadata["adaptive_microbatching"]["records"][0]
    if fault == "coverage":
        record["update"] = 2  # The total is still 64; per-update coverage is wrong.
    elif fault == "duplicate":
        record["logical_indices"][0] = 8
    elif fault == "budget":
        record.update(max_length=4096, padded_tokens=32768)
    elif fault == "size":
        record["examples"] = 3
    elif fault == "timing":
        record["seconds"] = float("nan")
    with pytest.raises(ValueError, match="adaptive"):
        validate_training_metadata(
            metadata, config, job, expected_steps=2, require_canary=True
        )
