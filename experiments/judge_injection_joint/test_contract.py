"""Check task quotas, masked losses, packed readouts and recipe identity."""

from collections import Counter

import pytest
import torch
import torch.nn.functional as F

from gleipnir.binary_task_training import (
    TaskMixtureSampler,
    binary_task_feature,
    binary_task_loss,
)
from gleipnir.packed_sequences import collate_packed_monitoring


def test_mixture_preserves_monitor_epoch_and_batch_quotas():
    groups = ["monitor"] * 8688
    labels = [i % 2 for i in range(8688)]
    for group in ("clean", "preferred_injected", "disfavored_injected"):
        groups.extend([group] * 2000)
        labels.extend([i % 2 for i in range(2000)])
    sampler = TaskMixtureSampler(groups, labels, seed=0)
    indices = list(sampler)
    assert len(indices) == 11584
    assert len(set(i for i in indices if groups[i] == "monitor")) == 8688
    assert Counter(groups[i] for i in indices) == {
        "monitor": 8688,
        "clean": 290,
        "preferred_injected": 1303,
        "disfavored_injected": 1303,
    }
    for start in range(0, len(indices), 32):
        assert sum(groups[i] == "monitor" for i in indices[start : start + 32]) == 24
    for group in set(groups) - {"monitor"}:
        counts = Counter(labels[i] for i in indices if groups[i] == group)
        assert abs(counts[0] - counts[1]) <= 1
    assert indices == list(sampler)
    sampler.set_epoch(1)
    assert indices != list(sampler)


def test_partial_epoch_and_pool_recycling():
    groups = ["monitor"] * 25 + [
        g
        for g in ("clean", "preferred_injected", "disfavored_injected")
        for _ in range(2)
    ]
    sampler = TaskMixtureSampler(groups, [0] * 25 + [0, 1] * 3, seed=4)
    indices = list(sampler)
    assert len(indices) == 41
    assert Counter(groups[i] for i in indices)["monitor"] == 25
    assert set(i for i in indices if groups[i] == "monitor") == set(range(25))


def test_task_losses_match_original_objectives_and_gradients():
    logits = torch.tensor([[2.0, -1.0], [-0.5, 1.2], [0.3, 0.4]], requires_grad=True)
    labels = torch.tensor([0, 1, 0])
    targets = torch.tensor([0.2, 0.0, 0.7])
    hard = torch.tensor([False, True, False])
    loss = binary_task_loss(logits, labels, targets, hard)
    expected = (
        F.binary_cross_entropy_with_logits(logits[0, 1] - logits[0, 0], targets[0])
        + F.cross_entropy(logits[1:2], labels[1:2])
        + F.binary_cross_entropy_with_logits(logits[2, 1] - logits[2, 0], targets[2])
    ) / 3
    assert torch.allclose(loss, expected)
    assert torch.allclose(
        torch.autograd.grad(loss, logits, retain_graph=True)[0],
        torch.autograd.grad(expected, logits)[0],
    )
    changed = targets.clone()
    changed[1] = 1
    assert torch.equal(loss, binary_task_loss(logits, labels, changed, hard))


class Tokenizer:
    def encode(self, value, **kwargs):
        return {"0": [15], "1": [16], "A": [32], "B": [33]}[value]


def test_surfaces_and_objectives_remain_distinct_after_packing():
    monitor = {
        "sampling_group": "monitor",
        "decision_tokens": ["0", "1"],
        "decision_prefix": "Prediction:",
        "binary_objective": "soft",
        "label": 0,
        "_soft_target": 0.25,
    }
    preference = {
        "sampling_group": "preferred_injected",
        "decision_tokens": ["A", "B"],
        "decision_prefix": "",
        "binary_objective": "hard",
        "label": 1,
    }
    features = [
        {
            **binary_task_feature(r, Tokenizer()),
            "direct_input_ids": ids,
            "binary_label": r["label"],
            "dataset_id": i,
        }
        for i, (r, ids) in enumerate([(preference, [7]), (monitor, [8, 9])])
    ]
    batch = collate_packed_monitoring(features)
    assert batch["row_decision_token_ids"].tolist() == [[32, 33], [15, 16]]
    assert batch["hard_objectives"].tolist() == [True, False]
    logits = torch.randn(2, 40, requires_grad=True)
    selected = logits.gather(-1, batch["row_decision_token_ids"])
    assert torch.equal(selected[0], logits[0, [32, 33]])
    binary_task_loss(
        selected,
        batch["binary_labels"],
        batch["soft_targets"],
        batch["hard_objectives"],
    ).backward()
    assert torch.count_nonzero(logits.grad) == 4
    with pytest.raises(ValueError, match="teacher target"):
        binary_task_feature({**preference, "_soft_target": 0.5}, Tokenizer())


def test_hydra_command_uses_regular_lr_and_fresh_adapter_contract():
    from hydra import compose, initialize_config_dir

    from experiments.judge_injection_joint.prepare import ROOT, configuration
    from experiments.judge_injection_joint.train import make_job
    from gleipnir.monitoring_training_command import training_command

    job = make_job(configuration(), {"training_draws": 11584})
    command = training_command(job)
    with initialize_config_dir(
        version_base=None,
        config_dir=str(ROOT / "experiments/tool_trajectory_monitoring"),
    ):
        cfg = compose(config_name="distillation_config", overrides=command[6:])
    assert cfg.student.training.learning_rate == 5e-5
    assert cfg.student.training.task_mixture.monitoring_per_batch == 24
    assert cfg.student.training.task_mixture.preference_per_batch == 8
    assert cfg.student.training.per_record_binary_task
    assert cfg.student.init_adapter is None


def test_actual_tokenization_preserves_each_task_prefix():
    from experiments.deception_distillation.train_student_sft import tokenize_record

    class ChatTokenizer(Tokenizer):
        def apply_chat_template(self, messages, **kwargs):
            return "chat:" + messages[0]["content"]

        def encode(self, value, **kwargs):
            if value in {"A", "B", "0", "1"}:
                return super().encode(value, **kwargs)
            return list(value.encode())

    common = {"index": "row", "student_prompt": "trace", "label": 0}
    monitor = {
        **common,
        "sampling_group": "monitor",
        "decision_tokens": ["0", "1"],
        "decision_prefix": "Prediction:",
        "binary_objective": "soft",
        "_soft_target": 0.25,
    }
    pref = {
        **common,
        "sampling_group": "clean",
        "decision_tokens": ["A", "B"],
        "decision_prefix": "",
        "binary_objective": "hard",
    }
    features = [
        tokenize_record(
            r,
            ChatTokenizer(),
            100,
            include_direct_target=True,
            include_completion_target=False,
            dataset_id=0,
            per_record_binary_task=True,
        )
        for r in (monitor, pref)
    ]
    assert bytes(features[0]["direct_input_ids"]).decode() == "chat:tracePrediction:"
    assert bytes(features[1]["direct_input_ids"]).decode() == "chat:trace"
    assert features[0]["row_decision_token_ids"] == [15, 16]
    assert features[1]["row_decision_token_ids"] == [32, 33]


def test_real_trainer_epoch_uses_draw_count_instead_of_pool_count(tmp_path):
    from datasets import Dataset
    from transformers import Trainer, TrainingArguments

    from gleipnir.adaptive_microbatching import (
        AdaptiveMicrobatchTrainerMixin,
        MicrobatchPolicy,
    )
    from gleipnir.packed_sequences import PackedSequenceLayout
    from gleipnir.training import configure_mean_loss_accumulation

    class MixtureTrainer(Trainer):
        def _get_train_sampler(self, train_dataset=None):
            dataset = self.train_dataset if train_dataset is None else train_dataset
            return TaskMixtureSampler(
                dataset["sampling_group"], dataset["binary_label"], seed=0
            )

        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            positions = PackedSequenceLayout(
                tuple(inputs["packed_lengths"])
            ).decision_positions()
            raw = model(inputs["direct_input_ids"][0, positions])
            selected = raw.gather(-1, inputs["row_decision_token_ids"])
            loss = binary_task_loss(
                selected,
                inputs["binary_labels"],
                inputs["soft_targets"],
                inputs["hard_objectives"],
            )
            return (loss, {}) if return_outputs else loss

    class AdaptiveMixture(AdaptiveMicrobatchTrainerMixin, MixtureTrainer):
        pass

    rows = []
    for group, count in [
        ("monitor", 48),
        ("clean", 80),
        ("preferred_injected", 80),
        ("disfavored_injected", 80),
    ]:
        for i in range(count):
            is_monitor = group == "monitor"
            rows.append(
                {
                    "sampling_group": group,
                    "binary_label": i % 2,
                    "direct_input_ids": [1, 2] if is_monitor else [3],
                    "row_decision_token_ids": [15, 16] if is_monitor else [32, 33],
                    "hard_objective": not is_monitor,
                    "dataset_id": 0,
                    "soft_target": (0.25 if i % 2 == 0 else 0.75)
                    if is_monitor
                    else 0.0,
                }
            )
    trainer = AdaptiveMixture(
        model=torch.nn.Embedding(128, 64),
        train_dataset=Dataset.from_list(rows),
        data_collator=collate_packed_monitoring,
        args=TrainingArguments(
            output_dir=str(tmp_path),
            use_cpu=True,
            per_device_train_batch_size=32,
            gradient_accumulation_steps=1,
            num_train_epochs=1,
            learning_rate=0.01,
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
        MicrobatchPolicy(128, 8),
        collate_packed_monitoring,
        profile=False,
        require_finite_gradients=True,
        sequence_packing=True,
    )
    trainer.train()
    assert trainer.state.global_step == 2
    assert trainer.logical_batch_sizes == [32, 32]
    assert sum(r["examples"] for r in trainer.microbatch_records) == 64
