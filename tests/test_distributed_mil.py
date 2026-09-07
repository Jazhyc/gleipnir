"""Actual two-process Gloo reducer canary, including mixed and tail batches."""

import os
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.multiprocessing as mp
from torch import nn

from experiments.deception_distillation.train_student_sft import (
    forward_final_and_mil_binary_logits,
)
from gleipnir.distributed_training import (
    install_mil_forward,
    verify_distributed_parameters,
)
from gleipnir.mil import masked_mil_bce
from gleipnir.training import configure_mean_loss_accumulation


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(8, 4)

    def forward(self, input_ids, **kwargs):
        return SimpleNamespace(last_hidden_state=self.embed(input_ids))


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = Decoder()
        self.lm_head = nn.Linear(4, 3)

    def forward(self, input_ids):
        return self.lm_head(self.model(input_ids).last_hidden_state)


def loss(model, index):
    ids = torch.tensor([[index % 8, (index + 1) % 8]])
    final, mil, _ = forward_final_and_mil_binary_logits(
        model, ids, torch.ones_like(ids), torch.tensor([[0]]), torch.tensor([0, 1])
    )
    target = torch.tensor([0.2 + 0.1 * (index % 5)])
    return torch.nn.functional.binary_cross_entropy_with_logits(
        final[:, 1] - final[:, 0], target
    ) + 0.25 * masked_mil_bce(
        mil[..., 1] - mil[..., 0],
        torch.tensor([[index % 2 == 0]]),
        target,
        pool=lambda values, mask: values[:, 0],
    )


def worker(rank, rendezvous):
    torch.set_num_threads(1)
    torch.distributed.init_process_group(
        "gloo", init_method=rendezvous, rank=rank, world_size=2
    )
    try:
        torch.manual_seed(0)
        model = Model()
        reference = Model()
        reference.load_state_dict(model.state_dict())
        install_mil_forward(model, forward_final_and_mil_binary_logits)
        distributed = nn.parallel.DistributedDataParallel(
            model, find_unused_parameters=False
        )
        for count in (8, 2):
            model.zero_grad()
            reference.zero_grad()
            for i in range(count):
                (loss(reference, i) / count).backward()
            local_count = count // 2
            for i in range(local_count):
                with distributed.no_sync() if i < local_count - 1 else nullcontext():
                    (loss(distributed, i * 2 + rank) / local_count).backward()
            for actual, expected in zip(
                model.parameters(), reference.parameters(), strict=True
            ):
                torch.testing.assert_close(
                    actual.grad, expected.grad, atol=1e-7, rtol=1e-5
                )
                actual.data.add_(actual.grad, alpha=-0.01)
                expected.data.add_(expected.grad, alpha=-0.01)
            assert verify_distributed_parameters(model)["max_replica_difference"] == 0
        # Exercise the actual Trainer/Accelerate accumulation and final-window
        # logic too, not only manually normalized DDP gradients.
        from transformers import Trainer, TrainingArguments

        os.environ.update(
            LOCAL_RANK=str(rank),
            RANK=str(rank),
            WORLD_SIZE="2",
            MASTER_ADDR="127.0.0.1",
            MASTER_PORT="29591",
        )

        class TinyTrainer(Trainer):
            def compute_loss(self, model, inputs, **kwargs):
                return loss(model, int(inputs["index"].item()))

        for start, stop in ((0, 4), (4, 8), (8, 10)):
            reference.zero_grad()
            for index in range(start, stop):
                (loss(reference, index) / (stop - start)).backward()
            with torch.no_grad():
                for parameter in reference.parameters():
                    parameter.add_(parameter.grad, alpha=-0.01)
        trainer = TinyTrainer(
            model=model,
            train_dataset=[{"index": i} for i in range(10)],
            args=TrainingArguments(
                output_dir=rendezvous.removeprefix("file://") + "-trainer",
                use_cpu=True,
                report_to="none",
                save_strategy="no",
                per_device_train_batch_size=1,
                gradient_accumulation_steps=2,
                num_train_epochs=1,
                learning_rate=0.01,
                optim="sgd",
                lr_scheduler_type="constant",
                max_grad_norm=0,
                remove_unused_columns=False,
                train_sampling_strategy="sequential",
                ddp_find_unused_parameters=False,
                dataloader_pin_memory=False,
            ),
        )
        configure_mean_loss_accumulation(trainer)
        trainer.train()
        for actual, expected in zip(
            model.parameters(), reference.parameters(), strict=True
        ):
            torch.testing.assert_close(actual, expected, atol=1e-7, rtol=1e-5)
        if rank == 1:
            with torch.no_grad():
                next(model.parameters()).add_(1)
        with pytest.raises(RuntimeError, match="replicas diverged"):
            verify_distributed_parameters(model)
    finally:
        torch.distributed.destroy_process_group()


def test_ddp_selected_mil_matches_global_batch(tmp_path):
    mp.spawn(worker, args=(f"file://{tmp_path / 'rendezvous'}",), nprocs=2, join=True)


def test_launcher_preserves_per_rank_batch_contract():
    from experiments.monitoring_lr_sweep.prepare import make_jobs
    from experiments.tool_trajectory_monitoring.run_distillation_train import (
        training_command,
    )

    job = make_jobs(Path("data"), Path("results"))[0]
    single = training_command(job)
    assert "torch.distributed.run" not in single
    job.update(
        world_size=2, gradient_accumulation_steps=16, nonreentrant_checkpointing=True
    )
    command = training_command(job)
    assert command[1:5] == [
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc_per_node=2",
    ]
    assert "student.training.gradient_accumulation_steps=16" in command
    assert "++student.training.nonreentrant_checkpointing=true" in command
    with pytest.raises(ValueError):
        training_command({**job, "world_size": 3})


def test_dispatch_preserves_fp32_head_under_accelerate_autocast():
    model = Model()
    ids = torch.tensor([[1, 2]])
    mask, positions, binary = (
        torch.ones_like(ids),
        torch.tensor([[0]]),
        torch.tensor([0, 1]),
    )
    expected, _, _ = forward_final_and_mil_binary_logits(
        model, ids, mask, positions, binary
    )
    install_mil_forward(model, forward_final_and_mil_binary_logits)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        actual = model(
            input_ids=ids,
            attention_mask=mask,
            gleipnir_mil_positions=positions,
            gleipnir_binary_ids=binary,
        )
    assert actual["logits"].dtype == torch.float32
    torch.testing.assert_close(actual["logits"], expected, atol=0, rtol=0)


def test_rank_cache_seed_never_overwrites_or_recursively_copies_ranks(
    tmp_path, monkeypatch
):
    from gleipnir.distributed_training import prepare_rank_caches

    calls = []
    monkeypatch.setenv("TRITON_CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(
        "gleipnir.distributed_training.subprocess.run",
        lambda command, **kwargs: calls.append(command),
    )
    prepare_rank_caches(1)
    assert os.environ["TRITON_CACHE_DIR"] == str(tmp_path / "rank-1")
    assert calls[0][1:4] == ["-a", "--ignore-existing", "--exclude=rank-*"]
