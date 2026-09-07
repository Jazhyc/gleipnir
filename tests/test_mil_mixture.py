import json
from functools import partial
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from experiments.deception_distillation.train_student_sft import (
    CompletionOnlyCollator,
    pool_mil_margins,
    tokenize_record,
)
from experiments.monitoring_mil_mixture.run import select_mixture
from gleipnir.mil import masked_mil_bce, mil_row_enabled

POOL = partial(pool_mil_margins, mode="logmeanexp", temperature=1.0, top_k=3)


def test_distributed_mixture_keeps_global_batch_and_epochs(monkeypatch):
    from omegaconf import OmegaConf

    from experiments.monitoring_lr_sweep.prepare import make_jobs
    from experiments.monitoring_mil_mixture import run

    config = OmegaConf.to_container(
        OmegaConf.load("experiments/monitoring_mil_mixture/config.yaml")
    )
    source = make_jobs(Path("data"), Path("results"))[0]
    source.update(
        job_name=config["source_job"],
        mil_loss_weight=0.25,
        mil_pooling="logmeanexp",
        num_train_epochs=3,
    )
    monkeypatch.setattr(run, "read_jsonl", lambda path: [source])
    monkeypatch.setattr(run, "sha256_file", lambda path: "frozen")
    config.update(
        world_size=2,
        nonreentrant_checkpointing=True,
        selective_torch_compile_policy="none",
    )
    job = run.make_job(config)
    assert (
        job["world_size"] * job["micro_batch_size"] * job["gradient_accumulation_steps"]
        == 32
    )
    assert job["expected_steps"] == 1398
    assert job["train_rows"] == 14887
    assert job["num_train_epochs"] == 3
    assert job["learning_rate"] == 2e-5


def test_slow_distributed_screen_cannot_be_promoted(tmp_path, monkeypatch):
    from experiments.monitoring_mil_mixture import run

    screen = tmp_path / "screen"
    screen.mkdir()
    (screen / "status.json").write_text(json.dumps({"state": "complete"}))
    jobs = []
    for name, world, steps, timing in (
        ("single", 1, 8, 20),
        ("ddp", 2, 8, 30),
        ("long", 2, 1, 100),
    ):
        output = screen / name
        output.mkdir()
        jobs.append(
            dict(
                job_name=name,
                world_size=world,
                expected_steps=steps,
                causal_adapter_dir=str(output),
                student_rows_sha256="matched",
            )
        )
        (output / "training_metadata.json").write_text(
            json.dumps(
                {
                    "mil_population": {
                        "enabled_rows": 16 if steps == 1 else 30,
                        "disabled_rows": 16 if steps == 1 else 226,
                    },
                    "train_metrics": {"train_loss": 0.6},
                    "distributed_training": {"mil_projection_autocast": False},
                    "optimizer_step_timing": {"steady_mean_seconds": timing},
                }
            )
        )
    (screen / "jobs.jsonl").write_text("\n".join(map(json.dumps, jobs)))
    monkeypatch.setattr(run, "validate_mil", lambda *args, **kwargs: None)
    with pytest.raises(ValueError, match="practical speed gate"):
        run.promote_distributed_screen(screen, tmp_path / "production")
    assert not (tmp_path / "production").exists()


def test_disabled_rows_have_exactly_zero_mil_gradient():
    margins = torch.tensor([[1.0, 2.0], [5.0, 9.0]], requires_grad=True)
    mask = torch.tensor([[True, True], [False, False]])
    targets = torch.tensor([0.7, 0.2])
    actual = masked_mil_bce(margins, mask, targets, pool=POOL)
    expected = (
        F.binary_cross_entropy_with_logits(POOL(margins[:1], mask[:1]), targets[:1]) / 2
    )
    torch.testing.assert_close(actual, expected)
    actual.backward()
    assert torch.count_nonzero(margins.grad[1]) == 0
    assert torch.count_nonzero(margins.grad[0]) == 2


def test_all_disabled_batch_is_finite_zero():
    margins = torch.tensor([[3.0], [4.0]], requires_grad=True)
    loss = masked_mil_bce(
        margins,
        torch.zeros_like(margins, dtype=torch.bool),
        torch.tensor([0.1, 0.9]),
        pool=POOL,
    )
    assert loss.item() == 0
    loss.backward()
    assert torch.count_nonzero(margins.grad) == 0


def test_all_enabled_matches_old_loss_and_accumulation():
    margins = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    mask = torch.ones_like(margins, dtype=torch.bool)
    targets = torch.tensor([0.2, 0.8])
    actual = masked_mil_bce(margins, mask, targets, pool=POOL)
    old = F.binary_cross_entropy_with_logits(POOL(margins, mask), targets)
    torch.testing.assert_close(actual, old)
    mixed_mask = torch.tensor([[True, True], [False, False]])
    batch = masked_mil_bce(margins, mixed_mask, targets, pool=POOL)
    accumulated = (
        sum(
            masked_mil_bce(
                margins[i : i + 1], mixed_mask[i : i + 1], targets[i : i + 1], pool=POOL
            )
            for i in range(2)
        )
        / 2
    )
    torch.testing.assert_close(batch, accumulated)


def test_eligibility_is_explicit_and_backward_compatible():
    assert mil_row_enabled({})
    assert not mil_row_enabled({"mil_enabled": False})
    with pytest.raises(ValueError):
        mil_row_enabled({"mil_enabled": "false"})


def test_collator_accepts_empty_and_mixed_bags():
    feature = {
        "direct_input_ids": [1, 2],
        "binary_label": 1,
        "dataset_id": 0,
        "soft_target": 0.7,
        "mil_positions": [],
    }
    collator = CompletionOnlyCollator(0)
    batch = collator([feature])
    assert batch["mil_positions"].shape == (1, 1)
    assert not batch["mil_position_mask"].any()
    batch = collator([feature, {**feature, "mil_positions": [1]}])
    assert batch["mil_position_mask"].tolist() == [[False], [True]]


def test_disabled_tokenization_skips_action_parser(monkeypatch):
    from experiments.deception_distillation import train_student_sft as trainer

    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            return messages[0]["content"]

        def encode(self, text, **kwargs):
            return list(range(len(text)))

    monkeypatch.setattr(
        trainer,
        "mil_token_positions",
        lambda *a, **k: pytest.fail("deception must not parse actions"),
    )
    row = {
        "index": "d",
        "label": 1,
        "student_prompt": "plain deception evidence",
        "student_target": "unused",
        "soft_target": 0.8,
        "mil_enabled": False,
    }
    kwargs = {
        "include_direct_target": True,
        "include_completion_target": False,
        "dataset_id": 0,
    }
    disabled = tokenize_record(row, Tokenizer(), 100, include_mil_target=True, **kwargs)
    standard = tokenize_record(row, Tokenizer(), 100, **kwargs)
    assert disabled.pop("mil_positions") == []
    assert disabled == standard


def test_selection_keeps_all_deception_and_exact_monitoring():
    rows = [
        {
            "dataset": "tool_trajectory/test",
            "index": i,
            "label": 1,
            "trajectory_sha256": str(i),
            "lineage_group": str(i),
            "student_direct_tokens": 10,
        }
        for i in range(2)
    ]
    rows.append({"dataset": "deception/test", "index": 3, "label": 0})
    targets = [{**r, "soft_target": 0.5} for r in rows]
    selected, soft = select_mixture(rows, targets, [rows[1]])
    assert [(r["index"], r["mil_enabled"]) for r in selected] == [(1, True), (3, False)]
    assert soft == targets[1:]
    with pytest.raises(ValueError, match="provenance"):
        select_mixture(rows, targets, [{**rows[1], "label": 0}])
