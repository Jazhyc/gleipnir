"""Check matched optimizer state, numerical comparisons, and audit restoration."""

import json

import pytest
import torch
from transformers import Trainer, TrainingArguments

from gleipnir.training_execution_audit import (
    compare_tensors,
    run_execution_audit,
    use_forwards,
)


def test_comparisons_measure_direction_scale_zeros_and_missing_gradients():
    x = [torch.tensor([1.0, -2.0])]
    same = compare_tensors(x, x)
    assert same["relative_l2_error"] == 0
    assert same["cosine"] == pytest.approx(1)
    scaled = compare_tensors(x, [2 * x[0]])
    assert scaled["relative_l2_error"] == pytest.approx(1)
    assert scaled["cosine"] == pytest.approx(1)
    opposite = compare_tensors(x, [-x[0]])
    assert opposite["cosine"] == pytest.approx(-1)
    assert opposite["sign_disagreement_fraction_nonzero"] == 1
    assert compare_tensors([torch.zeros(2)], [torch.zeros(2)])["relative_l2_error"] == 0
    with pytest.raises(ValueError, match="participation"):
        compare_tensors([None], [torch.zeros(2)])
    with pytest.raises(FloatingPointError):
        compare_tensors(x, [torch.tensor([float("nan"), 0])])


def test_forward_switch_restores_after_failure():
    model = torch.nn.Linear(1, 1)
    before = model.forward
    with pytest.raises(RuntimeError):
        with use_forwards([(model, lambda x: x * 2)]):
            assert float(model(torch.ones(1))) == 2
            raise RuntimeError("stop")
    assert model.forward == before


@pytest.mark.parametrize("fail", [False, True])
def test_audit_resets_weights_optimizer_state_and_schedule(tmp_path, fail):
    model = torch.nn.Linear(1, 1, bias=False).eval()
    with torch.no_grad():
        model.weight.fill_(0.75)
    before = model.weight.detach().clone()
    features = [
        {"direct_input_ids": list(range(1 + i % 8)), "x": (i + 1) / 32, "y": 0.2}
        for i in range(32)
    ]

    def collate(items):
        return {
            "x": torch.tensor([[i["x"]] for i in items]),
            "y": torch.tensor([[i["y"]] for i in items]),
        }

    def loss(batch):
        value = (model(batch["x"]) - batch["y"]).square().mean()
        return value * float("nan") if fail else value

    original = model.forward

    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=str(tmp_path / "trainer"),
            use_cpu=True,
            report_to="none",
            learning_rate=0.01,
            weight_decay=0.0,
            optim="adamw_torch",
            warmup_steps=0.03,
            lr_scheduler_type="linear",
            max_steps=2,
        ),
    )

    def optimizer_factory():
        trainer.optimizer = None
        return trainer.create_optimizer()

    def scheduler_factory(optimizer, steps):
        trainer.lr_scheduler = None
        return trainer.create_scheduler(steps, optimizer=optimizer)

    def compile_model():
        model.forward = torch.compile(original, backend="eager", dynamic=True)
        return [0]

    kwargs = dict(
        model=model,
        features=features,
        collator=collate,
        loss_forward=loss,
        optimizer_factory=optimizer_factory,
        scheduler_factory=scheduler_factory,
        install_compile=compile_model,
        output=tmp_path,
        seed=0,
        steps=2,
        logical_batch_size=16,
    )
    if fail:
        with pytest.raises(FloatingPointError):
            run_execution_audit(**kwargs)
        report = json.loads((tmp_path / "execution_audit.json").read_text())
        assert report["status"] == "failed"
    else:
        report = run_execution_audit(**kwargs)
        assert report["status"] == "completed"
        assert len(report["probes"]) == len(report["trajectories"]) == 4
        for name, probe in report["probes"].items():
            assert probe["update_norm"] > 0
            assert probe["initial_master_sha256"] == report["initial_master_sha256"]
            assert probe["initial_optimizer_state_entries"] == 0
            if name.startswith("compiled"):
                assert (
                    probe["eager_compiled_update_comparison"]["relative_l2_error"]
                    < 1e-5
                )
        for result in report["trajectories"].values():
            steps = result["steps"]
            assert len(steps) == 2
            assert all(lr == 0 for lr in steps[0]["lr"])
            assert all(lr == 0.01 for lr in steps[1]["lr"])
            assert steps[0]["common_probe"] == result["baseline_common_loss"]
            assert sorted(i for step in steps for i in step["logical_indices"]) == list(
                range(32)
            )
            assert result["initial_master_sha256"] == report["initial_master_sha256"]
        assert len(list(tmp_path.glob("*_fp32_master.pt"))) == 4
    assert torch.equal(model.weight, before)
    assert model.training is False
    assert model.weight.grad is None
    assert model.forward == original
