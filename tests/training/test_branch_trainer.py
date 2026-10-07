import pytest
import torch

from gleipnir.branch_trainer import accumulation_windows, parent_objective


def test_parent_objective_averages_losses_not_targets():
    logits = torch.tensor([[0.0, -2.0], [0.0, 3.0], [0.0, 1.0]], requires_grad=True)
    targets = [0.1, 0.8, 0.6]
    loss, full, prefix = parent_objective(logits, targets, 0.1)
    raw = torch.nn.functional.binary_cross_entropy_with_logits(
        logits[:, 1], torch.tensor(targets), reduction="none"
    )
    torch.testing.assert_close(loss, (raw[-1] + 0.1 * raw[:-1].mean()) / 1.1)
    loss.backward()
    assert torch.isfinite(logits.grad).all()
    torch.testing.assert_close(full, raw[-1])
    torch.testing.assert_close(prefix, raw[:-1].mean())


def test_prefix_free_full_loss_unchanged():
    logits = torch.tensor([[0.0, 1.0]])
    loss, full, prefix = parent_objective(logits, [0.2], 0.1)
    torch.testing.assert_close(loss, full)
    assert prefix is None


def test_short_accumulation_window_preserves_parent_mass():
    windows = list(accumulation_windows(list(range(35)), 32))
    assert [len(w) for _, w in windows] == [32, 3]
    values = torch.arange(35, dtype=torch.float32, requires_grad=True)
    for _, window in windows:
        for i in window:
            (values[i] / len(window)).backward()
    torch.testing.assert_close(values.grad[:32], torch.full((32,), 1 / 32))
    torch.testing.assert_close(values.grad[32:], torch.full((3,), 1 / 3))


@pytest.mark.parametrize("target", [float("nan"), -0.1, 1.1])
def test_invalid_target_rejected(target):
    with pytest.raises(ValueError, match="invalid soft target"):
        parent_objective(torch.zeros(1, 2), [target], 0.1)


def test_checkpoint_roundtrip_preserves_next_adam_update(monkeypatch, tmp_path):
    import json

    from gleipnir.branch_trainer import save_checkpoint

    class Model(torch.nn.Linear):
        def save_pretrained(self, path, **kwargs):
            path.mkdir()
            torch.save(self.state_dict(), path / "weights.pt")

    monkeypatch.setattr(torch.cuda, "get_rng_state_all", lambda: [])
    original = Model(2, 1)
    optimizer = torch.optim.AdamW(original.parameters(), lr=2e-5, weight_decay=0)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    inputs = torch.ones(1, 2)
    original(inputs).sum().backward()
    optimizer.step()
    scheduler.step()
    state = {"step": 1, "parents": 32, "prefixes": 10, "contract_sha256": "abc"}
    save_checkpoint(original, optimizer, scheduler, tmp_path, state)
    path = tmp_path / "checkpoints/step-0001"
    assert (
        json.loads((tmp_path / "latest_checkpoint.json").read_text())["parents"] == 32
    )
    restored = Model(2, 1)
    restored.load_state_dict(torch.load(path / "adapter/weights.pt", weights_only=True))
    saved = torch.load(path / "state.pt", weights_only=True)
    restored_optimizer = torch.optim.AdamW(
        restored.parameters(), lr=2e-5, weight_decay=0
    )
    restored_optimizer.load_state_dict(saved["optimizer"])
    restored_scheduler = torch.optim.lr_scheduler.LambdaLR(
        restored_optimizer, lambda _: 1.0
    )
    restored_scheduler.load_state_dict(saved["scheduler"])
    for model, opt in ((original, optimizer), (restored, restored_optimizer)):
        opt.zero_grad(set_to_none=True)
        model(inputs).sum().backward()
        opt.step()
    for a, b in zip(original.parameters(), restored.parameters(), strict=True):
        torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_budget_schedule_uses_short_horizon():
    from gleipnir.branch_trainer import budget_lr_multiplier

    assert budget_lr_multiplier(0, 0.03) == 0
    assert budget_lr_multiplier(0.015, 0.03) == 0.5
    assert budget_lr_multiplier(0.03, 0.03) == 1
    assert budget_lr_multiplier(0.515, 0.03) == pytest.approx(0.5)
    assert budget_lr_multiplier(1.2, 0.03) == 0


def test_budget_partial_window_rescales_before_clipping():
    from gleipnir.branch_trainer import normalize_partial_window

    parameter = torch.nn.Parameter(torch.tensor(1.0))
    for value in [1.0, 3.0, 5.0]:
        (parameter * value / 32).backward()
    normalize_partial_window([parameter], nominal=32, completed=3)
    assert parameter.grad.item() == pytest.approx(3.0)
    with pytest.raises(ValueError, match="invalid partial"):
        normalize_partial_window([parameter], nominal=32, completed=0)


def test_budget_stops_between_parents_and_flushes_partial_update(monkeypatch, tmp_path):
    import contextlib
    import json

    from gleipnir import branch_trainer
    from gleipnir.branch_training import plan_branches

    clock = [0.0]
    visited = []
    model = torch.nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        model.weight.fill_(1.0)

    def forward(model, plan, *args, **kwargs):
        visited.append(plan)
        clock[0] += 10.0
        return torch.stack([model.weight.reshape(()) * 0, model.weight.reshape(())])[
            None
        ]

    monkeypatch.setattr(branch_trainer.time, "perf_counter", lambda: clock[0])
    monkeypatch.setattr(branch_trainer, "branched_decision_logits", forward)
    monkeypatch.setattr(branch_trainer, "synchronize", lambda: None)
    monkeypatch.setattr(branch_trainer, "save_checkpoint", lambda *args: None)
    monkeypatch.setattr(
        torch, "autocast", lambda *args, **kwargs: contextlib.nullcontext()
    )
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda _: 0)
    dataset = [
        {"parent_id": i, "plan": plan_branches([[1, 2]]), "targets": [0.3]}
        for i in range(5)
    ]
    config = {
        "training_wall_seconds": 25.0,
        "seed": 0,
        "effective_batch": 32,
        "learning_rate": 2e-5,
        "warmup_ratio": 0.03,
        "alignment": 64,
        "prefix_weight": 0.1,
        "checkpoint_every": 4,
    }
    result = branch_trainer.train(model, dataset, config, tmp_path)
    assert len(visited) == 3
    assert result["parents"] == 3 and result["budget_reached"]
    assert result["training_gpu_seconds"] == 60
    assert result["budget_overrun_seconds"] == 5
    progress = json.loads((tmp_path / "progress.jsonl").read_text())
    assert progress["gradient_norm"] == pytest.approx(
        torch.sigmoid(torch.tensor(1.0)).item() - 0.3
    )
    assert progress["learning_rate"] > 0
    assert model.weight.item() < 1.0
