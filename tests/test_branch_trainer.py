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
