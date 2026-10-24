import pytest
import torch

from experiments.probe_generalization.prepare import stratified
from gleipnir.probes import architectures, fp8
from gleipnir.probes.store import Store, StoreWriter
from gleipnir.probes.training import (
    length_batches,
    make_runs,
    score,
    train_epoch,
)

HIDDEN = 256
ARCHITECTURES = (
    "mean_linear",
    "mean_mlp",
    "attention",
    "multimax",
    "rolling_w4",
    "rolling_w64",
)


def test_fp8_round_trip_and_block_isolation():
    torch.manual_seed(0)
    values = torch.randn(32, HIDDEN)
    values[:, 3] = 5000.0  # one outlier channel in block 0
    values[5] = 0.0
    codes, scales = fp8.quantize(values)
    assert codes.dtype == torch.uint8 and scales.shape == (32, HIDDEN // fp8.BLOCK)
    restored = fp8.dequantize(codes, scales, torch.float32)
    assert torch.equal(restored[5], torch.zeros(HIDDEN))
    error = (restored[:, 128:] - values[:, 128:]).abs() / values[:, 128:].abs().amax()
    assert float(error.max()) < 0.04


def test_store_round_trip(tmp_path):
    torch.manual_seed(0)
    rows = [torch.randn(n, HIDDEN) for n in (3, 7, 1)]
    writer = StoreWriter(tmp_path / "s", HIDDEN, ("bf16", "fp8"))
    for i, r in enumerate(rows):
        writer.append(r, {"id": str(i), "label": i % 2})
    writer.close({"note": "test"})
    bf16, quantized = Store(tmp_path / "s", "bf16"), Store(tmp_path / "s", "fp8")
    assert [r["length"] for r in bf16.rows] == [3, 7, 1]
    for i, r in enumerate(rows):
        assert torch.equal(bf16.load(i), r.to(torch.bfloat16))
        assert torch.allclose(quantized.load(i).float(), r, atol=0.15, rtol=0.07)


def test_store_skip_drops_leading_tokens(tmp_path):
    rows = [torch.randn(n, HIDDEN) for n in (3, 5)]
    writer = StoreWriter(tmp_path / "s", HIDDEN, ("bf16",))
    for i, r in enumerate(rows):
        writer.append(r, {"id": str(i)})
    writer.close({})
    store = Store(tmp_path / "s", "bf16", skip=1)
    assert [r["length"] for r in store.rows] == [2, 4]
    assert torch.equal(store.load(1), rows[1][1:].to(torch.bfloat16))
    with pytest.raises(ValueError, match="no tokens left"):
        Store(tmp_path / "s", "bf16", skip=3)


def test_store_rejects_nonfinite(tmp_path):
    writer = StoreWriter(tmp_path / "s", HIDDEN, ("fp8",))
    with pytest.raises(ValueError, match="nonfinite"):
        writer.append(torch.full((2, HIDDEN), float("nan")), {"id": "x"})


@pytest.mark.parametrize("name", ARCHITECTURES)
def test_padding_does_not_change_scores(name):
    torch.manual_seed(0)
    probe = architectures.build(name, HIDDEN).eval()
    a, b = torch.randn(9, HIDDEN), torch.randn(5, HIDDEN)
    x = torch.zeros(2, 9, HIDDEN)
    x[0], x[1, :5] = a, b
    x[1, 5:] = 100.0  # padding content must not leak
    mask = torch.zeros(2, 9, dtype=torch.bool)
    mask[0], mask[1, :5] = True, True
    with torch.no_grad():
        batched = probe(x, mask)
        alone = probe(b.unsqueeze(0), torch.ones(1, 5, dtype=torch.bool))
    assert torch.allclose(batched[1], alone[0], atol=1e-5)


@pytest.mark.parametrize("name", ARCHITECTURES)
def test_gradients_are_finite_with_padding(name):
    torch.manual_seed(0)
    probe = architectures.build(name, HIDDEN)
    x = torch.randn(3, 70, HIDDEN)
    mask = torch.zeros(3, 70, dtype=torch.bool)
    mask[0], mask[1, :2], mask[2, :65] = True, True, True
    probe(x, mask).sum().backward()
    assert all(torch.isfinite(p.grad).all() for p in probe.parameters())


def test_rolling_window_wider_than_row_equals_attention():
    torch.manual_seed(0)
    attention = architectures.build("attention", HIDDEN).eval()
    rolling = architectures.build("rolling_w64", HIDDEN).eval()
    rolling.load_state_dict(attention.state_dict())
    x, mask = torch.randn(1, 20, HIDDEN), torch.ones(1, 20, dtype=torch.bool)
    with torch.no_grad():
        assert torch.allclose(rolling(x, mask), attention(x, mask), atol=1e-5)


def test_rolling_chunks_match_unchunked(monkeypatch):
    torch.manual_seed(0)
    probe = architectures.build("rolling_w4", HIDDEN).eval()
    x, mask = torch.randn(2, 50, HIDDEN), torch.ones(2, 50, dtype=torch.bool)
    with torch.no_grad():
        whole = probe(x, mask)
        monkeypatch.setattr(architectures, "ROLLING_CHUNK", 7)
        assert torch.allclose(probe(x, mask), whole, atol=1e-6)


def test_length_batches_cover_every_row_once():
    batches = length_batches(list(range(100, 0, -1)), batch=8, seed=3, chunk=32)
    flat = sorted(i for b in batches for i in b)
    assert flat == list(range(100)) and max(len(b) for b in batches) == 8


def test_matched_initialization_across_objectives():
    runs = make_runs(
        ["attention"],
        ["hard", "soft"],
        [0],
        HIDDEN,
        lr=1e-3,
        weight_decay=0,
        device="cpu",
    )
    hard, soft = (r.probe.state_dict() for r in runs)
    assert all(torch.equal(hard[k], soft[k]) for k in hard)


def test_training_separates_synthetic_rows(tmp_path):
    torch.manual_seed(0)
    writer = StoreWriter(tmp_path / "s", HIDDEN, ("bf16",))
    for i in range(64):
        label = i % 2
        values = torch.randn(int(torch.randint(5, 40, ())), HIDDEN)
        values[:3, 0] += 4.0 * label  # a short evidence span
        writer.append(
            values, {"id": str(i), "label": label, "soft_target": 0.8 * label}
        )
    writer.close({})
    store = Store(tmp_path / "s", "bf16")
    runs = make_runs(
        ["multimax", "rolling_w4"],
        ["hard", "soft"],
        [0],
        HIDDEN,
        lr=3e-3,
        weight_decay=0,
        device="cpu",
    )
    lengths = [r["length"] for r in store.rows]
    for epoch in range(15):
        train_epoch(runs, store, length_batches(lengths, 8, seed=epoch), "cpu")
    logits = score(runs, store, 8, "cpu")
    labels = torch.tensor([r["label"] for r in store.rows])
    for run in runs:
        values = torch.tensor(logits[run.name])
        assert values[labels == 1].mean() > values[labels == 0].mean() + 1.0
        assert run.losses[-1] < run.losses[0]


def test_stratified_sample_is_proportional_and_seeded():
    rows = [
        {"id": f"{i:03d}", "source": "ab"[i % 2], "label": i % 3 == 0}
        for i in range(300)
    ]
    first = stratified(rows, 30, ("source", "label"))
    assert first == stratified(rows, 30, ("source", "label"))
    assert len(first) == 30 and sum(r["source"] == "a" for r in first) == 15
