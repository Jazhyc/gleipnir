import pytest
import torch

from experiments.probe_generalization.prepare import stratified
from gleipnir.probes import architectures, fp8
from gleipnir.probes.bank import ProbeBank
from gleipnir.probes.store import ResidentStore, Store, StoreWriter
from gleipnir.probes.training import (
    length_batches,
    make_sweep,
    micro_batches,
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
    "alphaevolve",
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


def test_alphaevolve_penalty_and_names():
    torch.manual_seed(0)
    probe = architectures.build("alphaevolve_l1e-05_o0.01", HIDDEN)
    assert (probe.l1, probe.ortho) == (1e-05, 0.01)
    assert architectures.build("alphaevolve", HIDDEN).l1 == 1e-6
    with torch.no_grad():
        probe.project.weight.copy_(torch.eye(10, 100))
    l1 = sum(p.abs().sum() for p in probe.parameters())
    assert torch.allclose(probe.regularization(), 1e-05 * l1)


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


BANK_ARCHITECTURES = (
    "mean_linear",
    "mean_mlp",
    "attention",
    "multimax",
    "rolling_w4",
    "alphaevolve_l1e-05_o0.01",
)


def padded_batch():
    torch.manual_seed(1)
    x = torch.randn(3, 30, HIDDEN)
    mask = torch.zeros(3, 30, dtype=torch.bool)
    mask[0], mask[1, :2], mask[2, :17] = True, True, True
    return x, mask


def test_bank_matches_individual_probes():
    torch.manual_seed(0)
    names = [a for a in BANK_ARCHITECTURES for _ in range(2)]
    probes = [architectures.build(a, HIDDEN) for a in names]
    bank = ProbeBank(names, probes)
    x, mask = padded_batch()
    with torch.no_grad():
        fused = bank(x, mask)
        alone = torch.stack([p(x, mask) for p in probes], 1)
    assert torch.allclose(fused, alone, atol=1e-4)


def test_bank_step_matches_individual_steps():
    torch.manual_seed(0)
    names = list(BANK_ARCHITECTURES)
    probes = [architectures.build(a, HIDDEN) for a in names]
    bank = ProbeBank(names, probes)
    x, mask = padded_batch()
    target = torch.tensor([1.0, 0.0, 0.3])
    loss = torch.nn.functional.binary_cross_entropy_with_logits
    fused = torch.optim.AdamW(bank.parameters(), lr=1e-2, weight_decay=3e-3)
    fused_loss = loss(
        bank(x, mask), target[:, None].expand(3, len(names)), reduction="none"
    ).mean(0)
    (fused_loss + bank.regularization()).sum().backward()
    fused.step()
    singles = [
        torch.optim.AdamW(p.parameters(), lr=1e-2, weight_decay=3e-3) for p in probes
    ]
    for probe, optimizer in zip(probes, singles, strict=True):
        penalty = getattr(probe, "regularization", lambda: 0.0)()
        (loss(probe(x, mask), target) + penalty).backward()
        optimizer.step()
    with torch.no_grad():
        refreshed = ProbeBank(names, probes)(x, mask)
        assert torch.allclose(bank(x, mask), refreshed, atol=1e-4)


def test_micro_batches_respect_budget():
    lengths = [5, 30, 10, 10, 2]
    chunks = micro_batches([0, 1, 2, 3, 4], lengths, token_budget=40)
    assert sum(chunks, []) == [0, 1, 2, 3, 4]
    assert all(max(lengths[i] for i in c) * len(c) <= 40 or len(c) == 1 for c in chunks)
    assert micro_batches([0, 1], lengths, None) == [[0, 1]]


def test_micro_batched_step_matches_full_batch(tmp_path):
    torch.manual_seed(0)
    writer = StoreWriter(tmp_path / "s", HIDDEN, ("bf16",))
    for i in range(8):
        writer.append(torch.randn(3 + 4 * i, HIDDEN), {"id": str(i), "label": i % 2})
    writer.close({})
    store = Store(tmp_path / "s", "bf16")
    sweeps = [
        make_sweep(
            ["rolling_w4", "alphaevolve"],
            ["hard"],
            [0],
            HIDDEN,
            lr=1e-2,
            weight_decay=3e-3,
            device="cpu",
        )
        for _ in range(2)
    ]
    batch = [list(range(8))]
    train_epoch(sweeps[0], store, batch, "cpu")
    train_epoch(sweeps[1], store, batch, "cpu", token_budget=40)
    assert sweeps[0].specs[0].losses == pytest.approx(sweeps[1].specs[0].losses)
    for a, b in zip(
        sweeps[0].bank.parameters(), sweeps[1].bank.parameters(), strict=True
    ):
        assert torch.allclose(a, b, atol=1e-5)


def test_matched_initialization_across_objectives():
    sweep = make_sweep(
        ["attention"],
        ["hard", "soft"],
        [0],
        HIDDEN,
        lr=1e-3,
        weight_decay=0,
        device="cpu",
    )
    x, mask = padded_batch()
    with torch.no_grad():
        logits = sweep.bank(x, mask)
    assert torch.equal(logits[:, 0], logits[:, 1])


def test_resident_store_matches_disk(tmp_path):
    torch.manual_seed(0)
    writer = StoreWriter(tmp_path / "s", HIDDEN, ("bf16", "fp8"))
    for i, n in enumerate((4, 9, 2)):
        writer.append(torch.randn(n, HIDDEN), {"id": str(i)})
    writer.close({})
    for fmt in ("bf16", "fp8"):
        disk = Store(tmp_path / "s", fmt, skip=1)
        resident = ResidentStore(disk, "cpu")
        assert [r["length"] for r in resident.rows] == [3, 8, 1]
        for i in range(3):
            assert torch.equal(resident.load(i), disk.load(i))


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
    store = ResidentStore(Store(tmp_path / "s", "bf16"), "cpu")
    sweep = make_sweep(
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
        train_epoch(sweep, store, length_batches(lengths, 8, seed=epoch), "cpu")
    logits = score(sweep, store, 8, "cpu")
    labels = torch.tensor([r["label"] for r in store.rows])
    for spec in sweep.specs:
        values = torch.tensor(logits[spec.name])
        assert values[labels == 1].mean() > values[labels == 0].mean() + 1.0
        assert spec.losses[-1] < spec.losses[0]


def test_stratified_sample_is_proportional_and_seeded():
    rows = [
        {"id": f"{i:03d}", "source": "ab"[i % 2], "label": i % 3 == 0}
        for i in range(300)
    ]
    first = stratified(rows, 30, ("source", "label"))
    assert first == stratified(rows, 30, ("source", "label"))
    assert len(first) == 30 and sum(r["source"] == "a" for r in first) == 15


def test_selection_rule_uses_seed_median_and_families():
    from experiments.probe_generalization.evaluate_ood import select

    def entry(arch, objective, seed, pauroc, auroc=0.9, brier=0.1):
        macro = {"pauroc_at_20": pauroc, "auroc": auroc, "brier": brier}
        return f"{arch}-{objective}-s{seed}", {
            "architecture": arch,
            "objective": objective,
            "seed": seed,
            "fp8": {"macro": macro},
        }

    probes = dict(
        [
            # rolling: one lucky seed, low median
            entry("rolling_w64", "soft", 0, 0.99),
            entry("rolling_w64", "soft", 1, 0.50),
            entry("rolling_w64", "soft", 2, 0.50),
            # alphaevolve variants: second has the higher median
            entry("alphaevolve_l1e-06_o0.001", "soft", 0, 0.70),
            entry("alphaevolve_l1e-06_o0.001", "soft", 1, 0.70),
            entry("alphaevolve_l1e-06_o0.001", "soft", 2, 0.70),
            entry("alphaevolve_l1e-05_o0.01", "soft", 0, 0.80),
            entry("alphaevolve_l1e-05_o0.01", "soft", 1, 0.80),
            entry("alphaevolve_l1e-05_o0.01", "soft", 2, 0.60),
        ]
    )
    chosen = select({"checkpoints": {"2": {"probes": probes}}}, "fp8")["soft"]
    assert chosen["selected"]["architecture"] == "alphaevolve_l1e-05_o0.01"
    assert set(chosen["families"]) == {"rolling_w64", "alphaevolve"}
    assert (
        chosen["families"]["alphaevolve"]["architecture"] == "alphaevolve_l1e-05_o0.01"
    )
    assert len(chosen["selected"]["probes"]) == 3


def test_resumed_training_matches_uninterrupted(tmp_path):
    torch.manual_seed(0)
    writer = StoreWriter(tmp_path / "s", HIDDEN, ("bf16",))
    for i in range(12):
        writer.append(torch.randn(4 + i, HIDDEN), {"id": str(i), "label": i % 2})
    writer.close({})
    store = Store(tmp_path / "s", "bf16")
    lengths = [r["length"] for r in store.rows]

    def sweep():
        return make_sweep(
            ["rolling_w4"],
            ["hard"],
            [0],
            HIDDEN,
            lr=1e-2,
            weight_decay=3e-3,
            device="cpu",
        )

    straight, resumed = sweep(), sweep()
    for epoch in (1, 2):
        train_epoch(straight, store, length_batches(lengths, 4, seed=epoch), "cpu")
    train_epoch(resumed, store, length_batches(lengths, 4, seed=1), "cpu")
    torch.save(resumed.bank.state_dict(), tmp_path / "bank.pt")
    torch.save(resumed.optimizer.state_dict(), tmp_path / "optimizer.pt")
    restored = sweep()
    restored.bank.load_state_dict(torch.load(tmp_path / "bank.pt"))
    restored.optimizer.load_state_dict(torch.load(tmp_path / "optimizer.pt"))
    train_epoch(restored, store, length_batches(lengths, 4, seed=2), "cpu")
    for a, b in zip(
        straight.bank.parameters(), restored.bank.parameters(), strict=True
    ):
        assert torch.equal(a, b)


def test_budget_extension_triggers_only_at_last_checkpoint(tmp_path, monkeypatch):
    import json as _json
    from types import SimpleNamespace

    from experiments.probe_generalization import evaluate_ood

    run = tmp_path / "results/probe_generalization/x/prompted-fp8"
    run.mkdir(parents=True)
    config = {
        "store_root": "/s",
        "population": "full",
        "backbone": "prompted",
        "train_format": "fp8",
        "eval_formats": ["fp8"],
        "eval_split": "id",
        "architectures": ["rolling_w64"],
        "objectives": ["hard", "soft"],
        "seeds": [0],
        "checkpoints": [1, 2, 4, 8],
        "batch": 16,
        "lr": 1e-4,
        "weight_decay": 3e-3,
        "data_seed": 0,
        "resident": "none",
        "name": "x",
        "token_budget": 65536,
    }
    (run / "config.json").write_text(_json.dumps(config))
    monkeypatch.setattr(evaluate_ood, "ROOT", tmp_path)
    calls = []
    monkeypatch.setattr(
        evaluate_ood.subprocess, "run", lambda argv, **_: calls.append(argv)
    )
    args = SimpleNamespace(name="x", backbone="prompted")
    summary = {"checkpoints": {"1": {}, "8": {}}}
    early = {"hard": {"selected": {"epoch": 2}}, "soft": {"selected": {"epoch": 4}}}
    assert evaluate_ood.extend_if_boundary(args, early, summary) is None
    late = {"hard": {"selected": {"epoch": 2}}, "soft": {"selected": {"epoch": 8}}}
    record = evaluate_ood.extend_if_boundary(args, late, summary)
    assert record == {
        "boundary_cells": ["soft"],
        "from_epoch": 8,
        "to_epoch": 16,
        "fresh_optimizer": True,
    }
    argv = calls[0]
    assert argv[argv.index("--resume-from") + 1] == "8"
    assert argv[argv.index("--checkpoints") + 1 : argv.index("--batch")] == [
        "1",
        "2",
        "4",
        "8",
        "16",
    ]
    assert "--fresh-optimizer" in argv and "65536" in argv
    done = {"checkpoints": {"16": {}}}
    assert (
        evaluate_ood.extend_if_boundary(
            args, {"soft": {"selected": {"epoch": 16}}}, done
        )
        is None
    )


def test_zero_lr_replay_rebuilds_moments_without_moving_weights(tmp_path):
    torch.manual_seed(0)
    writer = StoreWriter(tmp_path / "s", HIDDEN, ("bf16",))
    for i in range(8):
        writer.append(torch.randn(5 + i, HIDDEN), {"id": str(i), "label": i % 2})
    writer.close({})
    store = Store(tmp_path / "s", "bf16")
    lengths = [r["length"] for r in store.rows]
    sweep = make_sweep(
        ["rolling_w4", "alphaevolve"],
        ["hard"],
        [0],
        HIDDEN,
        lr=1e-2,
        weight_decay=3e-3,
        device="cpu",
    )
    for group in sweep.optimizer.param_groups:
        group["lr"] = 0.0
    before = [p.detach().clone() for p in sweep.bank.parameters()]
    train_epoch(sweep, store, length_batches(lengths, 4, seed=0), "cpu")
    assert all(
        torch.equal(a, b) for a, b in zip(before, sweep.bank.parameters(), strict=True)
    )
    states = [sweep.optimizer.state[p] for p in sweep.bank.parameters()]
    assert all(int(s["step"]) == 2 and s["exp_avg_sq"].abs().sum() > 0 for s in states)
