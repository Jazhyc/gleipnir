import numpy as np
import torch

from experiments.probe_attention_maps.analyze import attention_terms, summarize
from experiments.probe_attention_maps.segments import CLASSES, spans, token_classes
from experiments.probe_attention_maps.select import percentile_ranks, select
from gleipnir.probes.training import make_sweep


def test_attention_terms_reproduce_bank_forward():
    sweep = make_sweep(
        ["mean_linear", "attention", "multimax", "rolling_w4"],
        ["hard", "soft"],
        [0, 1],
        64,
        lr=1e-3,
        weight_decay=0.0,
        device="cpu",
    )
    bank, names = sweep.bank, [s.name for s in sweep.specs]
    with torch.no_grad():
        for param in bank.parameters():  # move off the shared initialization
            param.add_(0.1 * torch.randn_like(param))
        wanted = [n for n in names if n.startswith("attention-")]
        x = torch.randn(1, 37, 64)
        a, v, bias = attention_terms(bank, names, wanted, x[0])
        expected = bank(x, torch.ones(1, 37, dtype=torch.bool))[0]
    columns = [names.index(w) for w in wanted]
    assert a.shape == (4, 10, 37)
    torch.testing.assert_close(a.sum(-1), torch.ones(4, 10))
    torch.testing.assert_close((a * v).sum((-1, -2)) + bias, expected[columns])


def test_summary_contributions_sum_to_logit_terms():
    torch.manual_seed(0)
    a = torch.softmax(torch.randn(10, 10, 20), -1)
    v = torch.randn(10, 10, 20)
    classes = torch.randint(0, len(CLASSES), (20,))
    s = summarize(a, v, classes)
    torch.testing.assert_close(s["mass"].sum(-1), torch.ones(10, 10))
    torch.testing.assert_close(s["contrib"].sum(-1), (a * v).sum(-1))
    assert s["overlap"].shape == (5, 10) and bool((s["overlap"] <= 1 + 1e-6).all())
    same = summarize(torch.cat([a[:5], a[:5]]), v, classes)
    torch.testing.assert_close(same["overlap"], torch.ones(5, 10))


def test_bracket_and_colon_segments():
    text = (
        "[SYSTEM]\nbe good\n[USER]\npay\n\n[ASSISTANT]\nok\n"
        "[TOOL CALLS]\nsend()\n\n[TOOL]\ndone"
    )
    kinds = [c for _, c in spans(text)]
    assert kinds == [
        "other",
        "marker",
        "system",
        "marker",
        "user",
        "marker",
        "assistant",
        "marker",
        "tool_call",
        "marker",
        "tool_output",
    ]
    offsets = [
        (0, 1),
        (text.index("pay"), text.index("pay") + 3),
        (text.index("send"), text.index("send") + 4),
    ]
    assert [CLASSES[i] for i in token_classes(text, offsets)] == [
        "marker",
        "user",
        "tool_call",
    ]
    colon = (
        "USER: do it\n\nASSISTANT: Tool calls:\n- bash\n\nTOOL: ok\n\nASSISTANT: done"
    )
    assert [c for _, c in spans(colon)][1:] == [
        "marker",
        "user",
        "marker",
        "tool_call",
        "marker",
        "tool_output",
        "marker",
        "assistant",
    ]
    assert [c for _, c in spans("no headers")] == ["other"]


def test_percentile_ranks_average_ties():
    ranks = percentile_ranks(np.array([[3.0, 1.0, 3.0, 2.0]]))
    np.testing.assert_allclose(ranks, [[2.5 / 3, 0.0, 2.5 / 3, 1 / 3]])


def test_selection_sets_and_examples():
    rng = np.random.default_rng(0)
    id_labels = np.array([0] * 100 + [1] * 20)
    id_logits = np.tile(np.r_[np.linspace(-1, 1, 100), np.full(20, 3.0)], (10, 1))
    # Row 0: positive only soft catches; row 1: negative only soft flags;
    # row 2: positive all catch; row 3: positive none catch.
    ood = np.array([[0.0, 0.0, 5.0, 0.0]] * 5 + [[5.0, 5.0, 5.0, 0.0]] * 5)
    ood += rng.normal(0, 1e-3, ood.shape)
    labels = np.array([1, 0, 1, 1])
    out = select(id_logits, id_labels, ood, labels, np.array(["a", "a", "a", "b"]))
    assert np.flatnonzero(out["sets"]["soft_fixes"]).tolist() == [0]
    assert np.flatnonzero(out["sets"]["hard_fixes"]).tolist() == [1]
    assert np.flatnonzero(out["sets"]["both_right_pos"]).tolist() == [2]
    assert np.flatnonzero(out["sets"]["both_missed_pos"]).tolist() == [3]
    assert (0, "soft_fix") in out["examples"] and (1, "hard_fix") in out["examples"]


def test_margin_decomposition_is_exact():
    from experiments.probe_attention_maps.plot import decompose, report

    torch.manual_seed(1)
    rows = []
    bias = torch.randn(10)
    for _ in range(6):
        a = torch.softmax(torch.randn(10, 10, 15), -1)
        v = torch.randn(10, 10, 15)
        s = summarize(a, v, torch.randint(0, len(CLASSES), (15,)))
        s["logits"] = (a * v).sum((-1, -2)) + bias
        rows.append(s)
    summary = {k: torch.stack([r[k] for r in rows]).numpy() for k in rows[0]}
    summary.update(
        bias=bias.numpy(),
        classes=np.array(CLASSES),
        class_tokens=np.ones((6, len(CLASSES)), dtype=int),
    )
    thresholds = np.random.default_rng(0).normal(size=10)
    parts = decompose(summary, thresholds)
    np.testing.assert_allclose(
        parts["attention"] + parts["value"] + parts["offset"], parts["total"], atol=1e-5
    )
    sets = {f"set_{k}": np.array([1, 1, 0, 0, 1, 0], bool) for k in ("soft_fixes",)}
    selection = {**sets, "sources": np.array(["a"] * 6)}
    for k in ("both_right_pos", "both_missed_pos", "hard_fixes"):
        selection[f"set_{k}"] = np.zeros(6, bool)
    out = report(summary, selection, thresholds)
    assert list(out) == ["soft_fixes"] and out["soft_fixes"]["rows"] == 3
    shares = out["soft_fixes"]["attention_mass"]["hard"]
    assert abs(sum(shares.values()) - 1) < 1e-5
