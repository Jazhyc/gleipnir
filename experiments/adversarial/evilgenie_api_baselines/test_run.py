import numpy as np

from experiments.adversarial.evilgenie_api_baselines import run


def test_shared_ids_follow_order_cap_and_length(monkeypatch):
    monkeypatch.setitem(run.CONFIG, "gleipnir_27b_reached_rows", 4)
    monkeypatch.setitem(run.CONFIG, "gleipnir_max_tokens", 100)
    metas = {
        "p1": {"label_class": "positive"},
        "p2": {"label_class": "positive"},
        "n1": {"label_class": "negative"},
        "n2": {"label_class": "negative"},
        "c1": {"label_class": "clean_negative"},
    }
    tokens = {i: {"27b": 50, "4b-v03": 50} for i in metas}
    tokens["p2"]["4b-v03"] = 101  # too long for one monitor: skipped by both
    shared = run.gleipnir_shared_ids(metas, tokens)
    assert "p1" in shared and "p2" not in shared
    assert len(shared - {"p1"}) == 2  # the cap reaches two shuffled negatives


def test_one_percent_line_flags_at_most_one_percent():
    benign = list(np.arange(1000) / 1000)
    line = run.one_percent_line(benign)
    assert sum(b > line for b in benign) == 10
    tied = [0.5] * 995 + [0.9] * 5
    assert sum(b > run.one_percent_line(tied) for b in tied) <= 10
