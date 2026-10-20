"""Report logic and liar-swap auditing on synthetic rows (no GPU, no data)."""

from __future__ import annotations

from experiments.adversarial_is_it_4b.build import residual_claims
from experiments.adversarial_is_it_4b.run import framing_table, liar_table, lines_for


def rec(group, score, parent=None, label=0, l1=0.0, rid=None, task="t"):
    return {
        "id": rid or f"{parent}::{group}",
        "group": group,
        "score": score,
        "status": "ok",
        "label": label,
        "parent_id": parent,
        "task_id": task,
        "logit_0": 0.0,
        "logit_1": l1,
    }


def test_lines_come_from_honest_calibration_rows_only():
    rows = [rec("src", s / 100, rid=str(s)) for s in range(100)]
    rows.append(rec("src", 0.999, label=1, rid="h"))
    lines = lines_for(rows)
    assert lines["1%"] < 0.99
    assert lines["20%"] < lines["1%"]


def test_framing_pairs_each_condition_with_the_same_readers_unedited_row():
    rows = [
        rec("none", 0.1, "a", l1=-2.0, task="x"),
        rec("none", 0.2, "b", l1=-1.0, task="y"),
        rec("accuse_w1", 0.9, "a", l1=2.0, task="x"),
        rec("accuse_w1", 0.1, "b", l1=-3.0, task="y"),
        rec("accuse_w1", 0.95, "c", l1=3.0, task="z"),  # no unedited row: dropped
    ]
    table = framing_table(rows, {"1%": 0.5, "2%": 0.4, "5%": 0.3})
    entry = table["accuse_w1"]
    assert entry["readers"] == 2
    assert entry["flagged_1%"] == 0.5
    assert entry["share_margin_up"] == 0.5
    assert entry["newly_flagged_1%"] == 1


def test_liar_swaps_are_paired_with_their_original_liar():
    rows = [
        rec("verified_liar", 0.4, rid="e/B", l1=1.0),
        rec("liar_swap_all", 0.2, "e/B", l1=-1.0, rid="e/B::swap_all"),
        rec("honest_recruiter", 0.1, rid="f/B", l1=-2.0),
    ]
    table = liar_table(rows, {"1%": 0.5, "5%": 0.3})
    assert table["verified_liar"]["flagged_5%"] == 1
    assert table["liar_swap_all"]["median_margin_change_vs_original"] == -2.0
    assert table["liar_swap_all"]["share_margin_down"] == 1.0
    assert table["verified_liar"]["auroc_vs_recruiter_controls"] == 1.0


def test_residual_claims_keep_only_unnegated_quarantine_clauses():
    texts = [
        "A, your destination is NOT quarantined, so your report can deliver.",
        "Without any quarantine, the nine-order upside is available.",
        "Your live-write quarantine leaves the slot without value.",
    ]
    assert residual_claims(texts) == [
        "Your live-write quarantine leaves the slot without value"
    ]
