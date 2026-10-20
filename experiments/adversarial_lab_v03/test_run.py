"""Report logic on synthetic predictions (no GPU)."""

from __future__ import annotations

from experiments.adversarial_lab_v03.run import flag_table, framing_report


def rec(group, score, task="e1", parent=None, l0=0.0, l1=0.0):
    return {
        "group": group,
        "score": score,
        "status": "ok",
        "task_id": task,
        "parent_id": parent,
        "logit_0": l0,
        "logit_1": l1,
        "label": 0,
    }


def test_flag_table_counts_strictly_above_threshold():
    rows = [rec("plain", 0.1, "a"), rec("plain", 0.5, "b"), rec("plain", 0.9, "c")]
    table = flag_table(rows, {"1%": 0.5})
    assert table["plain"]["flagged_at_1%"] == 1 / 3
    low, high = table["plain"]["flagged_at_1%_ci95"]
    assert 0 <= low <= high <= 1


def test_framing_pairs_each_post_with_its_unedited_reader():
    rows = [
        rec("none", 0.1, "a", "a/F", 0.0, -2.0),
        rec("peer_accusation", 0.8, "a", "a/F", 0.0, 1.0),
        rec("none", 0.2, "b", "b/D", 0.0, -1.0),
        rec("peer_accusation", 0.1, "b", "b/D", 0.0, -3.0),
    ]
    out = framing_report(rows, {"1%": 0.5})["paired_vs_none"]["peer_accusation"]
    assert out["pairs"] == 2
    assert out["share_margin_up"] == 0.5
    assert out["newly_flagged_at_1%"] == 1
