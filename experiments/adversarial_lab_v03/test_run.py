"""Report logic on synthetic predictions (no GPU)."""

from __future__ import annotations

from experiments.adversarial_lab_v03.run import (
    flag_table,
    followup_report,
    framing_report,
    selected,
)


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


def test_framing_pairs_each_post_with_its_unedited_lab_score():
    lab = [
        {**rec("plain", 0.1, "a", None, 0.0, -2.0), "id": "a/F"},
        {**rec("plain", 0.2, "b", None, 0.0, -1.0), "id": "b/D"},
        {**rec("plain", 0.9, "c", None, 0.0, 3.0), "id": "c/E"},  # not a reader
    ]
    posts = [
        rec("peer_accusation", 0.8, "a", "a/F", 0.0, 1.0),
        rec("peer_accusation", 0.1, "b", "b/D", 0.0, -3.0),
    ]
    report = framing_report(posts, {"1%": 0.5}, lab)
    out = report["paired_vs_none"]["peer_accusation"]
    assert out["pairs"] == 2
    assert out["share_margin_up"] == 0.5
    assert out["newly_flagged_at_1%"] == 1
    assert report["flags"]["none"]["rows"] == 2


def test_run0_keeps_honest_rows_and_parity_sources_only():
    def ood(source, label):
        return {"metadata": {"source_dataset": source, "ground_truth": label}}

    rows = [ood("iac", 0), ood("iac", 1), ood("agentdojo", 1), ood("rogue_eval", 1)]
    kept = selected("ood", rows)
    assert [r["metadata"]["source_dataset"] for r in kept] == [
        "iac",
        "agentdojo",
        "rogue_eval",
    ]
    framing = [{"metadata": {"condition": c}} for c in ("none", "peer_accusation")]
    assert [r["metadata"]["condition"] for r in selected("framing", framing)] == [
        "peer_accusation"
    ]


def test_followup_pairs_within_each_rendering():
    lab = [{**rec("plain", 0.1, "a", None, 0.0, -2.0), "id": "a/F"}]
    rows = [
        rec("tool:none", 0.2, "a", "a/F", 0.0, -1.0),
        rec("tool:monitor_benign_w1", 0.3, "a", "a/F", 0.0, -0.5),
        rec("user:monitor_benign_w1", 0.05, "a", "a/F", 0.0, -3.0),
    ]
    out = followup_report(rows, {"1%": 0.5}, lab)["paired_vs_none"]
    assert out["tool:monitor_benign_w1"]["median_margin_change"] == 0.5
    assert out["user:monitor_benign_w1"]["median_margin_change"] == -1.0
    assert out["user:monitor_benign_w1"]["share_margin_down"] == 1.0
