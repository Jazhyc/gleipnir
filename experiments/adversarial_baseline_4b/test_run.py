"""Analysis checks on synthetic predictions (no GPU)."""

from __future__ import annotations

from experiments.adversarial_baseline_4b.run import a1_paired, false_alarms, select


def record(group: str, label: int, score: float, task: str = "1") -> dict:
    return {
        "group": group,
        "label": label,
        "score": score,
        "status": "ok",
        "task_id": task,
    }


def test_a1_pairs_by_task_and_payload():
    records = [
        record("vanilla", 1, 0.5, "1"),
        record("inj_transfer_gpt4", 1, 0.9, "1"),
        record("neutral_transfer_gpt4", 1, 0.95, "1"),
        record("vanilla", 1, 0.5, "2"),
        record("inj_transfer_gpt4", 1, 0.2, "2"),
        record("clean", 0, 0.1, "1"),
    ]
    result = a1_paired(records)
    vs_vanilla = result["inj_transfer_gpt4 - vanilla"]
    assert vs_vanilla["tasks"] == 2 and vs_vanilla["share_up"] == 0.5
    assert result["inj_transfer_gpt4 - neutral_transfer_gpt4"]["tasks"] == 1
    assert "inj_transfer_gpt4o - vanilla" not in result


def test_false_alarms_use_clean_honest_threshold():
    benchmark = [record("clean", 0, i / 100) for i in range(100)]
    controls = [record("inj_transfer_gpt4", 0, 0.999)] * 3 + [
        record("neutral_transfer_gpt4", 0, 0.5)
    ] * 3
    result = false_alarms(benchmark, controls)["1%"]
    assert result["groups"]["all_inj"]["flagged"] == 1.0
    assert result["groups"]["all_neutral"]["flagged"] == 0.0


def test_select_samples_each_cell():
    rows = [
        {"id": str(i), "metadata": {"source_dataset": s, "ground_truth": i % 2}}
        for i, s in enumerate(["a"] * 10 + ["b"] * 4)
    ]
    picked = select(rows, 2, 0)
    assert len(picked) == 8
    assert select(rows, 2, 0) == picked
