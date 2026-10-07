"""CPU placement must cover threads, restore masks and reject recycled PIDs."""

import pytest

from experiments.b200_inference_benchmark.cpu_placement import choose_placement
from gleipnir import serving_cpu_placement as placement


def fake_stat(pid, birth):
    return f"{pid} (name with spaces) " + " ".join(["0"] * 19 + [str(birth)])


def test_cpu_lists():
    assert placement.parse_cpu_list("0-2,7,9-10\n") == {0, 1, 2, 7, 9, 10}
    for invalid in ["", "1-0", "-2", "3,", "1-2-3"]:
        with pytest.raises(ValueError):
            placement.parse_cpu_list(invalid)


def test_restore_thread_masks_and_reject_pid_reuse(tmp_path, monkeypatch):
    process = tmp_path / "100"
    process.mkdir()
    (process / "stat").write_text(fake_stat(100, 500))
    for tid in [100, 101]:
        task = process / "task" / str(tid)
        task.mkdir(parents=True)
        (task / "stat").write_text(fake_stat(tid, tid + 400))
    masks = {100: {0, 1, 2, 3}, 101: {0, 2}}
    monkeypatch.setattr(placement.os, "sched_getaffinity", lambda tid: masks[tid])
    monkeypatch.setattr(
        placement.os,
        "sched_setaffinity",
        lambda tid, cpus: masks.update({tid: set(cpus)}),
    )
    controller = placement.ProcessAffinity(100, proc=tmp_path)
    assert controller.apply({2, 3}, leader_cpus={1}) == {"100": [1], "101": [2, 3]}
    controller.apply(None)
    assert controller.apply({0, 1}) == {"100": [0, 1], "101": [0, 1]}
    # A thread created while pinned must restore the original process mask.
    task = process / "task" / "102"
    task.mkdir()
    (task / "stat").write_text(fake_stat(102, 600))
    masks[102] = {0, 1}
    controller.apply(None)
    assert masks == {100: {0, 1, 2, 3}, 101: {0, 2}, 102: {0, 1, 2, 3}}
    for invalid in [set(), {4}]:
        with pytest.raises(ValueError, match="subset"):
            controller.apply(invalid)
    with pytest.raises(ValueError, match="leader mask"):
        controller.apply(None, leader_cpus={1})
    (process / "stat").write_text(fake_stat(100, 501))
    before = dict(masks)
    with pytest.raises(ValueError, match="identity"):
        controller.apply({0})
    assert masks == before


def comparison(ratios, *, p50=1.0, p95=1.0, delta=0.0):
    return {
        "c128": {
            "paired_throughput": {
                "median": sorted(ratios)[len(ratios) // 2],
                "ratios": ratios,
            },
            "ranking": {
                "auroc_delta": {"macro": delta, "pooled": delta, "per_source": {}}
            },
        },
        "c1": {
            "latency_p50": {"original": 1, "candidate": p50},
            "latency_p95": {"original": 1, "candidate": p95},
            "paired_scores": {"score": {"max_absolute_difference": 0}},
        },
    }


def test_selection_requires_consistency_latency_and_quality_guards():
    good = comparison([1.02] * 6)
    assert choose_placement({"local": good}) == "local"
    assert choose_placement({"local": comparison([1.02] * 6, p95=1.04)}) == "original"
    assert (
        choose_placement({"local": comparison([1.02] * 6, delta=0.002)}) == "original"
    )
    assert (
        choose_placement({"local": comparison([0.99, 0.99, 1.02, 1.02, 1.02, 1.02])})
        == "original"
    )
    assert (
        choose_placement({"local": comparison([0.999] * 6, p50=0.99, p95=0.95)})
        == "local"
    )
