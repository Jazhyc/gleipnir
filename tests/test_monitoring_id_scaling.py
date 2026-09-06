import math

import pytest
from omegaconf import OmegaConf

from experiments.monitoring_duration.run import two_gpu_lanes
from experiments.monitoring_id_scaling import run
from gleipnir.nested_subsets import nested_subsets


def population():
    return [
        {
            "dataset": "monitoring",
            "source_dataset": source,
            "label": label,
            "index": f"{source}-{label}-{i}",
            "lineage_group": f"{source}-{label}-{i}",
        }
        for source, n in (("a", 30), ("b", 10))
        for label in (0, 1)
        for i in range(n)
    ]


def test_nested_proportional_order_independent_selection():
    rows = population()
    selections = nested_subsets(rows, [4, 8, 16, 40, 80], 0)
    assert selections == nested_subsets(list(reversed(rows)), [4, 8, 16, 40, 80], 0)
    previous = set()
    for count, subset in selections.items():
        ids = {r["index"] for r in subset}
        assert len(ids) == count and previous <= ids
        assert abs(sum(r["source_dataset"] == "a" for r in subset) - 0.75 * count) <= 1
        assert abs(sum(r["label"] for r in subset) - 0.5 * count) <= 1
        previous = ids


def test_shared_lineages_fail_closed():
    rows = population()
    rows[1]["lineage_group"] = rows[0]["lineage_group"]
    with pytest.raises(ValueError, match="lineage"):
        nested_subsets(rows, [4, 8], 0)


def test_four_jobs_stay_on_two_gpus():
    jobs = [{"train_rows": n, "num_train_epochs": 1} for n in (434, 869, 1738, 4344)]
    lanes = two_gpu_lanes(jobs)
    assert len(lanes) == 2
    assert [j["train_rows"] for j in lanes[0]] == [4344]
    assert [j["train_rows"] for j in lanes[1]] == [1738, 869, 434]


def test_one_epoch_jobs_and_exact_steps(monkeypatch, tmp_path):
    config = OmegaConf.to_container(
        OmegaConf.load("experiments/monitoring_id_scaling/config.yaml")
    )
    config["result_dir"] = str(tmp_path)
    monkeypatch.setattr(run, "sha256_file", lambda _: "selection-sha")
    jobs = run.make_jobs(config)
    assert [j["expected_steps"] for j in jobs] == [14, 28, 55, 136]
    for job in jobs:
        assert job["num_train_epochs"] == 1
        assert job["learning_rate"] == 2e-5
        assert job["save_steps"] == math.ceil(job["train_rows"] / 32)
        assert job["mil_loss_weight"] == job["prefix_loss_weight"] == 0
    with pytest.raises(ValueError, match="design drift"):
        run.make_jobs({**config, "epochs": 2})


@pytest.mark.parametrize("state", ["failed", "cancelled_by_user", "running", "waiting"])
def test_dependency_failure_never_releases_queue(state):
    assert not run.dependency_ready({"state": state, "phase": "complete"})


def test_dependency_requires_complete_phase():
    assert not run.dependency_ready({"state": "complete", "phase": "training"})
    assert run.dependency_ready({"state": "complete", "phase": "complete"})


def test_scaling_plot_contains_both_diagnostics(tmp_path):
    curve = [
        {
            "fraction": f,
            "metrics": {
                "macro": {"pauroc_at_20": 0.8, "brier": 0.1},
                "groups": [
                    {"group": name, "pauroc_at_20": 0.8}
                    for name in (
                        "gloom_exfiltration",
                        "test_stride",
                    )
                ],
            },
        }
        for f in (0.05, 0.1, 0.2, 0.5, 1.0)
    ]
    destination = tmp_path / "curve.svg"
    run.plot_curve(curve, destination)
    text = destination.read_text()
    assert "pAUROC" in text and "Brier" in text and "historical 100%" in text
