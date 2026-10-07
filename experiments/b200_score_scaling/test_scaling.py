"""Exact requested concurrency, workload and reference scheduler contract."""

import json
from pathlib import Path

import pytest

from experiments.b200_score_scaling.run import reference_command, validate_settings
from experiments.b200_score_scaling.summarize import aggregate


def settings():
    return json.loads((Path(__file__).parent / "config.json").read_text())


def test_all_requested_levels_use_full_workload_and_three_repeats():
    config = settings()
    validate_settings(config)
    assert [p["concurrency"] for p in config["benchmark_passes"]] == [
        1,
        2,
        4,
        16,
        32,
        64,
        128,
    ]
    config["benchmark_passes"][0]["workload"] = "quick"
    with pytest.raises(ValueError, match="contract changed"):
        validate_settings(config)


def test_scaling_keeps_stock_scheduler_and_token_budget():
    command = [
        "python",
        "--runner",
        "pooling",
        "--max-num-seqs",
        "128",
        "--max-num-batched-tokens",
        "32768",
    ]
    assert reference_command(command, settings(), {}) == command
    with pytest.raises(ValueError, match="stock reference"):
        reference_command(command + ["--scheduler-cls", "experimental"], settings(), {})


def test_summary_rejects_quick_cohort_or_missing_repeats():
    report = {"status": "complete", "trials": []}
    with pytest.raises(ValueError, match="repeat/workload"):
        aggregate(report)
    report["trials"] = [{"concurrency": 1, "latency": {"rows": 64}}] * 3
    with pytest.raises(ValueError, match="repeat/workload"):
        aggregate(report)
