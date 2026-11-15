import json

import pytest

from experiments.serving_concurrency import run, summarize


def test_levels_sweep_only_at_both_ends():
    assert run.levels("bf16") == run.SWEEP
    assert run.levels("fast") == run.SWEEP
    assert run.levels("quant") == (128, 1)
    assert set(run.STEPS["27b"]) <= set(run.STEPS["4b"])


def test_swap_worker_changes_only_the_worker():
    command = ["python", "--worker-cls", "a.B", "--port", "1"]
    swapped = run.swap_worker(command, run.NO_TRIM)
    assert swapped == ["python", "--worker-cls", run.NO_TRIM, "--port", "1"]
    assert command[2] == "a.B"


def test_prepare_rejects_unknown_step(tmp_path):
    with pytest.raises(ValueError):
        run.prepare("27b", "route", tmp_path)


def test_percentile_interpolates():
    assert summarize.percentile([1.0, 2.0, 3.0, 4.0], 0.5) == 2.5
    assert summarize.percentile([5.0], 0.95) == 5.0


def test_condition_summary_takes_medians(tmp_path):
    for repeat, seconds in enumerate((1.0, 2.0, 4.0)):
        rows = [
            {"latency_seconds": 0.1 * (repeat + 1), "prompt_tokens": 100},
            {"latency_seconds": 0.3 * (repeat + 1), "prompt_tokens": 300},
        ]
        (tmp_path / f"full_c4_repeat{repeat}.json").write_text(
            json.dumps({"rows": rows, "seconds": seconds})
        )
    (tmp_path / "condition.json").write_text("{}")
    item = summarize.condition_summary(tmp_path)["full_c4"]
    assert item["passes"] == 3
    assert item["prompt_tokens_per_second"] == 200.0
    assert item["p50_ms"] == pytest.approx(400.0)
