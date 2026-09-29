"""The systems campaign preserves workload identity and the allocation deadline."""

import datetime
import json
from pathlib import Path

import pytest

from experiments.fp4_inference.run import condition_config, remaining_seconds


def test_deadline_leaves_ten_minutes_before_gpu_expiry():
    expiry = datetime.datetime(2026, 9, 29, 23, 39, 21, tzinfo=datetime.UTC)
    assert remaining_seconds(expiry) == -600
    assert remaining_seconds(expiry - datetime.timedelta(minutes=20)) == 600


@pytest.mark.parametrize("name", ["../baseline", "baseline.json", "", "/tmp/config"])
def test_condition_resolution_rejects_path_traversal(name):
    with pytest.raises(ValueError):
        condition_config(name)


def test_blackwell_baseline_preserves_historical_serving_contract():
    historical = json.loads(
        Path("experiments/local_inference/iteration32.json").read_text()
    )
    blackwell = json.loads(condition_config("baseline").read_text())
    blackwell.pop("log_dir")
    assert blackwell.pop("output") != historical.pop("output")
    assert blackwell == historical
