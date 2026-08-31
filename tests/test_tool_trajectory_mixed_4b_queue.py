import os

from experiments.tool_trajectory_monitoring.run_mixed_4b_after_scaling import (
    process_matches,
)
from experiments.tool_trajectory_monitoring.run_mixed_4b_lambda import recipe


def test_process_match_is_bound_to_pid_and_marker() -> None:
    assert process_matches(os.getpid(), "pytest")
    assert not process_matches(os.getpid(), "definitely-not-this-command")
    assert not process_matches(2**31 - 1, "pytest")


def test_mixed_4b_status_recipe_is_explicitly_soft_only() -> None:
    frozen = recipe()
    assert frozen["model"] == "Qwen/Qwen3.5-4B"
    assert frozen["target"] == "kimi_soft_only"
    assert frozen["rank"] == 128
    assert frozen["effective_batch_size"] == 32
    assert frozen["max_length"] == 29_696
    assert frozen["epochs"] == 1
