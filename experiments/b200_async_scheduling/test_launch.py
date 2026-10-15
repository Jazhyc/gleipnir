"""The async launch changes only the scheduling flag and scheduler class."""

import pytest

from experiments.b200_async_scheduling.launch import (
    ASYNC_SCHEDULER,
    SYNC_SCHEDULER,
    async_command,
)


def command() -> list[str]:
    return [
        "python",
        "-m",
        "server",
        "--no-async-scheduling",
        "--scheduler-cls",
        SYNC_SCHEDULER,
        "--max-num-seqs",
        "128",
    ]


def test_only_scheduling_mode_and_class_change():
    original = command()
    changed = async_command(original)
    assert original == command()
    assert [a for a, b in zip(original, changed, strict=True) if a != b] == [
        "--no-async-scheduling",
        SYNC_SCHEDULER,
    ]
    assert "--async-scheduling" in changed and ASYNC_SCHEDULER in changed


def test_rejects_unexpected_commands():
    with pytest.raises(ValueError, match="synchronous"):
        async_command(["--async-scheduling", "--scheduler-cls", SYNC_SCHEDULER])
    other = command()
    other[5] = "custom.Scheduler"
    with pytest.raises(ValueError, match="scheduler class"):
        async_command(other)
