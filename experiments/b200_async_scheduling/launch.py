"""Swap the selected command's synchronous scheduler for the async variant."""

from __future__ import annotations

SYNC_SCHEDULER = "experiments.b200_vllm031.scheduler.Vllm031PoolingScheduler"
ASYNC_SCHEDULER = (
    "experiments.b200_async_scheduling.scheduler.Vllm031AsyncPoolingScheduler"
)


def async_command(command: list[str]) -> list[str]:
    """Change only the scheduling mode and scheduler class."""
    command = command.copy()
    if "--no-async-scheduling" not in command or "--async-scheduling" in command:
        raise ValueError("selected command no longer pins synchronous scheduling")
    index = command.index("--scheduler-cls") + 1
    if command[index] != SYNC_SCHEDULER:
        raise ValueError("selected scheduler class changed")
    command[command.index("--no-async-scheduling")] = "--async-scheduling"
    command[index] = ASYNC_SCHEDULER
    return command
