"""Render the exact teacher and student prompts for human review."""

from __future__ import annotations

import argparse
from pathlib import Path

from experiments.tool_trajectory_monitoring.prompting import load_prompt_set

PLACEHOLDER_TRAJECTORY = """USER: Repair the failing service and report what changed.
ASSISTANT to=terminal.exec: {\"cmd\":\"systemctl status example\"}
TOOL: The example service is inactive.
ASSISTANT: I inspected the service and reported its current state."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--role",
        choices=("teacher", "student", "both"),
        default="both",
        help="prompt role to render (default: both)",
    )
    parser.add_argument(
        "--trajectory-file",
        type=Path,
        help="UTF-8 trajectory file; otherwise a short placeholder is used",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    trajectory = (
        args.trajectory_file.read_text(encoding="utf-8")
        if args.trajectory_file is not None
        else PLACEHOLDER_TRAJECTORY
    )
    prompt_set = load_prompt_set()
    roles = ("teacher", "student") if args.role == "both" else (args.role,)
    for index, role in enumerate(roles):
        template = prompt_set.for_role(role)
        if index:
            print()
        print(f"=== {role.upper()} ===")
        print(f"prompt_set_id: {prompt_set.prompt_set_id}")
        print(f"template_sha256: {template.template_sha256}")
        print(f"cache_prefix_chars: {len(template.cache_prefix)}")
        print()
        print(template.render(trajectory), end="")


if __name__ == "__main__":
    main()
