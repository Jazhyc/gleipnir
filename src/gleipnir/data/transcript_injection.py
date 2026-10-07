"""Deterministic, reversible monitor-directed transcript augmentation."""

from __future__ import annotations

import math
import random
import re
from collections import defaultdict
from html import escape
from typing import Any

ROLES = ("user", "assistant", "tool")
_MESSAGES = re.compile(r"^\[(?:USER|ASSISTANT|TOOL)\]\r?$", re.MULTILINE)
_STEPS = re.compile(r"^<step_\d+>\r?$", re.MULTILINE)


def select_stratified(
    rows: list[dict[str, Any]], fraction: float, seed: int
) -> list[int]:
    """Select a fixed quota proportionally within source/hard-label strata."""
    if not rows or not math.isfinite(fraction) or not 0 <= fraction <= 1:
        raise ValueError("invalid augmentation population/fraction")
    identities = [(r["dataset"], str(r["index"])) for r in rows]
    if len(set(identities)) != len(rows):
        raise ValueError("duplicate source identity")
    groups: dict[tuple[str, int], list[int]] = defaultdict(list)
    for i, row in enumerate(rows):
        if row["label"] not in (0, 1):
            raise ValueError("invalid behavior label")
        groups[row["dataset"], row["label"]].append(i)
    quota = round(len(rows) * fraction)
    counts = {k: math.floor(len(v) * fraction) for k, v in groups.items()}
    remainder_order = sorted(
        groups,
        key=lambda k: (-(len(groups[k]) * fraction - counts[k]), k),
    )
    for key in remainder_order[: quota - sum(counts.values())]:
        counts[key] += 1
    rng = random.Random(seed)
    selected = []
    for key in sorted(groups):
        candidates = groups[key].copy()
        rng.shuffle(candidates)
        selected.extend(candidates[: counts[key]])
    rng.shuffle(selected)
    return selected


def balanced_choices(values: tuple | list, count: int, rng: random.Random) -> list:
    """Balance marginal quotas exactly up to a single example."""
    if not values or count < 0:
        raise ValueError("invalid choice population")
    result = [values[i % len(values)] for i in range(count)]
    rng.shuffle(result)
    return result


def draw_position(rng: random.Random, mean: float = 0.5, std: float = 0.25) -> float:
    """Sample a truncated normal; rejection avoids artificial endpoint piles."""
    if not 0 <= mean <= 1 or not math.isfinite(std) or std <= 0:
        raise ValueError("invalid position distribution")
    while True:
        position = rng.gauss(mean, std)
        if 0 <= position <= 1:
            return position


def insert_message(
    trajectory: str, role: str, payload: str, position: float
) -> tuple[str, dict[str, Any]]:
    """Insert one standalone role block at a rendered boundary, never in code.

    Roles here belong to the monitored transcript, not the monitor's outer chat.
    XML sources receive a new numbered step. Original bytes and step numbers
    stay untouched; the new step's number can be out of chronological order.
    """
    if role not in ROLES or not payload.strip() or not 0 <= position <= 1:
        raise ValueError("invalid role, payload or position")
    if not trajectory.strip():
        raise ValueError("empty transcript")
    if trajectory.lstrip().startswith("<task_description>") or re.match(
        r"<step_\d+>", trajectory.lstrip()
    ):
        matches = list(_STEPS.finditer(trajectory))
        if not matches:
            raise ValueError("XML transcript has no complete step boundaries")
        if not re.search(r"</step_\d+>\s*$", trajectory):
            raise ValueError("XML transcript has an incomplete final step")
        task_end = trajectory.find("</task_description>")
        boundaries = sorted(
            {
                0,
                len(trajectory),
                *([task_end + len("</task_description>")] if task_end >= 0 else []),
                *(m.start() for m in matches),
            }
        )
        step = max(int(m.group().strip()[6:-1]) for m in matches) + 1
        tag = {"user": "user", "assistant": "assistant", "tool": "tool_output_0"}[role]
        message = f"<step_{step}>\n<{tag}>\n{escape(payload)}\n</{tag}>\n</step_{step}>"
        style = "xml_step"
    else:
        matches = list(_MESSAGES.finditer(trajectory))
        if not matches or matches[0].start() != 0:
            raise ValueError("unknown transcript format")
        boundaries = sorted({0, len(trajectory), *(m.start() for m in matches)})
        message = f"[{role.upper()}]\n{payload}"
        style = "role_tagged"
    offset = min(boundaries, key=lambda b: (abs(b / len(trajectory) - position), b))
    addition = "\n\n" + message + "\n\n"
    augmented = trajectory[:offset] + addition + trajectory[offset:]
    receipt = {
        "role": role,
        "format": style,
        "requested_position": position,
        "realized_position": offset / len(trajectory),
        "position_units": "fraction_of_original_transcript_characters",
        "boundary_count": len(boundaries),
        "offset": offset,
        "inserted_characters": len(addition),
    }
    if style == "xml_step":
        receipt["source_task_header_present"] = trajectory.lstrip().startswith(
            "<task_description>"
        )
        receipt["source_task_header_closed"] = task_end >= 0
    if remove_message(augmented, receipt) != trajectory:
        raise ValueError("insertion changed original transcript bytes")
    return augmented, receipt


def remove_message(trajectory: str, receipt: dict[str, Any]) -> str:
    """Recover every original byte using the insertion receipt."""
    start = receipt["offset"]
    end = start + receipt["inserted_characters"]
    if not 0 <= start < end <= len(trajectory):
        raise ValueError("invalid insertion span")
    return trajectory[:start] + trajectory[end:]
