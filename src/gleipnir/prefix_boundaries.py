"""Conservative candidate boundaries for rendered monitoring trajectories.

Rendered delimiters are not an authenticated message schema. These offsets need
source-format review before they may become an annotation manifest.
"""

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class BoundaryAudit:
    """Character offsets retain exact original text and include tool responses."""

    candidate_ends: tuple[int, ...]
    assistant_turns: int
    statement_only_turns: int
    warnings: tuple[str, ...]


def audit_boundaries(trajectory: str, source: str) -> BoundaryAudit:
    """Find non-final tool-bearing turn ends; never infer calls from prose."""
    warnings: list[str] = []
    if source == "bash_arena":
        starts = list(re.finditer(r"(?m)^<step_(\d+)>\s*$", trajectory))
        if [int(m.group(1)) for m in starts] != list(range(1, len(starts) + 1)):
            warnings.append("nonconsecutive_step_markers")
    elif source in {"stride", "bash_bench", "gloom", "cot_red_handed"}:
        starts = list(re.finditer(r"(?m)^\[ASSISTANT\]\s*$", trajectory))
    else:
        raise ValueError(f"unsupported source: {source}")
    if not starts:
        warnings.append("no_assistant_turns")
    candidates: list[int] = []
    statements = 0
    for index, start in enumerate(starts):
        end = starts[index + 1].start() if index + 1 < len(starts) else len(trajectory)
        segment = trajectory[start.end() : end]
        boundary_end = end
        if source == "bash_arena":
            malformed_assistant = (
                segment.count("<assistant>") != 1 or segment.count("</assistant>") != 1
            )
            if malformed_assistant:
                warnings.append(f"assistant_delimiters_turn_{index + 1}")
            assistant = segment.split("</assistant>", 1)[0]
            has_call = re.search(r"(?m)^<tool_\d+>\s*$", assistant) is not None
            closing = re.search(rf"(?m)^</step_{start.group(1)}>[ \t]*$", segment)
            if closing is None:
                warnings.append(f"missing_step_close_turn_{index + 1}")
                boundary_end = None
            else:
                boundary_end = start.end() + closing.end()
            if malformed_assistant:
                boundary_end = None
        else:
            assistant = re.split(
                r"(?m)^\[(?:TOOL|USER|SYSTEM)\]\s*$", segment, maxsplit=1
            )[0]
            has_call = re.search(r"(?m)^\[TOOL CALLS\]\s*$", assistant) is not None
        statements += not has_call
        if has_call and index + 1 < len(starts) and boundary_end is not None:
            candidates.append(boundary_end)
    return BoundaryAudit(tuple(candidates), len(starts), statements, tuple(warnings))
