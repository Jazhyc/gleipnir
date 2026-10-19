"""Render the full teacher specification as a single Noul question.

Mechanical edits only: the OUTPUT INTERFACE section and the trailing
"trajectory follows" line are dropped (the readout replaces them, and the
state precedes the question), and the labels ``Prediction:1``/``Prediction:0``
become ``true``/``false``. The two DECISION BOUNDARY definitions are reused
verbatim as the Noul criteria.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SPEC = ROOT / "experiments/tool_trajectory_monitoring/prompts/teacher.txt"
OUT = ROOT / "experiments/pplx_decider_ood/question_teacher.json"
TRUE_PREFIX = "Prediction:1 means that the trajectory contains "
FALSE_PREFIX = "Prediction:0 means that the visible actions "


def definition(text: str, prefix: str, subject: str) -> str:
    paragraph = next(p for p in text.split("\n\n") if p.startswith(prefix))
    return subject + paragraph[len(prefix) :]


def build() -> dict:
    spec = SPEC.read_text()
    body, interface = spec.split("\n\nOUTPUT INTERFACE\n\n")
    if "Prediction:0\nPrediction:1" not in interface:
        raise ValueError("unexpected teacher output interface")
    instructions = body.replace("Prediction:1", "true").replace("Prediction:0", "false")
    return {
        "question_set_id": "tool-trajectory-teacher-spec-noul-v1",
        "question_id": "problematic_behavior",
        "source": {
            "path": str(SPEC.relative_to(ROOT)),
            "sha256": hashlib.sha256(spec.encode()).hexdigest(),
        },
        "question": {
            "type": "noul",
            "instructions": instructions.strip(),
            "criteria": {
                "true": definition(spec, TRUE_PREFIX, "The trajectory contains "),
                "false": definition(spec, FALSE_PREFIX, "The visible actions "),
            },
        },
    }


if __name__ == "__main__":
    OUT.write_text(json.dumps(build(), indent=2, ensure_ascii=False) + "\n")
    print(OUT)
