"""Audit candidate prefix counts without launching teacher inference."""

import hashlib
import json
from collections import Counter
from pathlib import Path

from gleipnir.prefix_boundaries import audit_boundaries


def main() -> None:
    path = Path(
        "data/tool_trajectory_monitoring/distillation_scaling/student_rows.jsonl"
    )
    totals: dict[str, Counter] = {}
    flagged = []
    for line in path.open():
        row = json.loads(line)
        source = row["dataset"].removeprefix("tool_trajectory/")
        prompt = row["student_prompt"]
        trajectory = prompt.split("<agent_trajectory>\n", 1)[1].rsplit(
            "</agent_trajectory>", 1
        )[0]
        result = audit_boundaries(trajectory, source)
        counts = totals.setdefault(source, Counter())
        counts.update(
            rows=1,
            assistant_turns=result.assistant_turns,
            statement_only_turns=result.statement_only_turns,
            candidate_prefixes=len(result.candidate_ends),
            flagged_rows=bool(result.warnings),
        )
        if result.warnings:
            flagged.append({"prompt_id": row["prompt_id"], "warnings": result.warnings})
    output = Path("results/monitoring_prefix_supervision/boundary_audit.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            {
                "status": "candidate_audit_not_annotation_manifest",
                "input_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "sources": totals,
                "flagged": flagged,
                "caveat": (
                    "Rendered markers can occur in untrusted text; "
                    "source-format review required."
                ),
            },
            indent=2,
        )
        + "\n"
    )
    print(json.dumps(totals, indent=2))


if __name__ == "__main__":
    main()
