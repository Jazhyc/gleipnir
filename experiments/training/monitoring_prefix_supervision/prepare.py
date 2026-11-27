"""Materialize compact prefix references; no teacher calls or GPU allocation."""

import hashlib
import json
from pathlib import Path

from gleipnir.prefix_boundaries import audit_boundaries


def digest(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def main() -> None:
    source = Path(
        "data/tool_trajectory_monitoring/distillation_scaling/student_rows.jsonl"
    )
    root = Path("data/monitoring_prefix_supervision")
    root.mkdir(parents=True, exist_ok=True)
    prompt = Path(__file__).with_name("teacher_prefix.txt").read_text()
    output = root / "prefix_references.jsonl"
    # Refuse overwrites: a changed parser or prompt requires explicit new artifacts.
    count = 0
    with output.open("x") as stream:
        for line in source.open():
            row = json.loads(line)
            trajectory = row["student_prompt"].split("<agent_trajectory>\n", 1)[1]
            trajectory = trajectory.rsplit("</agent_trajectory>", 1)[0]
            dataset = row["dataset"].removeprefix("tool_trajectory/")
            audit = audit_boundaries(trajectory, dataset)
            for ordinal, end in enumerate(audit.candidate_ends):
                prefix = trajectory[:end]
                rendered = (
                    f"{prompt.rstrip()}\n<agent_trajectory>\n{prefix}"
                    + ("" if prefix.endswith("\n") else "\n")
                    + "</agent_trajectory>\n"
                )
                record = {
                    "id": f"{row['prompt_id']}:prefix:{end}",
                    "parent_prompt_id": row["prompt_id"],
                    "lineage_group": row["lineage_group"],
                    "source": dataset,
                    "prefix_ordinal": ordinal,
                    "prefix_count": len(audit.candidate_ends),
                    "end_character": end,
                    "trajectory_sha256": digest(trajectory),
                    "prefix_sha256": digest(prefix),
                    "rendered_user_prompt_sha256": digest(rendered),
                    "format_warnings": audit.warnings,
                }
                stream.write(json.dumps(record) + "\n")
                count += 1
    manifest = {
        "status": "prepared_references_not_teacher_cache",
        "rows": count,
        "source": str(source),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "instruction_sha256": digest(prompt),
        "references_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "decision_prefix": "Prediction:",
        "full_endpoint_targets": "unchanged existing Kimi K3 cache",
        "excluded": "statement-only, final turns, malformed step/assistant boundaries",
        "warnings": "Retain source compression warnings; no invented missing actions.",
    }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
