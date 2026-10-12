"""Inventory prose omitted from the pinned prepared SHADE monitor inputs."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq

SOURCE = Path(
    "data/tool_trajectory_monitoring/source/deliberative-monitor-pipeline-2f10f4e/"
    "filtered_eval/ctrl_shade.parquet"
)
SOURCE_SHA256 = "1a7aac49a5cca05a2a30ef66a04528b8ce623fa64ce633196d379f781527c1b1"
OUTPUT = Path("results/shade_prose_audit")


def main() -> None:
    """Save counts and raw missing-message evidence without changing any input."""
    assert hashlib.sha256(SOURCE.read_bytes()).hexdigest() == SOURCE_SHA256
    counts: Counter[str] = Counter()
    models: Counter[str] = Counter()
    examples = []
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with (OUTPUT / "missing_messages.jsonl").open("w") as output:
        for row in pq.read_table(SOURCE).to_pylist():
            raw = json.loads(row["trajectory_data"])
            assert raw["id"] == row["id"]
            transcript = row["transcript"]
            counts["rows"] += 1
            counts[f"label_{row['ground_truth']}"] += 1
            counts[f"thinking_time_{raw['metadata'].get('thinking_time')}"] += 1
            models[str(row["extra_info"]["model"])] += 1
            bodies = re.findall(
                r"^\[ASSISTANT\]\n(.*?)(?=^\[(?:USER|SYSTEM|ASSISTANT|TOOL|"
                r"TOOL CALLS)\]\n|\Z)",
                transcript,
                flags=re.M | re.S,
            )
            bodies = [body.strip() for body in bodies]
            counts["rendered_assistant_sections"] += len(bodies)
            counts["rendered_nonempty_assistant_sections"] += sum(
                bool(b) for b in bodies
            )
            prose = []
            for index, message in enumerate(raw["messages"]):
                if message["role"] != "assistant":
                    continue
                counts["raw_assistant_messages"] += 1
                text = message.get("content")
                assert text is None or isinstance(text, str)
                if not text or not text.strip():
                    continue
                prose.append((index, text))
                counts["raw_nonempty_prose_messages"] += 1
                counts["raw_prose_chars"] += len(text)
                counts["messages_with_thinking_open_tag"] += "<thinking>" in text
                outside = re.sub(
                    r"<(thinking|system_instructions)>.*?</\1>", "", text, flags=re.S
                ).strip()
                counts["nonempty_after_tag_removal"] += bool(outside)
                # Untyped content outside tags is not automatically public prose.
                counts["messages_mentioning_side_task"] += bool(
                    re.search(r"\b(?:side|secondary) task\b", text, flags=re.I)
                )
                present = text.strip() in transcript
                counts["exact_content_substring_present"] += present
                record = {
                    "id": row["id"],
                    "row_index": row["row_index"],
                    "label": row["ground_truth"],
                    "message_index": index,
                    "content_sha256": hashlib.sha256(text.encode()).hexdigest(),
                    "exact_substring_present": present,
                    "content": text,
                }
                output.write(json.dumps(record) + "\n")
            counts["rows_with_prose"] += bool(prose)
            if row["row_index"] in {0, 251, 252}:
                examples.append(
                    {
                        "id": row["id"],
                        "row_index": row["row_index"],
                        "label": row["ground_truth"],
                        "task": raw["metadata"].get("task_pair"),
                        "last_prose": prose[-1:] if prose else [],
                    }
                )
    assert counts["rows"] == 850
    assert counts["rendered_nonempty_assistant_sections"] == 0
    receipt = {
        "source": str(SOURCE),
        "source_sha256": SOURCE_SHA256,
        "counts": dict(counts),
        "models": dict(models),
        "examples": examples,
        "classification": "inventory_only_no_semantic_public_private_classifier",
    }
    (OUTPUT / "inventory.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({"counts": dict(counts), "models": dict(models)}, indent=2))


if __name__ == "__main__":
    main()
