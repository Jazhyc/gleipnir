"""Materialize ID prompts without assistant prose and count inference tokens."""

from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path

from transformers import AutoTokenizer

from experiments.tool_trajectory_monitoring.prepare_distillation_ood import (
    atomic_write_json,
    atomic_write_jsonl,
    extract_trajectory,
    sha256_file,
)
from experiments.tool_trajectory_monitoring.prompting import load_prompt_set

HEADER = re.compile(
    r"^\[(assistant|user|system|tool|tool calls|tool_call:[^\]\n]+|"
    r"tool_result[^\]\n]*)\][ \t]*\n",
    re.MULTILINE | re.IGNORECASE,
)


def digest(text: str) -> str:
    """Hash exact UTF-8 text."""
    return hashlib.sha256(text.encode()).hexdigest()


def strip_assistant_prose(trajectory: str) -> tuple[str, int]:
    """Remove assistant bodies; retain their header only before tool calls."""
    matches = list(HEADER.finditer(trajectory))
    if not matches or matches[0].start() != 0:
        raise ValueError("Unrecognized ID transcript format")
    output = []
    removed = 0
    for i, match in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(trajectory)
        if match[1].lower() != "assistant":
            output.append(trajectory[match.start() : end])
            continue
        body = trajectory[match.end() : end]
        removed += bool(body.strip())
        next_role = matches[i + 1][1].lower() if i + 1 < len(matches) else ""
        if next_role == "tool calls" or next_role.startswith("tool_call:"):
            output.append(match[0])
    return "".join(output), removed


def main() -> None:
    from experiments.id_assistant_prose_audit.thinking import (
        source_records,
        strip_thinking_calls,
    )

    source_data = source_records()
    config = json.loads(Path(__file__).with_name("config.json").read_text())
    source = Path(config["input"])
    if sha256_file(source) != config["input_sha256"]:
        raise ValueError("Frozen input drift")
    tokenizer = AutoTokenizer.from_pretrained(
        config["tokenizer"], revision=config["revision"], local_files_only=True
    )
    template = load_prompt_set().student
    rows = [json.loads(line) for line in source.open()]
    if len(rows) != config["rows"] or len({r["id"] for r in rows}) != len(rows):
        raise ValueError("ID membership drift")
    output, measurements = [], []
    totals = defaultdict(lambda: defaultdict(int))
    for index, row in enumerate(rows):
        metadata = row["metadata"]
        if digest(row["prompt"]) != metadata["rendered_prompt_sha256"]:
            raise ValueError("Prompt hash drift")
        trajectory = extract_trajectory(
            row["prompt"], template.cache_prefix, f"{template.trajectory_close}\n"
        )
        if digest(trajectory) != metadata["trajectory_sha256"]:
            trajectory = trajectory.removesuffix("\n")
        if digest(trajectory) != metadata["trajectory_sha256"]:
            raise ValueError("Trajectory hash drift")
        stripped, removed = strip_assistant_prose(trajectory)
        prose_only = stripped
        raw = source_data.get(row["id"], {"tool_blocks": [], "reasoning": []})
        stripped, thinking_calls = strip_thinking_calls(prose_only, raw["tool_blocks"])
        traces = raw["reasoning"]
        if any(trace not in trajectory for trace in traces):
            raise ValueError("Source reasoning not located in frozen prompt")
        if any(trace in stripped for trace in traces):
            raise ValueError("Source reasoning remains after cleanup")
        prompt = template.render(stripped)
        metrics = {
            "rows": 1,
            "prose_messages_removed": removed,
            "thinking_calls_removed": thinking_calls,
            "rows_with_thinking_calls": int(thinking_calls > 0),
            "reasoning_blocks": len(traces),
            "rows_with_reasoning_blocks": int(bool(traces)),
            "rows_with_either_thinking": int(bool(traces or thinking_calls)),
        }
        for name, text in (
            ("before", row["prompt"]),
            ("prose_only", template.render(prose_only)),
            ("after", prompt),
        ):
            rendered = tokenizer.apply_chat_template(
                [{"role": "user", "content": text}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            if not rendered.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n"):
                raise ValueError("Chat boundary drift")
            metrics[f"{name}_tokens"] = len(
                tokenizer.encode(rendered + "Prediction:", add_special_tokens=False)
            )
        for group in ("total", metadata["source_dataset"]):
            for key, value in metrics.items():
                totals[group][key] += value
        measurements.append(
            {"id": row["id"], "source": metadata["source_dataset"], **metrics}
        )
        output.append(
            {
                "id": row["id"],
                "prompt": prompt,
                "metadata": {
                    "source_dataset": metadata["source_dataset"],
                    "ground_truth": metadata["ground_truth"],
                    "ground_truth_provenance": metadata["ground_truth_provenance"],
                    "prompt_role": "student",
                    "prompt_set_id": template.prompt_set_id,
                    "prompt_template_sha256": template.template_sha256,
                    "rendered_prompt_sha256": digest(prompt),
                    "trajectory_sha256": digest(stripped),
                    "transformation": config["transformation"],
                    "original_metadata": metadata,
                },
            }
        )
        if (index + 1) % 250 == 0:
            print(f"Counted {index + 1}/{len(rows)} rows", flush=True)
    for counts in totals.values():
        counts["saved_tokens"] = counts["before_tokens"] - counts["after_tokens"]
        counts["reduction_percent"] = (
            100 * counts["saved_tokens"] / counts["before_tokens"]
        )
    destination = Path("data/id_assistant_prose_audit/prompts.jsonl")
    atomic_write_jsonl(destination, output)
    atomic_write_jsonl(
        Path("results/id_assistant_prose_audit/rows.jsonl"), measurements
    )
    summary = {
        "config": config,
        "output": str(destination),
        "output_sha256": sha256_file(destination),
        "totals": dict(totals),
    }
    atomic_write_json(Path("results/id_assistant_prose_audit/summary.json"), summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
