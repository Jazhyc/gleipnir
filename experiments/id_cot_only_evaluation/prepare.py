"""Materialize source-grounded CoT-only removal and freeze evaluation inputs."""

from __future__ import annotations

import json
from pathlib import Path

from experiments.id_assistant_prose_audit.run import HEADER, digest
from experiments.id_assistant_prose_audit.thinking import (
    source_records,
    strip_thinking_calls,
)
from experiments.tool_trajectory_monitoring.benchmark_distilled_ood import (
    validate_config,
    validate_inputs,
    validate_jobs,
)
from experiments.tool_trajectory_monitoring.prepare_distillation_ood import (
    atomic_write_json,
    atomic_write_jsonl,
    extract_trajectory,
    sha256_file,
)
from experiments.tool_trajectory_monitoring.prompting import load_prompt_set

ROOT = Path("results/id_cot_only_evaluation")
CONFIG = Path("experiments/id_cot_only_evaluation/config.json")


def strip_cot(trajectory: str, raw: dict) -> tuple[str, int, int]:
    """Remove exact typed reasoning spans only in assistant bodies, then think calls."""
    matches = list(HEADER.finditer(trajectory))
    output, removed, index = [], 0, 0
    for i, match in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(trajectory)
        chunk = trajectory[match.start() : end]
        if match[1].lower() == "assistant":
            blocks = raw["assistant_blocks"][index]
            index += 1
            body = trajectory[match.end() : end]
            texts = [text for _, text in blocks]
            separator = next(
                (sep for sep in ["\n", "\n\n"] if sep.join(texts) == body.rstrip()),
                None,
            )
            if separator is None:
                raise ValueError("Assistant body differs from structured source")
            kept = [text if kind != "reasoning" else "" for kind, text in blocks]
            removed += sum(kind == "reasoning" for kind, _ in blocks)
            chunk = match[0] + separator.join(kept) + body[len(body.rstrip()) :]
        output.append(chunk)
    if index != len(raw["assistant_blocks"]):
        raise ValueError("Assistant message coverage drift")
    text, calls = strip_thinking_calls(
        "".join(output), raw["tool_blocks"], remove_prose=False
    )
    return text, calls, removed


def main() -> None:
    if (ROOT / "status.json").exists():
        raise FileExistsError("Existing run must not be reset")
    config = json.loads(CONFIG.read_text())
    source = Path("data/tool_trajectory_monitoring/distillation_id/prompts.jsonl")
    assert sha256_file(source) == config["scope"]["source_input_sha256"]
    raw = source_records()
    template = load_prompt_set().student
    output, audit = [], []
    for line in source.open():
        row = json.loads(line)
        meta = row["metadata"]
        assert digest(row["prompt"]) == meta["rendered_prompt_sha256"]
        trajectory = extract_trajectory(
            row["prompt"], template.cache_prefix, f"{template.trajectory_close}\n"
        )
        if digest(trajectory) != meta["trajectory_sha256"]:
            trajectory = trajectory.removesuffix("\n")
        assert digest(trajectory) == meta["trajectory_sha256"]
        if row["id"] in raw:
            cleaned, calls, blocks = strip_cot(trajectory, raw[row["id"]])
            prompt = template.render(cleaned)
            new_meta = dict(
                meta,
                trajectory_sha256=digest(cleaned),
                rendered_prompt_sha256=digest(prompt),
                transformation="source-typed-reasoning-and-think-only-v1",
                original_metadata=meta,
            )
            output.append(dict(row, prompt=prompt, metadata=new_meta))
        else:
            calls = blocks = 0
            cleaned = trajectory
            output.append(row)
        audit.append(
            {
                "id": row["id"],
                "think_calls_removed": calls,
                "assistant_reasoning_spans_removed": blocks,
                "changed": cleaned != trajectory,
            }
        )
    assert len(output) == len({r["id"] for r in output}) == 3012
    atomic_write_jsonl(Path(config["scope"]["input"]), output)
    config["scope"]["input_sha256"] = sha256_file(Path(config["scope"]["input"]))
    manifest = {
        "output_sha256": config["scope"]["input_sha256"],
        "source_sha256": config["scope"]["source_input_sha256"],
        "student_template_sha256": template.template_sha256,
        "transformation": "source-typed-reasoning-and-think-only-v1",
        "rows": len(output),
    }
    atomic_write_json(Path(config["scope"]["manifest"]), manifest)
    config["scope"]["manifest_sha256"] = sha256_file(Path(config["scope"]["manifest"]))
    atomic_write_json(CONFIG, config)
    ROOT.mkdir(parents=True, exist_ok=True)
    (ROOT / "jobs.jsonl").write_bytes(
        Path(
            "results/tool_trajectory_distillation_mixed_qwen4b/lambda_jobs.jsonl"
        ).read_bytes()
    )
    validate_config(config)
    validate_inputs(config)
    validate_jobs(config, "4b")
    atomic_write_jsonl(ROOT / "removal_audit.jsonl", audit)
    totals = {
        k: sum(r[k] for r in audit)
        for k in ["think_calls_removed", "assistant_reasoning_spans_removed", "changed"]
    }
    atomic_write_json(ROOT / "removal_summary.json", totals)
    atomic_write_json(ROOT / "status.json", {"state": "prepared", "phase": "prepared"})
    print(json.dumps(totals))


if __name__ == "__main__":
    main()
