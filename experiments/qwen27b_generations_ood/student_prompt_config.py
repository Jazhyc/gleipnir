"""Freeze the unadapted Qwen3.8-27B ID config under the compact student prompt.

Copies the zero-shot ID config (teacher prompt, 0.8869) and changes only the
prompt role/template and the input: the campaign's student-prompt ID rows
(the same 3,012 ids). Question: how much of the trained 27B's gain is the
model adapting to the compact prompt rather than learning from Kimi?
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

from experiments.tool_trajectory_monitoring.benchmark_qwen_ood import (
    render_margin_prompt,
)
from gleipnir.qwen35_adapter_rebase import sha256_file

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
INPUT = "data/id_cot_only_evaluation"
STUDENT_CONFIG = ROOT / "experiments/id_cot_only_evaluation/config.json"


def main() -> None:
    from transformers import AutoTokenizer

    base = json.loads((HERE / "qwen38_27b_id.json").read_text())
    student = json.loads(STUDENT_CONFIG.read_text())["prompt"]
    path = ROOT / INPUT / "prompts.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    teacher_ids = {
        json.loads(line)["id"]
        for line in (ROOT / base["scope"]["input"]).read_text().splitlines()
    }
    if {r["id"] for r in rows} != teacher_ids:
        raise ValueError("student and teacher ID rows differ")
    model = base["model"]
    tokenizer = AutoTokenizer.from_pretrained(model["id"], revision=model["revision"])
    rendered = [
        render_margin_prompt(
            tokenizer,
            row["prompt"],
            enable_thinking=student["enable_thinking"],
            assistant_suffix=student["assistant_suffix"],
            decision_prefix=student["decision_prefix"],
        )
        for row in rows
    ]
    lengths = [len(ids) for ids in tokenizer(rendered)["input_ids"]]
    config = copy.deepcopy(base)
    config["campaign_id"] = "qwen27b-generations-id-student-prompt-qwen38_27b-v1"
    config["hypothesis"] = (
        "Descriptive: unadapted Qwen3.8-27B ID ranking under the compact student "
        "prompt, against its 0.8869 under the full teacher prompt."
    )
    config["prompt"] = {
        **config["prompt"],
        **{k: student[k] for k in ("role", "template_sha256", "prompt_set_id")},
    }
    config["scope"] = {
        **config["scope"],
        "input": f"{INPUT}/prompts.jsonl",
        "input_sha256": sha256_file(path),
        "manifest": f"{INPUT}/evaluation.manifest.json",
        "manifest_sha256": sha256_file(ROOT / INPUT / "evaluation.manifest.json"),
        "rows": len(rows),
    }
    config["engine"]["audited_total_prompt_tokens"] = sum(lengths)
    config["engine"]["audited_max_prompt_tokens"] = max(lengths)
    if max(lengths) >= config["engine"]["max_model_len"]:
        raise ValueError("student prompt exceeds the engine limit")
    out = HERE / "qwen38_27b_id_student.json"
    out.write_text(json.dumps(config, indent=2) + "\n")
    print(len(rows), sum(lengths), max(lengths), out.name, flush=True)


if __name__ == "__main__":
    main()
