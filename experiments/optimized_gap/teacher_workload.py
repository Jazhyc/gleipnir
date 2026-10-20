"""Render the 27B full-teacher-prompt OOD workload and its BF16 reference.

Gleipnir 27B is evaluated under the full teacher prompt (user decision,
2026-10-20). Its BF16 reference is the vocab-distillation extraction
(`gleipnir27b_vocab_distillation/extract.py`, stock vLLM 0.31 BF16 with an
FP32 LM head; 0.9352 Mean-OOD pAUROC@20). This renders the same 6,395 teacher
prompts with the same template and requires every rendered prompt to match
the extraction's recorded SHA-256, then writes the serving workload and the
reference scores (sigmoid of the recorded 1-vs-0 margin).
"""

from __future__ import annotations

import hashlib
import json
import math

from experiments.gleipnir27b_vocab_distillation.extract import INPUTS, SUFFIX
from experiments.optimized_gap.prepare import OUT, ROOT
from gleipnir.data.monitoring import read_rows, write_rows

MODEL = "/tmp/gleipnir-merged/qwen38-27b-s0-01"
EXTRACT = ROOT / "results/gleipnir27b_vocab_distillation/extract01/gleipnir_teacher_ood"
WORKLOAD = ROOT / "data/optimized_gap/teacher_ood_workload.jsonl"
REFERENCE = OUT / "bf16_teacher_ood.jsonl"


def build() -> None:
    from transformers import AutoTokenizer

    from experiments.tool_trajectory_monitoring.benchmark_qwen_ood import (
        render_margin_prompt,
    )

    path, expected = INPUTS[("teacher", "ood")]
    source = ROOT / path
    if hashlib.sha256(source.read_bytes()).hexdigest() != expected:
        raise ValueError("frozen teacher OOD prompts changed")
    reference = {r["id"]: r for r in read_rows(EXTRACT / "rows.jsonl")}
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    workload = []
    for row in read_rows(source):
        text = render_margin_prompt(
            tokenizer,
            str(row["prompt"]),
            enable_thinking=False,
            assistant_suffix=SUFFIX,
            decision_prefix="Prediction:",
        )
        digest = hashlib.sha256(text.encode()).hexdigest()
        if digest != reference[str(row["id"])]["rendered_prompt_sha256"]:
            raise ValueError(f"rendered teacher prompt differs: {row['id']}")
        workload.append(
            {
                "id": str(row["id"]),
                "prompt": text,
                "prompt_sha256": digest,
                "prompt_tokens": reference[str(row["id"])]["prompt_tokens"],
            }
        )
    if len(workload) != len(reference) != 6395:
        raise ValueError("teacher OOD coverage changed")
    WORKLOAD.parent.mkdir(parents=True, exist_ok=True)
    write_rows(WORKLOAD, workload)
    write_rows(
        REFERENCE,
        [
            {"id": i, "score": 1 / (1 + math.exp(-r["margin"]))}
            for i, r in reference.items()
        ],
    )
    print("teacher_workload_built", json.dumps({"rows": len(workload)}), flush=True)


if __name__ == "__main__":
    build()
