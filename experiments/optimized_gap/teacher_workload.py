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

import argparse
import hashlib
import json
import math

from experiments.gleipnir27b_vocab_distillation.extract import INPUTS, SUFFIX
from experiments.optimized_gap.prepare import OUT, ROOT
from gleipnir.data.monitoring import read_rows, write_rows

MODEL = "/tmp/gleipnir-merged/qwen38-27b-s0-01"
EXTRACTION = ROOT / "results/gleipnir27b_vocab_distillation/extract01"
# split: (extraction job, extract.py input key, row count)
SPLITS = {
    "ood": ("gleipnir_teacher_ood", ("teacher", "ood"), 6395),
    # The canonical CoT-removed ID rows under the teacher template (0.9420).
    "id": ("gleipnir_teacher_id_cot_removed", ("teacher", "id_cot_removed"), 3012),
}


def paths(split: str) -> tuple:
    return (
        ROOT / f"data/optimized_gap/teacher_{split}_workload.jsonl",
        OUT / f"bf16_teacher_{split}.jsonl",
    )


def build(split: str = "ood") -> None:
    from transformers import AutoTokenizer

    from experiments.gleipnir27b_vocab_distillation.extract import load_rows
    from experiments.tool_trajectory_monitoring.benchmark_qwen_ood import (
        render_margin_prompt,
    )

    job, key, count = SPLITS[split]
    WORKLOAD, REFERENCE = paths(split)
    path, expected = INPUTS[key]
    if hashlib.sha256((ROOT / path).read_bytes()).hexdigest() != expected:
        raise ValueError(f"frozen teacher {split} prompts changed")
    rows, _ = load_rows(*key)  # re-renders CoT-removed ID with the teacher template
    reference = {r["id"]: r for r in read_rows(EXTRACTION / job / "rows.jsonl")}
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    workload = []
    for row in rows:
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
    if not len(workload) == len(reference) == count:
        raise ValueError(f"teacher {split} coverage changed")
    WORKLOAD.parent.mkdir(parents=True, exist_ok=True)
    write_rows(WORKLOAD, workload)
    write_rows(
        REFERENCE,
        [
            {"id": i, "score": 1 / (1 + math.exp(-r["margin"]))}
            for i, r in reference.items()
        ],
    )
    print("teacher_workload_built", json.dumps({"split": split, "rows": len(workload)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=SPLITS, default="ood")
    build(parser.parse_args().split)
