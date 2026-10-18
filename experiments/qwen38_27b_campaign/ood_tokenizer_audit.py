"""Check the frozen OOD workload's stored token counts with the 27B tokenizer."""

from __future__ import annotations

import json
import sys

from transformers import AutoTokenizer

from experiments.qwen38_27b_campaign.prepare import MODEL, REVISION
from experiments.teacher_seed_variation.prepare import OOD

tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
rows = [json.loads(line) for line in open(OOD["ood_workload"][0])]
mismatch = sum(
    len(tokenizer(r["prompt"], add_special_tokens=False)["input_ids"])
    != r["prompt_tokens"]
    for r in rows
)
print(json.dumps({"rows": len(rows), "count_mismatch": mismatch}))
sys.exit(1 if mismatch else 0)
