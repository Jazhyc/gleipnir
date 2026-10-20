"""Generate Position/Reason soft targets from a local 27B teacher.

One stock vLLM 0.31 BF16 engine renders each control training row under the
auxiliary teacher template (`targets.aux_template`), forces the strict-filter
label (equal to Kimi's argmax on these rows) as ``Prediction:{label}`` and
then ``\\nPosition:``, and greedily generates the rest. Every generated token
keeps its top-64 raw logprobs, so the student can be distilled on the
teacher's distribution, not only on its sampled text.

``--limit`` with ``--sample`` draws a source-by-label stratified pilot.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from collections import defaultdict
from pathlib import Path

from experiments.aux_supervision.targets import (
    aux_template,
    grounding,
    parse_continuation,
)
from gleipnir.data.monitoring import digest, read_rows, trajectory_from_prompt

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/aux_supervision"
CONTROL_ROWS = ROOT / "data/teacher-seed-sdpa-s1-01/strict_filter/training.jsonl"
MODELS = {
    "gleipnir": "/tmp/gleipnir-merged/qwen38-27b-s0-01",
    "base": "/root/hf-27b/hub/models--Qwen--Qwen3.8-27B/snapshots/"
    "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
}
SUFFIX = "<|im_start|>assistant\n<think>\n\n</think>\n\n"
TOP = 64
MAX_TOKENS = 160


def stratified(rows: list[dict], count: int, seed: int) -> list[dict]:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        groups[(row["raw_source"], row["label"])].append(row)
    rng = random.Random(seed)
    picked = []
    keys = sorted(groups)
    for key in keys:
        share = max(1, round(count * len(groups[key]) / len(rows)))
        picked += rng.sample(groups[key], min(share, len(groups[key])))
    return sorted(picked, key=lambda row: row["index"])


def load(limit: int | None, seed: int) -> tuple[list[dict], dict]:
    rows = read_rows(CONTROL_ROWS)
    source = {"path": str(CONTROL_ROWS.relative_to(ROOT)), "rows": len(rows)}
    if limit:
        rows = stratified(rows, limit, seed)
    for row in rows:
        if digest(row["student_prompt"]) != row["student_prompt_sha256"]:
            raise ValueError("student prompt checksum drift")
        row["trajectory"] = trajectory_from_prompt(
            row["student_prompt"], row["trajectory_sha256"]
        )
    return rows, source


def run(model_name: str, name: str, limit: int | None, seed: int) -> None:
    import vllm
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    from experiments.tool_trajectory_monitoring.benchmark_qwen_ood import (
        render_margin_prompt,
    )

    out = OUT / name / model_name
    out.mkdir(parents=True, exist_ok=False)
    template = aux_template()
    rows, source = load(limit, seed)
    model = Path(MODELS[model_name])
    tokenizer = AutoTokenizer.from_pretrained(model)
    prompts = []
    for row in rows:
        context = render_margin_prompt(
            tokenizer,
            template.render(row["trajectory"]),
            enable_thinking=False,
            assistant_suffix=SUFFIX,
            decision_prefix=template.decision_prefix,
        )
        prompts.append(f"{context}{row['label']}\nPosition:")
    started = time.time()
    llm = LLM(
        model=str(model),
        dtype="bfloat16",
        max_model_len=32768,
        max_num_seqs=128,
        max_num_batched_tokens=32768,
        enable_prefix_caching=False,
        language_model_only=True,
        gpu_memory_utilization=0.85,
        max_logprobs=TOP,
        logprobs_mode="raw_logprobs",
        seed=0,
    )
    engine_seconds = time.time() - started
    params = SamplingParams(
        temperature=0.0, max_tokens=MAX_TOKENS, logprobs=TOP, stop=["<|im_end|>"]
    )
    generated_at = time.time()
    outputs = llm.generate(prompts, params, use_tqdm=False)
    seconds = time.time() - generated_at
    records = []
    for row, prompt, output in zip(rows, prompts, outputs, strict=True):
        completion = output.outputs[0]
        target = parse_continuation(completion.text)
        steps = []
        for token_id, top in zip(
            completion.token_ids, completion.logprobs, strict=True
        ):
            ranked = sorted(top.items(), key=lambda item: -item[1].logprob)[:TOP]
            steps.append(
                {
                    "token": int(token_id),
                    "top_ids": [int(k) for k, _ in ranked],
                    "top_logprobs": [round(float(v.logprob), 5) for _, v in ranked],
                }
            )
        records.append(
            {
                "id": row["index"],
                "label": row["label"],
                "source": row["raw_source"],
                "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                "prompt_tokens": len(output.prompt_token_ids),
                "text": completion.text,
                "finish_reason": completion.finish_reason,
                "position": target.position,
                "reason": target.reason,
                "well_formed": target.well_formed,
                "grounding": grounding(row["trajectory"], row["label"], target),
                "steps": steps,
            }
        )
    (out / "rows.jsonl").write_text(
        "".join(json.dumps(r, sort_keys=True) + "\n" for r in records)
    )
    summary = {
        "model": model_name,
        "model_path": str(model),
        "input": source,
        "rows": len(records),
        "limit": limit,
        "seed": seed,
        "template_sha256": template.template_sha256,
        "vllm": vllm.__version__,
        "engine_seconds": engine_seconds,
        "generation_seconds": seconds,
        "prompt_tokens": sum(r["prompt_tokens"] for r in records),
        "generated_tokens": sum(len(r["steps"]) for r in records),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("extract_complete", json.dumps(summary), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=tuple(MODELS), required=True)
    parser.add_argument("--name", default="pilot01")
    parser.add_argument("--limit", type=int, help="stratified pilot of N rows")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    run(args.model, args.name, args.limit, args.seed)


if __name__ == "__main__":
    main()
