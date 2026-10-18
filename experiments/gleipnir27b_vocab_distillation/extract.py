"""Extract decision-boundary hidden states and full-vocabulary logits.

One persistent stock vLLM 0.31 BF16 engine per model (``--model gleipnir``:
the merged trained Gleipnir 27B; ``--model base``: unadapted Qwen3.8-27B at
the pinned revision) pools the final, normed hidden state of the last prompt
token (``Prediction:``) for every row of each requested job. Logits are
``hidden @ lm_head.T`` in FP32 offline; the 0/1 entries are the monitor's
two-row head, so the existing scorers are a parity check.

Rows are rendered exactly as the frozen ID/OOD benchmark renders them
(`benchmark_qwen_ood.render_margin_prompt`: Qwen chat template, thinking off,
``Prediction:``). Jobs:
- ``teacher:train``: the 8,688-row teacher-prompt pool (the 7,836 strict-
  filtered student rows are flagged);
- ``teacher:id``: the 3,012 ID rows under the full teacher prompt;
- ``student:id``: the 3,012 ID rows under the compact student prompt;
- ``teacher:ood``: the frozen 6,395-row OOD suite under the full teacher prompt.

Writes per job: ``hidden.npy`` (rows x hidden, FP32), ``rows.jsonl`` (id,
label, source, prompt tokens, rendered-prompt hash, logprob_0/1, margin,
probability outside {0, 1}, top-64 tokens) and ``summary.json``.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/gleipnir27b_vocab_distillation"
MODELS = {
    "gleipnir": "/tmp/gleipnir-merged/qwen38-27b-s0-01",
    "base": "/root/hf-27b/hub/models--Qwen--Qwen3.8-27B/snapshots/"
    "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
}
INPUTS = {
    ("teacher", "train"): (
        "data/tool_trajectory_monitoring/teacher_training/prompts.jsonl",
        "c06833bc0c5e8893ea45e92ec642a71e751749ee1cd0796eb88f7ab38402763f",
    ),
    ("teacher", "id"): (
        "data/tool_trajectory_monitoring/teacher_id_benchmark/prompts.jsonl",
        None,
    ),
    ("student", "id"): ("data/id_cot_only_evaluation/prompts.jsonl", None),
    ("teacher", "ood"): (
        "data/tool_trajectory_monitoring/teacher_ood_benchmark/prompts.jsonl",
        "fa4b0d917e51eb91a52efa93cddf0eb154b88993269c97cfdb161c42680136a9",
    ),
}
STUDENT_TRAINING = ROOT / "data/qwen38-27b-s0-01/strict_filter/training.jsonl"
SUFFIX = "<|im_start|>assistant\n<think>\n\n</think>\n\n"
TOP = 64
BATCH = 512


def file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_rows(prompt: str, split: str) -> tuple[list[dict], dict]:
    path, expected = INPUTS[(prompt, split)]
    path = ROOT / path
    digest = file_sha(path)
    if expected is not None and digest != expected:
        raise ValueError(f"frozen input changed: {path}")
    rows = [json.loads(line) for line in path.read_text().splitlines() if line]
    return rows, {"path": str(path.relative_to(ROOT)), "sha256": digest}


def row_meta(row: dict) -> dict:
    meta = row.get("metadata", {})
    label = row.get("label", meta.get("ground_truth", meta.get("label")))
    source = row.get("source", meta.get("raw_source", meta.get("source_dataset")))
    return {"id": str(row["id"]), "label": int(label), "source": str(source)}


def lm_head(model: Path):
    import torch
    from safetensors import safe_open

    index = json.loads((model / "model.safetensors.index.json").read_text())
    shard = model / index["weight_map"]["lm_head.weight"]
    with safe_open(shard, framework="pt", device="cuda") as handle:
        weight = handle.get_tensor("lm_head.weight")
    return weight, hashlib.sha256(weight.view(torch.uint8).cpu().numpy()).hexdigest()


def logit_stats(hidden, weight) -> list[dict]:
    """FP32 log-softmax statistics of hidden @ lm_head.T, per row."""
    import torch

    logits = hidden.float() @ weight.float().T
    logprobs = torch.log_softmax(logits, dim=-1)
    p01 = logprobs[:, 15:17].exp().sum(-1)
    top = logprobs.topk(TOP, dim=-1)
    out = []
    for i in range(hidden.shape[0]):
        out.append(
            {
                "logprob_0": float(logprobs[i, 15]),
                "logprob_1": float(logprobs[i, 16]),
                "margin": float(logits[i, 16] - logits[i, 15]),
                "outside_01_mass": float(1 - p01[i]),
                "top_ids": top.indices[i].tolist(),
                "top_logprobs": [round(float(v), 5) for v in top.values[i]],
            }
        )
    return out


def run(
    model_name: str, jobs: list[tuple[str, str]], name: str, limit: int | None = None
) -> None:
    import torch
    import vllm
    from transformers import AutoTokenizer
    from vllm import LLM
    from vllm.config import PoolerConfig

    from experiments.tool_trajectory_monitoring.benchmark_qwen_ood import (
        render_margin_prompt,
    )

    model = Path(MODELS[model_name] if model_name in MODELS else model_name)
    if not model.is_dir():
        model = Path(glob.glob(str(model))[0])
    tokenizer = AutoTokenizer.from_pretrained(model)
    if tokenizer.convert_tokens_to_ids(["0", "1"]) != [15, 16]:
        raise ValueError("decision tokens are not ids 15/16")
    student_ids = {
        json.loads(line)["index"] for line in STUDENT_TRAINING.read_text().splitlines()
    }
    started = time.time()
    llm = LLM(
        model=str(model),
        runner="pooling",
        convert="embed",
        pooler_config=PoolerConfig(pooling_type="LAST", use_activation=False),
        dtype="bfloat16",
        max_model_len=32768,
        max_num_seqs=128,
        max_num_batched_tokens=32768,
        enable_prefix_caching=False,
        language_model_only=True,
        gpu_memory_utilization=0.85,
        seed=0,
    )
    engine_seconds = time.time() - started
    weight, weight_sha = lm_head(model)
    for prompt, split in jobs:
        out = OUT / name / f"{model_name}_{prompt}_{split}"
        out.mkdir(parents=True, exist_ok=False)
        rows, source = load_rows(prompt, split)
        rows = rows[:limit] if limit else rows
        job_started = time.time()
        hidden_parts, records = [], []
        for start in range(0, len(rows), BATCH):
            batch = rows[start : start + BATCH]
            texts = [
                render_margin_prompt(
                    tokenizer,
                    str(r["prompt"]),
                    enable_thinking=False,
                    assistant_suffix=SUFFIX,
                    decision_prefix="Prediction:",
                )
                for r in batch
            ]
            outputs = llm.embed(texts, use_tqdm=False)
            hidden = torch.tensor(
                np.array([o.outputs.embedding for o in outputs], dtype=np.float32),
                device="cuda",
            )
            if not bool(torch.isfinite(hidden).all()):
                raise ValueError("nonfinite hidden state")
            stats = logit_stats(hidden, weight)
            for r, text, o, s in zip(batch, texts, outputs, stats, strict=True):
                meta = row_meta(r)
                records.append(
                    {
                        **meta,
                        "student_training_row": meta["id"] in student_ids,
                        "prompt_tokens": len(o.prompt_token_ids),
                        "rendered_prompt_sha256": hashlib.sha256(
                            text.encode()
                        ).hexdigest(),
                        **s,
                    }
                )
            hidden_parts.append(hidden.cpu().numpy())
            print(
                f"{model_name} {prompt}:{split} {len(records)}/{len(rows)}", flush=True
            )
        np.save(out / "hidden.npy", np.concatenate(hidden_parts))
        (out / "rows.jsonl").write_text(
            "".join(json.dumps(r, sort_keys=True) + "\n" for r in records)
        )
        summary = {
            "model": model_name,
            "model_path": str(model),
            "prompt": prompt,
            "split": split,
            "input": source,
            "rows": len(records),
            "student_training_rows": sum(r["student_training_row"] for r in records),
            "lm_head_sha256": weight_sha,
            "hidden_size": int(hidden_parts[0].shape[1]),
            "vllm": vllm.__version__,
            "engine_seconds": engine_seconds,
            "seconds": time.time() - job_started,
            "prompt_tokens": sum(r["prompt_tokens"] for r in records),
            "median_outside_01_mass": float(
                np.median([r["outside_01_mass"] for r in records])
            ),
        }
        (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        print("job_complete", json.dumps(summary), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--jobs", nargs="+", required=True, help="prompt:split")
    parser.add_argument("--name", default="extract01")
    parser.add_argument("--limit", type=int, help="smoke test: first N rows")
    args = parser.parse_args()
    jobs = [tuple(j.split(":")) for j in args.jobs]
    if any(j not in INPUTS for j in jobs):
        parser.error(f"jobs must be among {sorted(INPUTS)}")
    run(args.model, jobs, args.name, args.limit)


if __name__ == "__main__":
    main()
