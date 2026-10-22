"""Score a released BF16 checkpoint through stock ``vllm serve`` (any GPU).

``serve_command`` is the documented recipe: vLLM's pooling runner converts the
causal LM into a two-label classifier over the literal ``0``/``1`` tokens at
the last position, returning raw logits (``use_activation=false``). The
classifier overrides are repeated inside ``text_config`` because Qwen3.5/3.8
wrap the language model, and the GDN state stays FP32 as in the project's
runs. ``main`` scores the frozen ID/OOD workloads against a running server and
compares them with the recorded BF16 rows.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import time
from pathlib import Path

import httpx
import numpy as np

from experiments.release_serving.targets import ROOT, TARGETS
from experiments.release_serving.validate import BATCH_ROWS, labels, macro_pauroc, rows
from gleipnir.data.monitoring import read_rows, write_json

CLASSIFIER = {
    "classifier_from_token": ["0", "1"],
    "method": "no_post_processing",
    "num_labels": 2,
    "is_causal": True,
}
POOLER = {"pooling_type": "LAST", "use_activation": False}
# Stock BF16 references recorded with the project's own BF16 launcher.
BF16_ROWS = {
    "4b": {
        "id": "results/qad-v02-bf16-01/evaluation/id.jsonl",
        "ood": "results/qad-v02-bf16-01/evaluation/ood.jsonl",
    },
    "27b": {
        "id": "results/optimized_gap/bf16_teacher_id.jsonl",
        "ood": "results/optimized_gap/bf16_teacher_ood.jsonl",
    },
}


def serve_command(model: str, revision: str, port: int = 8000) -> list[str]:
    overrides = {
        **CLASSIFIER,
        "text_config": {**CLASSIFIER, "mamba_ssm_dtype": "float32"},
    }
    return [
        "vllm", "serve", model,
        "--revision", revision,
        "--served-model-name", "monitor",
        "--port", str(port),
        "--dtype", "bfloat16",
        "--language-model-only",
        "--max-model-len", "32768",
        "--no-enable-prefix-caching",
        "--runner", "pooling",
        "--convert", "classify",
        "--pooler-config", json.dumps(POOLER),
        "--hf-overrides", json.dumps(overrides),
    ]  # fmt: skip


def score(logits: list[float]) -> float:
    margin = logits[1] - logits[0]
    return (
        1 / (1 + math.exp(-margin)) if margin >= 0 else 1 - 1 / (1 + math.exp(margin))
    )


async def classify(batch: list[dict], port: int) -> list[dict]:
    async with httpx.AsyncClient(
        base_url=f"http://127.0.0.1:{port}", trust_env=False, timeout=600
    ) as client:

        async def one(row: dict) -> dict:
            response = await client.post(
                "/classify",
                json={
                    "model": "monitor",
                    "input": row["prompt"],
                    "add_special_tokens": False,
                    "use_activation": False,
                },
            )
            response.raise_for_status()
            payload = response.json()
            logits = [float(x) for x in payload["data"][0]["probs"]]
            if payload["usage"]["prompt_tokens"] != row["prompt_tokens"]:
                raise ValueError(f"token count differs for {row['id']}")
            return {"id": row["id"], "logits": logits, "score": score(logits)}

        return await asyncio.gather(*(one(r) for r in batch))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", choices=sorted(BF16_ROWS))
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--splits", nargs="+", choices=("id", "ood"))
    args = parser.parse_args()
    target = TARGETS[args.model]
    report = {
        "model": args.model,
        "command": serve_command(target.merged_repo, target.version),
        "splits": {},
    }
    for split, (workload_path, _, _) in target.validation.items():
        if args.splits and split not in args.splits:
            continue
        workload = rows(workload_path)
        started = time.perf_counter()
        scored = []
        for start in range(0, len(workload), BATCH_ROWS):
            scored += asyncio.run(
                classify(workload[start : start + BATCH_ROWS], args.port)
            )
        seconds = time.perf_counter() - started
        new = {r["id"]: r["score"] for r in scored}
        recorded = {
            r["id"]: float(r["score"])
            for r in read_rows(ROOT / BF16_ROWS[args.model][split])
        }
        ids = sorted(new)
        a, b = np.array([new[i] for i in ids]), np.array([recorded[i] for i in ids])
        truth = labels(split)
        report["splits"][split] = {
            "rows": len(ids),
            "score_mae": float(np.abs(a - b).mean()),
            "score_max_abs": float(np.abs(a - b).max()),
            "correlation": float(np.corrcoef(a, b)[0, 1]),
            "pauroc_at_20": macro_pauroc(new, truth, split),
            "recorded_bf16_pauroc_at_20": macro_pauroc(recorded, truth, split),
            "prompt_tokens_per_second": sum(r["prompt_tokens"] for r in workload)
            / seconds,
        }
        print(json.dumps({split: report["splits"][split]}), flush=True)
    write_json(args.output, report)


if __name__ == "__main__":
    main()
