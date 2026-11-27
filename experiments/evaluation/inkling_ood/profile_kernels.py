#!/usr/bin/env python3
"""Where does Inkling 2 Mini spend its GPU time when scoring one prompt?

Uses four of the 96 agent trajectories (`data/b200_inference_benchmark/full.json`),
picked by Inkling token count near 2K, 8K, 16K and 32K, rendered at the
primary OOD arm's boundary (effort none). The engine matches the OOD runs
(one sequence, 40,960 maximum length, no prefix caching). Two phases:

1. Timing: each prompt alone, one warmup and five timed requests.
2. Kernels: one torch-profiler trace per prompt; GPU kernel time is summed
   by category (expert GEMMs, MoE routing and quantization, attention,
   short convolutions, dense BF16 GEMMs, norms and elementwise, other), and
   the GPU's idle time inside the request is reported.

Timing only; no labels are read and scores are not used."""

from __future__ import annotations

import argparse
import gzip
import json
import re
import statistics
import time
from collections import Counter
from pathlib import Path

from experiments.evaluation.inkling_ood.agent_bench import agent_rows
from experiments.evaluation.inkling_ood.interface import MODEL_ID, REVISION

TARGETS = (2_048, 8_192, 16_384, 32_768)
REPEATS = 5
# First matching pattern wins; kernel names are lower-cased first.
CATEGORIES = (
    ("attention", r"fa4|flash|attn|attention|fmha|softmax"),
    ("short_conv", r"sconv|conv"),
    ("moe_routing_quant", r"route|routing|topk|permute|combine|scatter|gather|"
        r"swiglu|quantiz|fp4|mxfp8|block_scale|swizzle|expert_plan|counter"),
    ("moe_expert_gemm", r"trtllm|bmm|batched_gemm|batchedgemm|moe"),
    ("dense_gemm", r"gemm|nvjet|cutlass|cublas|sm100_xmma|matmul"),
    ("norm_elementwise", r"norm|rms|elementwise|vectorized|reduce|silu|gelu|"
        r"mul|add|copy|cat|clip|l2|gate|fill"),
)  # fmt: skip


def categorize(name: str) -> str:
    lowered = name.lower()
    for category, pattern in CATEGORIES:
        if re.search(pattern, lowered):
            return category
    return "other"


def pick_rows(rows: list[dict]) -> list[dict]:
    picked = []
    for target in TARGETS:
        best = min(
            (r for r in rows if r not in picked),
            key=lambda r: abs(r["prompt_tokens"] - target),
        )
        picked.append(best)
    return picked


def summarize_trace(path: Path) -> dict:
    with gzip.open(path, "rt") if path.suffix == ".gz" else path.open() as handle:
        trace = json.load(handle)
    kernels = [
        e
        for e in trace.get("traceEvents", [])
        if e.get("ph") == "X" and e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")
    ]
    if not kernels:
        return {"kernels": 0}
    by_category: Counter = Counter()
    by_name: Counter = Counter()
    counts: Counter = Counter()
    for event in kernels:
        duration = float(event["dur"])
        category = (
            "memcpy_memset" if event["cat"] != "kernel" else categorize(event["name"])
        )
        by_category[category] += duration
        by_name[event["name"][:120]] += duration
        counts[category] += 1
    start = min(float(e["ts"]) for e in kernels)
    end = max(float(e["ts"]) + float(e["dur"]) for e in kernels)
    busy = sum(by_category.values())
    return {
        "kernels": len(kernels),
        "gpu_span_ms": (end - start) / 1000,
        "gpu_busy_ms": busy / 1000,
        "gpu_idle_fraction": 1 - busy / (end - start),
        "by_category_ms": {k: v / 1000 for k, v in by_category.most_common()},
        "by_category_share": {k: v / busy for k, v in by_category.most_common()},
        "kernel_counts": dict(counts),
        "top_kernels_ms": [(n, v / 1000) for n, v in by_name.most_common(30)],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    trace_dir = (args.output / "traces").resolve()

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION)
    token_ids = [tokenizer.encode(v, add_special_tokens=False)[0] for v in ("0", "1")]
    rows = pick_rows(agent_rows(tokenizer))
    config = json.loads(
        Path("experiments/evaluation/inkling_ood/inkling_effort_none.json").read_text()
    )
    engine = config["engine"]
    llm = LLM(
        model=MODEL_ID,
        revision=REVISION,
        tokenizer_revision=REVISION,
        dtype="bfloat16",
        tensor_parallel_size=1,
        gpu_memory_utilization=float(engine["gpu_memory_utilization"]),
        max_model_len=int(engine["max_model_len"]),
        max_num_seqs=int(engine["max_num_seqs"]),
        max_num_batched_tokens=int(engine["max_num_batched_tokens"]),
        enable_prefix_caching=False,
        limit_mm_per_prompt=dict(engine["limit_mm_per_prompt"]),
        seed=int(engine["seed"]),
        profiler_config={
            "profiler": "torch",
            "torch_profiler_dir": str(trace_dir),
            "torch_profiler_with_stack": False,
        },
    )
    sampling = SamplingParams(
        max_tokens=1,
        temperature=0.0,
        logprobs=2,
        allowed_token_ids=token_ids,
    )

    def one(prompt: str) -> float:
        before = time.perf_counter()
        llm.generate([prompt], sampling, use_tqdm=False)
        return time.perf_counter() - before

    report: dict = {"model": MODEL_ID, "revision": REVISION, "prompts": []}
    for row in rows:
        one(row["prompt"])  # warmup
        seconds = [one(row["prompt"]) for _ in range(REPEATS)]
        median = statistics.median(seconds)
        entry = {
            "id": row["id"],
            "prompt_tokens": row["prompt_tokens"],
            "median_seconds": median,
            "all_seconds": seconds,
            "prompt_tokens_per_second": row["prompt_tokens"] / median,
        }
        report["prompts"].append(entry)
        print(
            f"{row['prompt_tokens']} tokens: {1000 * median:.1f} ms, "
            f"{entry['prompt_tokens_per_second']:,.0f} tok/s",
            flush=True,
        )

    for entry, row in zip(report["prompts"], rows, strict=True):
        prefix = f"len{row['prompt_tokens']}"
        before = set(trace_dir.rglob("*")) if trace_dir.exists() else set()
        llm.start_profile(prefix)
        one(row["prompt"])
        llm.stop_profile()
        time.sleep(5)
        new = sorted(
            p
            for p in set(trace_dir.rglob("*")) - before
            if p.is_file() and ".json" in p.name and prefix in p.name
        )
        entry["traces"] = [str(p.relative_to(args.output)) for p in new]
        entry["kernel_breakdown"] = [summarize_trace(p) for p in new]
        for item in entry["kernel_breakdown"]:
            if item.get("kernels"):
                print(
                    f"{prefix}: busy {item['gpu_busy_ms']:.1f} of "
                    f"{item['gpu_span_ms']:.1f} ms; "
                    + ", ".join(
                        f"{k} {100 * v:.0f}%"
                        for k, v in item["by_category_share"].items()
                    ),
                    flush=True,
                )
    (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print("PROFILE_DONE", flush=True)


if __name__ == "__main__":
    main()
