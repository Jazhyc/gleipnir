"""Capture bounded baseline prefill activity, separate from benchmark timing."""

from __future__ import annotations

import argparse
import gzip
import json
import os
import sys
from collections import Counter
from pathlib import Path

from experiments.local_inference.core import ROOT, canary_rows, read_rows, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--delay", type=int, default=2)
    parser.add_argument("--early-cupti", action="store_true")
    parser.add_argument("--kind", choices=("torch", "cuda"), default="torch")
    parser.add_argument("--record-shapes", action="store_true")
    args = parser.parse_args()
    if args.kind == "cuda" and args.early_cupti:
        parser.error("Do not combine Nsight capture with the torch CUPTI subscriber")
    args.output.mkdir(parents=True, exist_ok=False)
    config_path = Path("experiments/local_inference/iteration32.json")
    config = json.loads(config_path.read_text())
    os.environ.update(config["environment"])
    os.environ["PATH"] = (
        str(Path(sys.executable).parent) + os.pathsep + os.environ["PATH"]
    )
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    from gleipnir.binary_evaluation import binary_token_ids, score_from_output

    rows = read_rows(Path(config["input"]))
    chosen = rows[::4]
    write_json(args.output / "selection.json", chosen)
    write_json(args.output / "config.json", config)
    tokenizer = AutoTokenizer.from_pretrained(ROOT / "merged_bf16")
    ids = binary_token_ids(tokenizer)
    sampling = SamplingParams(
        temperature=0,
        max_tokens=1,
        logprobs=2,
        logprob_token_ids=ids,
        allowed_token_ids=ids,
    )
    profiler_config = {
        "profiler": args.kind,
        "torch_profiler_with_stack": False,
        "torch_profiler_record_shapes": args.record_shapes,
        "ignore_frontend": True,
        "delay_iterations": args.delay,
        "max_iterations": 16,
    }
    if args.kind == "torch":
        profiler_config["torch_profiler_dir"] = str(args.output.resolve())
    write_json(args.output / "profiler_config.json", profiler_config)
    instrumentation = {}
    if args.early_cupti:
        instrumentation["worker_cls"] = (
            "experiments.local_inference.profile_worker.EarlyCuptiWorker"
        )
    write_json(args.output / "instrumentation.json", instrumentation)
    llm = LLM(
        model=str(ROOT / "merged_bf16"),
        profiler_config=profiler_config,
        **instrumentation,
        **config["engine"],
    )
    try:
        warmup = canary_rows(read_rows()) + [max(rows, key=lambda r: r["tokens"])]
        llm.generate([r["prompt"] for r in warmup], sampling, use_tqdm=False)
        print("Capturing at most 16 engine steps from eight frozen rows", flush=True)
        llm.start_profile()
        try:
            outputs = llm.generate([r["prompt"] for r in chosen], sampling)
        finally:
            llm.stop_profile()
    finally:
        llm.llm_engine.engine_core.shutdown()
    write_json(
        args.output / "scores.json",
        [
            {"id": r["id"], "score": score_from_output(o, ids)}
            for r, o in zip(chosen, outputs, strict=True)
        ],
    )
    if args.kind == "cuda":
        print(
            "CUDA capture range complete; validate the external Nsight report",
            flush=True,
        )
        return
    counts = Counter()
    for path in args.output.glob("*.pt.trace.json.gz"):
        with gzip.open(path, "rt") as handle:
            trace = json.load(handle)
        counts.update(e.get("cat", "uncategorized") for e in trace["traceEvents"])
    write_json(args.output / "capture_validation.json", dict(counts))
    if not counts["kernel"]:
        raise RuntimeError(
            "No CUDA kernel events captured; do not use CPU times as GPU times"
        )
    print(f"Captured {counts['kernel']} CUDA kernel events", flush=True)


if __name__ == "__main__":
    main()
