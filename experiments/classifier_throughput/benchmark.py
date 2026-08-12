#!/usr/bin/env python3
"""Benchmark direct binary classification under one or all vLLM conditions."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.classifier_throughput.core import (  # noqa: E402
    median_scores,
    read_jsonl,
    repeat_stability,
    sha256_file,
    summarize_results,
    token_length_summary,
    validate_config,
)
from gleipnir.binary_evaluation import (  # noqa: E402
    binary_token_ids,
    metric_views,
    score_from_output,
)


def write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def rendered_prompts(tokenizer: Any, records: list[dict[str, Any]]) -> list[str]:
    return [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": record["student_prompt"]}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        + "Prediction:"
        for record in records
    ]


def resolved_engine_config(llm: Any) -> dict[str, Any]:
    config = llm.llm_engine.vllm_config
    scheduler = config.scheduler_config
    cache = config.cache_config
    return {
        "performance_mode": str(config.performance_mode),
        "max_num_batched_tokens": scheduler.max_num_batched_tokens,
        "max_num_seqs": scheduler.max_num_seqs,
        "enable_chunked_prefill": scheduler.enable_chunked_prefill,
        "enable_prefix_caching": cache.enable_prefix_caching,
    }


def benchmark_condition(
    config_path: Path,
    config: dict[str, Any],
    condition: dict[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    import pandas as pd
    import torch
    import transformers
    import vllm
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    adapter = Path(os.environ.get("CLASSIFIER_ADAPTER", config["adapter"])).resolve()
    validation = (ROOT / config["validation"]).resolve()
    weights = adapter / "adapter_model.safetensors"
    for required in (validation, adapter / "adapter_config.json", weights):
        if not required.is_file():
            raise FileNotFoundError(required)

    records = read_jsonl(validation)
    tokenizer = AutoTokenizer.from_pretrained(adapter)
    prompts = rendered_prompts(tokenizer, records)
    encoded = tokenizer(prompts, add_special_tokens=False, truncation=False)
    lengths = [len(token_ids) for token_ids in encoded["input_ids"]]
    prompt_tokens = sum(lengths)
    label_token_ids = binary_token_ids(tokenizer)
    sampling = SamplingParams(
        max_tokens=1,
        temperature=0.0,
        logprobs=len(label_token_ids),
        logprob_token_ids=label_token_ids,
        allowed_token_ids=label_token_ids,
    )
    engine_args: dict[str, Any] = {
        "model": config["model"],
        "revision": config["model_revision"],
        "tokenizer": str(adapter),
        "dtype": "bfloat16",
        "tensor_parallel_size": 1,
        "gpu_memory_utilization": float(config["gpu_memory_utilization"]),
        "enable_lora": True,
        "max_lora_rank": 16,
        "max_model_len": int(config["max_model_len"]),
        "performance_mode": condition["performance_mode"],
    }
    if condition.get("max_num_batched_tokens") is not None:
        engine_args["max_num_batched_tokens"] = int(
            condition["max_num_batched_tokens"]
        )

    initialized = time.perf_counter()
    llm = LLM(**engine_args)
    initialization_seconds = time.perf_counter() - initialized
    request = LoRARequest("phoenix-v8.1", 1, str(adapter))
    warmup_rows = min(int(config["warmup_rows"]), len(prompts))
    llm.generate(
        prompts[:warmup_rows],
        sampling,
        use_tqdm=False,
        lora_request=request,
    )
    if not llm.reset_prefix_cache():
        raise RuntimeError("vLLM refused to reset the prefix cache after warmup")

    repeats = []
    score_repeats: list[list[float]] = []
    for repeat in range(int(config["repeats"])):
        started = time.perf_counter()
        outputs = llm.generate(
            prompts,
            sampling,
            use_tqdm=bool(condition["use_tqdm"]),
            lora_request=request,
        )
        elapsed = time.perf_counter() - started
        if any(
            output.prompt != prompt
            for output, prompt in zip(outputs, prompts, strict=True)
        ):
            raise RuntimeError("vLLM returned outputs in an unexpected prompt order")
        scores = [score_from_output(output, label_token_ids) for output in outputs]
        score_repeats.append(scores)
        repeats.append(
            {
                "repeat": repeat,
                "seconds": elapsed,
                "rows_per_second": len(records) / elapsed,
                "prompt_tokens_per_second": prompt_tokens / elapsed,
                "score_sha256_float64": hashlib.sha256(
                    pd.Series(scores).to_numpy(dtype="float64").tobytes()
                ).hexdigest(),
            }
        )
        if not llm.reset_prefix_cache():
            raise RuntimeError("vLLM refused to reset the prefix cache between repeats")

    reference_scores = median_scores(score_repeats)
    frame = pd.DataFrame(
        {
            "dataset": [record["dataset"] for record in records],
            "index": [record["index"] for record in records],
            "label": [int(record["label"]) for record in records],
            "score": reference_scores,
        }
    )
    seconds = [float(repeat["seconds"]) for repeat in repeats]
    result = {
        "condition": condition["name"],
        "requested_engine_config": condition,
        "resolved_engine_config": resolved_engine_config(llm),
        "model": config["model"],
        "model_revision": config["model_revision"],
        "adapter": str(adapter),
        "adapter_sha256": sha256_file(weights),
        "validation": str(validation),
        "validation_sha256": sha256_file(validation),
        "config_sha256": sha256_file(config_path),
        "benchmark_sha256": sha256_file(Path(__file__)),
        "rows": len(records),
        "prompt_tokens": token_length_summary(lengths),
        "initialization_seconds": initialization_seconds,
        "repeats": repeats,
        "repeat_stability": repeat_stability(score_repeats),
        "score_repeats": score_repeats,
        "median_seconds": statistics.median(seconds),
        "median_rows_per_second": len(records) / statistics.median(seconds),
        "median_prompt_tokens_per_second": prompt_tokens
        / statistics.median(seconds),
        "scores": reference_scores,
        "score_sha256_float64": hashlib.sha256(
            frame["score"].to_numpy(dtype="float64").tobytes()
        ).hexdigest(),
        "metrics": metric_views(frame),
        "hardware": {
            "gpu": torch.cuda.get_device_name(0),
            "gpu_memory_bytes": torch.cuda.get_device_properties(0).total_memory,
            "cuda": torch.version.cuda,
        },
        "software": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "vllm": vllm.__version__,
        },
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    }
    write_json_atomic(output_dir / condition["name"] / "result.json", result)
    print(
        f"{condition['name']}: median={result['median_seconds']:.3f}s "
        f"rows/s={result['median_rows_per_second']:.3f} "
        f"prompt_tokens/s={result['median_prompt_tokens_per_second']:.1f}",
        flush=True,
    )
    return result


def run_all(config_path: Path, config: dict[str, Any], output_dir: Path) -> None:
    status: dict[str, Any] = {"completed": [], "failed": {}}
    for condition in config["conditions"]:
        command = [
            sys.executable,
            str(Path(__file__)),
            "--config",
            str(config_path),
            "--output-dir",
            str(output_dir),
            "--condition",
            condition["name"],
        ]
        completed = subprocess.run(command, cwd=ROOT, check=False)
        if completed.returncode == 0:
            status["completed"].append(condition["name"])
        else:
            status["failed"][condition["name"]] = completed.returncode
        write_json_atomic(output_dir / "status.json", status)

    results = []
    for condition in config["conditions"]:
        path = output_dir / condition["name"] / "result.json"
        if path.is_file():
            results.append(json.loads(path.read_text()))
    if not results or results[0]["condition"] != "balanced_current":
        raise RuntimeError("the current baseline did not complete")
    summary = summarize_results(results, config["parity"])
    summary["failures"] = status["failed"]
    write_json_atomic(output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("experiments/classifier_throughput/config.json"),
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("results/classifier_throughput")
    )
    parser.add_argument("--condition")
    args = parser.parse_args()

    config_path = args.config.resolve()
    config = json.loads(config_path.read_text())
    validate_config(config)
    output_dir = args.output_dir.resolve()
    if args.condition is None:
        run_all(config_path, config, output_dir)
        return
    matches = [
        condition
        for condition in config["conditions"]
        if condition["name"] == args.condition
    ]
    if len(matches) != 1:
        raise ValueError(f"unknown condition {args.condition!r}")
    benchmark_condition(config_path, config, matches[0], output_dir)


if __name__ == "__main__":
    main()
