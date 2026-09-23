"""Parity-gated merged-model vLLM baseline on the frozen subset."""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.local_inference.core import (
    CONFIG,
    DATA,
    ROOT,
    canary_rows,
    compare,
    digest,
    parity_override,
    parity_passes,
    read_rows,
    write_json,
)
from gleipnir.binary_evaluation import binary_token_ids, metric_views
from gleipnir.qwen35_adapter_rebase import sha256_file


def main() -> None:
    process_started = time.perf_counter()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    os.environ.update(config.get("environment", {}))
    import torch
    import transformers
    import vllm
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    input_path = Path(config.get("input", DATA / "subset.jsonl"))
    rows = read_rows(input_path)
    manifest_path = Path(config.get("manifest", DATA / "manifest.json"))
    manifest = json.loads(manifest_path.read_text())
    reference = json.loads((ROOT / "reference.json").read_text())
    subset_hash = sha256_file(input_path)
    reference_input = Path(manifest.get("reference_input", input_path))
    reference_hash = sha256_file(reference_input)
    if not reference["passed"] or reference_hash != reference["subset_sha256"]:
        raise ValueError("reference gate or identity mismatch")
    if reference_hash != manifest.get("reference_input_sha256", subset_hash):
        raise ValueError("reference parent identity mismatch")
    if subset_hash != manifest["subset_sha256"]:
        raise ValueError("subset identity mismatch")
    merge_hash = sha256_file(ROOT / "merged_bf16/merge_manifest.json")
    if merge_hash != reference["merge_manifest_sha256"]:
        raise ValueError("merged artifact identity mismatch")
    merge = json.loads((ROOT / "merged_bf16/merge_manifest.json").read_text())
    for name, expected in merge["files"].items():
        if sha256_file(ROOT / "merged_bf16" / name) != expected:
            raise ValueError(f"merged file changed: {name}")
    output = Path(config.get("output", ROOT / "baseline"))
    output.mkdir(exist_ok=False)
    tokenizer = AutoTokenizer.from_pretrained(ROOT / "merged_bf16")
    ids = binary_token_ids(tokenizer)
    for row in rows:
        if digest(row["prompt"]) != row["prompt_sha256"]:
            raise ValueError("prompt hash mismatch")
        if (
            len(tokenizer.encode(row["prompt"], add_special_tokens=False))
            != row["tokens"]
        ):
            raise ValueError("token count mismatch")
    sampling = SamplingParams(
        temperature=0,
        max_tokens=1,
        logprobs=2,
        logprob_token_ids=ids,
        allowed_token_ids=ids,
    )
    write_json(output / "status.json", {"state": "loading"})
    write_json(output / "launch_config.json", config)
    preparation_seconds = time.perf_counter() - process_started
    started = time.perf_counter()
    llm = LLM(model=str(ROOT / "merged_bf16"), **config["engine"])
    initialization = time.perf_counter() - started
    write_json(output / "initialization.json", {"seconds": initialization})

    def generate(selected, *, progress=False):
        outputs = llm.generate(
            [r["prompt"] for r in selected],
            sampling,
            use_tqdm=progress,
        )
        predictions = []
        for row, item in zip(selected, outputs, strict=True):
            if (
                item.prompt != row["prompt"]
                or len(item.prompt_token_ids) != row["tokens"]
            ):
                raise ValueError("engine input/order mismatch")
            if len(item.outputs) != 1 or len(item.outputs[0].token_ids) != 1:
                raise ValueError("invalid one-token output")
            probabilities = item.outputs[0].logprobs[0]
            values = [float(probabilities[i].logprob) for i in ids]
            if not all(math.isfinite(x) for x in values):
                raise ValueError("nonfinite decision logprobs")
            margin = values[1] - values[0]
            score = 1 / (1 + math.exp(-max(-80, min(80, margin))))
            predictions.append(
                {
                    "id": row["id"],
                    "source": row["source"],
                    "label": row["label"],
                    "prompt_sha256": row["prompt_sha256"],
                    "tokens": row["tokens"],
                    "logprobs": values,
                    "logit_margin": margin,
                    "score": score,
                }
            )
        return predictions

    warmup_started = time.perf_counter()
    canaries = canary_rows(read_rows(reference_input))
    if [r["id"] for r in canaries] != reference["ids"]:
        raise ValueError("canary identity mismatch")
    canary = generate(canaries)
    scores = [r["score"] for r in canary]
    parity = {
        name: compare([r["score"] for r in reference[name]], scores)
        for name in ("master", "merged")
    }
    passed = all(parity_passes(p, config["parity"]) for p in parity.values())
    write_json(
        output / "serving_parity.json",
        {
            "comparisons": parity,
            "passed": passed,
            "diagnostic_override_reason": config.get("diagnostic_parity_override"),
            "predictions": canary,
        },
    )
    override = parity_override(config, passed)
    if override:
        print(
            f"DIAGNOSTIC ONLY: parity failed; authorized override: {override}",
            flush=True,
        )
    longest = max(rows, key=lambda r: r["tokens"])
    write_json(output / "longest_canary.json", generate([longest]))
    warmup_seconds = time.perf_counter() - warmup_started
    write_json(output / "warmup.json", {"seconds": warmup_seconds})
    resolved = llm.llm_engine.vllm_config
    write_json(
        output / "engine_config.json",
        {
            "full_repr": str(resolved),
            "max_num_seqs": resolved.scheduler_config.max_num_seqs,
            "max_num_batched_tokens": resolved.scheduler_config.max_num_batched_tokens,
            "enable_chunked_prefill": resolved.scheduler_config.enable_chunked_prefill,
            "enable_prefix_caching": resolved.cache_config.enable_prefix_caching,
        },
    )
    totals = sum(r["tokens"] for r in rows)
    runs, arrays = [], []
    for repeat in range(config["repeats"]):
        write_json(output / "status.json", {"state": "scoring", "repeat": repeat})
        started = time.perf_counter()
        predictions = generate(rows, progress=True)
        elapsed = time.perf_counter() - started
        write_json(output / f"predictions_{repeat}.json", predictions)
        values = [r["score"] for r in predictions]
        arrays.append(values)
        frame = pd.DataFrame(predictions).rename(columns={"source": "dataset"})
        run = {
            "repeat": repeat,
            "seconds": elapsed,
            "rows_per_second": len(rows) / elapsed,
            "prompt_tokens_per_second": totals / elapsed,
            "metrics": metric_views(frame),
        }
        runs.append(run)
        write_json(output / f"repeat_{repeat}.json", run)
        print(json.dumps({k: v for k, v in run.items() if k != "metrics"}), flush=True)
    matrix = np.asarray(arrays)
    medians = np.median(matrix, axis=0)
    frame = pd.DataFrame(
        {
            "dataset": [r["source"] for r in rows],
            "label": [r["label"] for r in rows],
            "score": medians,
        }
    )
    duration = float(np.median([r["seconds"] for r in runs]))
    result = {
        "config": config,
        "serving_parity_passed": passed,
        "diagnostic_parity_override_applied": override,
        "config_sha256": sha256_file(args.config),
        "subset_sha256": subset_hash,
        "merge_manifest_sha256": merge_hash,
        "reference_sha256": sha256_file(ROOT / "reference.json"),
        "rows": len(rows),
        "prompt_tokens": totals,
        "initialization_seconds": initialization,
        "preparation_seconds": preparation_seconds,
        "warmup_seconds": warmup_seconds,
        "main_wall_seconds": time.perf_counter() - process_started,
        "repeats": runs,
        "median_seconds": duration,
        "median_prompt_tokens_per_second": totals / duration,
        "median_scores": medians.tolist(),
        "metrics": metric_views(frame),
        "repeat_stability": None
        if len(arrays) == 1
        else {
            "max_score_range": float(np.ptp(matrix, axis=0).max()),
            "mean_score_range": float(np.ptp(matrix, axis=0).mean()),
            "threshold_unstable_rows": int(
                ((matrix >= 0.5).any(0) != (matrix >= 0.5).all(0)).sum()
            ),
        },
        "software": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "vllm": vllm.__version__,
        },
        "gpu": torch.cuda.get_device_name(),
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "runner_sha256": sha256_file(Path(__file__)),
    }
    write_json(output / "result.json", result)
    write_json(output / "status.json", {"state": "complete"})
    print(f"Baseline complete: {duration:.2f} seconds/pass", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        write_json(
            ROOT / "last_failure.json",
            {"type": type(error).__name__, "error": str(error)},
        )
        raise
