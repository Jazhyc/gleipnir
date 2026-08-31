#!/usr/bin/env python3
"""Eager causal-master parity canary for distilled OOD adapters."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.deception_distillation.train_student_sft import (  # noqa: E402
    gated_delta_kernel_modules,
)
from experiments.tool_trajectory_monitoring.benchmark_distilled_ood import (  # noqa: E402
    DEFAULT_CONFIG,
    adapter_metadata,
    validate_config,
    validate_inputs,
    validate_jobs,
)
from experiments.tool_trajectory_monitoring.benchmark_gpt_oss_ood import (  # noqa: E402
    balanced_canary_rows,
)
from experiments.tool_trajectory_monitoring.benchmark_qwen_ood import (  # noqa: E402
    render_margin_prompt,
)
from experiments.tool_trajectory_monitoring.qwen_reasoning_core import (  # noqa: E402
    load_json,
)
from experiments.training_procedure_screen.evaluate_causal import (  # noqa: E402
    score_adapter,
    write_result,
)
from gleipnir.binary_evaluation import binary_token_ids  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--model-size", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_json(args.config)
    validate_config(config)
    if args.model_size not in config["model_groups"]:
        raise ValueError(f"unknown model group {args.model_size!r}")
    records = validate_inputs(config)
    group = config["model_groups"][args.model_size]
    job_name = str(group["parity_job"])
    job = validate_jobs(config, args.model_size, job_name)[0]
    details = adapter_metadata(job)
    records = balanced_canary_rows(
        records,
        list(config["scope"]["sources"]),
        rows_per_source_label=int(config["engine"]["canary_rows_per_source_label"]),
    )

    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers.utils.import_utils import is_flash_linear_attention_available

    tokenizer = AutoTokenizer.from_pretrained(group["id"], revision=group["revision"])
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    prompt_config = config["prompt"]
    prompts = [
        render_margin_prompt(
            tokenizer,
            str(record["prompt"]),
            enable_thinking=False,
            assistant_suffix=str(prompt_config["assistant_suffix"]),
            decision_prefix=str(prompt_config["decision_prefix"]),
        )
        for record in records
    ]
    tokenized = [
        tokenizer.encode(prompt, add_special_tokens=False) for prompt in prompts
    ]
    maximum = max(map(len, tokenized))
    if maximum >= int(config["engine"]["max_model_len"]):
        raise ValueError(f"parity prompt reaches max_model_len: {maximum}")
    if not is_flash_linear_attention_available():
        raise RuntimeError("causal parity requires pinned FLA kernels")
    base = AutoModelForCausalLM.from_pretrained(
        group["id"],
        revision=group["revision"],
        torch_dtype=torch.bfloat16,
        device_map={"": torch.cuda.current_device()},
    )
    kernels = gated_delta_kernel_modules(base)
    if not kernels or any(not name.startswith("fla.ops.") for name in kernels):
        raise RuntimeError(f"causal parity did not bind FLA kernels: {kernels}")
    token_ids = binary_token_ids(tokenizer)
    flat_records = [
        {
            "dataset": record["metadata"]["source_dataset"],
            "index": record["id"],
            "label": int(record["metadata"]["ground_truth"]),
        }
        for record in records
    ]
    common = {
        "model": group["id"],
        "model_revision": group["revision"],
        "model_size": args.model_size,
        "flash_linear_attention": kernels,
        "parity_rows": len(records),
    }
    scores, seconds = score_adapter(
        base,
        tokenizer,
        tokenized,
        token_ids,
        batch_size=args.batch_size,
        decision_head_mode="token_logits",
    )
    write_result(
        args.output_root / args.model_size / "base" / "base",
        flat_records,
        scores,
        {**common, "job_name": "base", "adapter_path": None},
        seconds,
    )
    model = PeftModel.from_pretrained(
        base,
        str(job["causal_adapter_dir"]),
        adapter_name=job_name,
        is_trainable=False,
    )
    model.set_adapter(job_name)
    scores, seconds = score_adapter(
        model,
        tokenizer,
        tokenized,
        token_ids,
        batch_size=args.batch_size,
        decision_head_mode="token_logits",
    )
    write_result(
        args.output_root / args.model_size / "adapters" / job_name,
        flat_records,
        scores,
        {
            **common,
            "job_name": job_name,
            "adapter_path": str(job["causal_adapter_dir"]),
            "adapter_rebase_source_sha256": details["rebase_manifest"]["source_sha256"],
            "training_metadata_sha256": details["training_metadata_sha256"],
        },
        seconds,
    )
    print(json.dumps({"model_size": args.model_size, "job_name": job_name}))


if __name__ == "__main__":
    main()
