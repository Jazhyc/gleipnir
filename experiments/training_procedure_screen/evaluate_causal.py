#!/usr/bin/env python3
"""Evaluate one final adapter and all retained checkpoints causally."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.adapter_capacity_scaling.prepare import read_jsonl  # noqa: E402
from experiments.adapter_capacity_scaling.run_train import find_job  # noqa: E402
from experiments.deception_distillation.train_student_sft import (  # noqa: E402
    forward_final_token_hidden,
    forward_final_token_logits,
    gated_delta_kernel_modules,
)
from gleipnir.binary_evaluation import (  # noqa: E402
    balanced_smoke_records,
    binary_token_ids,
    metric_views,
)


def adapter_paths(job: dict[str, Any]) -> list[tuple[str, Path]]:
    """Return the final adapter and retained numeric checkpoints."""
    adapter = Path(job["causal_adapter_dir"])
    checkpoints = sorted(
        (
            path
            for path in adapter.glob("checkpoint-*")
            if path.is_dir() and path.name.removeprefix("checkpoint-").isdigit()
        ),
        key=lambda path: int(path.name.removeprefix("checkpoint-")),
    )
    return [("final", adapter), *[(path.name, path) for path in checkpoints]]


def score_adapter(
    model: Any,
    tokenizer: Any,
    tokenized: list[list[int]],
    token_ids: list[int],
    *,
    batch_size: int,
    decision_head_mode: str,
) -> tuple[list[float], float]:
    """Score direct binary margins using selected-position projection."""
    scores = [0.0] * len(tokenized)
    order = sorted(range(len(tokenized)), key=lambda index: len(tokenized[index]))
    started = time.time()
    model.eval()
    with torch.inference_mode():
        for offset in range(0, len(order), batch_size):
            indices = order[offset : offset + batch_size]
            features = [{"input_ids": tokenized[index]} for index in indices]
            batch = tokenizer.pad(features, padding=True, return_tensors="pt")
            batch = {key: value.to(model.device) for key, value in batch.items()}
            if decision_head_mode == "binary_head":
                hidden, _ = forward_final_token_hidden(
                    model, batch["input_ids"], batch["attention_mask"]
                )
                selected = model.get_base_model().decision_head(hidden.float())
            else:
                logits, _ = forward_final_token_logits(
                    model,
                    batch["input_ids"],
                    batch["attention_mask"],
                    "selected_positions",
                )
                selected = logits.index_select(
                    -1, torch.tensor(token_ids, device=logits.device, dtype=torch.long)
                )
            margins = (selected[:, 1].float() - selected[:, 0].float()).clamp(-80, 80)
            probabilities = torch.sigmoid(margins).cpu().tolist()
            for index, probability in zip(indices, probabilities, strict=True):
                scores[index] = float(probability)
    return scores, time.time() - started


def write_result(
    output_dir: Path,
    records: list[dict[str, Any]],
    scores: list[float],
    metadata: dict[str, Any],
    seconds: float,
) -> None:
    frame = pd.DataFrame(
        {
            "dataset": [record["dataset"] for record in records],
            "index": [record["index"] for record in records],
            "label": [int(record["label"]) for record in records],
            "score": scores,
        }
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_json(
        output_dir / "predictions.jsonl",
        orient="records",
        lines=True,
        double_precision=15,
    )
    result = {
        **metadata,
        "score": "causal selected-position probability for literal 1 versus 0",
        "threshold": 0.5,
        "rows": len(frame),
        "seconds": seconds,
        "rows_per_second": len(frame) / seconds,
        "metrics": metric_views(frame),
        "score_sha256_float64": hashlib.sha256(
            frame["score"].to_numpy(dtype="float64").tobytes()
        ).hexdigest(),
    }
    (output_dir / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--jobs",
        type=Path,
        default=Path("results/training_procedure_screen/lambda_jobs.jsonl"),
    )
    parser.add_argument("--job-name", required=True)
    parser.add_argument(
        "--validation",
        type=Path,
        default=Path("data/optimization_regularization_screen/development.jsonl"),
    )
    parser.add_argument("--model", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=4608)
    parser.add_argument("--smoke-rows", type=int)
    args = parser.parse_args()

    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from transformers.utils.import_utils import is_flash_linear_attention_available

    job = find_job(args.jobs.resolve(), args.job_name)
    training_metadata = json.loads(
        (Path(job["causal_adapter_dir"]) / "training_metadata.json").read_text()
    )
    records = read_jsonl(args.validation.resolve())
    if args.smoke_rows is not None:
        records = balanced_smoke_records(records, args.smoke_rows)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": record["student_prompt"]}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        + "Prediction:"
        for record in records
    ]
    tokenized = [
        tokenizer.encode(prompt, add_special_tokens=False)[-args.max_length :]
        for prompt in prompts
    ]
    model_kwargs: dict[str, Any] = {
        "torch_dtype": torch.bfloat16,
        "device_map": {"": torch.cuda.current_device()},
    }
    evaluation_base = str(job.get("evaluation_base", "bf16"))
    if evaluation_base == "nf4":
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    elif evaluation_base != "bf16":
        raise ValueError(f"unknown evaluation base: {evaluation_base}")
    if not is_flash_linear_attention_available():
        raise RuntimeError("causal evaluation requires pinned FLA kernels")
    base = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs)
    decision_head_mode = str(job.get("decision_head_mode", "token_logits"))
    decision_head_init = str(job.get("decision_head_init", "random"))
    if decision_head_mode == "binary_head":
        base.add_module(
            "decision_head",
            torch.nn.Linear(
                int(base.config.hidden_size),
                2,
                bias=decision_head_init == "random",
                device=base.device,
                dtype=torch.float32,
            ),
        )
    paths = adapter_paths(job)
    final_name, final_path = paths[0]
    model = PeftModel.from_pretrained(
        base, final_path.as_posix(), adapter_name=final_name, is_trainable=False
    )
    for adapter_name, adapter_path in paths[1:]:
        model.load_adapter(
            adapter_path.as_posix(), adapter_name=adapter_name, is_trainable=False
        )
    kernels = gated_delta_kernel_modules(model)
    if not kernels or any(not name.startswith("fla.ops.") for name in kernels):
        raise RuntimeError(f"causal evaluation did not bind FLA kernels: {kernels}")
    token_ids = binary_token_ids(tokenizer)
    for adapter_name, adapter_path in paths:
        model.set_adapter(adapter_name)
        scores, seconds = score_adapter(
            model,
            tokenizer,
            tokenized,
            token_ids,
            batch_size=args.batch_size,
            decision_head_mode=decision_head_mode,
        )
        output = Path(job["output_dir"]) / "causal_validation" / adapter_name
        write_result(
            output,
            records,
            scores,
            {
                "job_name": job["job_name"],
                "checkpoint": adapter_name,
                "adapter_path": adapter_path.as_posix(),
                "evaluation_base": evaluation_base,
                "decision_head_mode": decision_head_mode,
                "decision_head_init": decision_head_init,
                "flash_linear_attention": kernels,
                "training_metadata": training_metadata,
            },
            seconds,
        )
        primary = json.loads((output / "result.json").read_text())["metrics"][
            "macro"
        ]["macro"]
        print(
            f"{job['job_name']} {adapter_name}: "
            f"auroc={primary['auroc']:.6f} "
            f"ba={primary['balanced_accuracy']:.6f} seconds={seconds:.1f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
