"""Parity-gated persistent-vLLM evaluation of both matched student variants."""

from __future__ import annotations

import argparse
import json
import math
import time
from typing import Any

import yaml

from experiments.student_injection_awareness.prepare import (
    CONFIG,
    DATA,
    OUTPUT,
    VARIANTS,
    digest,
    file_hash,
    read_rows,
    templates,
    write_json,
    write_rows,
)
from experiments.tool_trajectory_monitoring.benchmark_gpt_oss_ood import (
    prediction_row,
    summarize,
)
from experiments.tool_trajectory_monitoring.benchmark_qwen_ood import (
    QWEN_NON_THINKING_ASSISTANT_SUFFIX,
    render_margin_prompt,
)
from gleipnir.binary_evaluation import binary_token_ids
from gleipnir.calibration import binary_calibration


def canary(variant: str, count: int) -> list[dict]:
    """Select deterministic shortest training examples per source and hard label."""
    rows = read_rows(DATA / variant / "student_rows.jsonl")
    selected = []
    groups = sorted({(r["source_dataset"], r["label"]) for r in rows})
    for source, label in groups:
        candidates = sorted(
            (r for r in rows if (r["source_dataset"], r["label"]) == (source, label)),
            key=lambda r: (len(r["student_prompt"]), r["index"]),
        )
        if len(candidates) < count:
            raise ValueError("insufficient parity training examples")
        for row in candidates[:count]:
            selected.append(
                {
                    "id": row["index"],
                    "prompt": row["student_prompt"],
                    "metadata": {
                        "source_dataset": source,
                        "ground_truth": label,
                        "rendered_prompt_sha256": row["student_prompt_sha256"],
                    },
                }
            )
    return selected


def render(tokenizer: Any, rows: list[dict]) -> list[str]:
    prompts = [
        render_margin_prompt(
            tokenizer,
            row["prompt"],
            enable_thinking=False,
            assistant_suffix=QWEN_NON_THINKING_ASSISTANT_SUFFIX,
            decision_prefix="Prediction:",
        )
        for row in rows
    ]
    lengths = [len(tokenizer.encode(p, add_special_tokens=False)) for p in prompts]
    if max(lengths) >= 32768:
        raise ValueError("zero-truncation evaluation gate failed")
    return prompts


def reference(config: dict, size: str) -> None:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers.utils.import_utils import is_flash_linear_attention_available

    from experiments.deception_distillation.train_student_sft import (
        gated_delta_kernel_modules,
    )
    from experiments.training_procedure_screen.evaluate_causal import score_adapter

    spec = config["models"][size]
    if not is_flash_linear_attention_available():
        raise RuntimeError("reference requires pinned FLA kernels")
    tokenizer = AutoTokenizer.from_pretrained(spec["id"], revision=spec["revision"])
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        spec["id"],
        revision=spec["revision"],
        dtype=torch.bfloat16,
        device_map={"": 0},
        attn_implementation="sdpa",
    )
    kernels = gated_delta_kernel_modules(base)
    if not kernels or any(not name.startswith("fla.ops.") for name in kernels):
        raise RuntimeError("reference did not bind FLA kernels")
    ids = binary_token_ids(tokenizer)
    model = None
    for variant in VARIANTS:
        directory = OUTPUT / size / variant
        if not (directory / "complete.json").exists():
            raise ValueError("adapter not completed")
        rows = canary(variant, config["parity"]["rows_per_source_label"])
        prompts = render(tokenizer, rows)
        tokenized = [tokenizer.encode(p, add_special_tokens=False) for p in prompts]
        if model is None:
            scores_base, _ = score_adapter(
                base,
                tokenizer,
                tokenized,
                ids,
                batch_size=1,
                decision_head_mode="token_logits",
            )
            model = PeftModel.from_pretrained(
                base, directory / "causal_adapter", adapter_name=variant
            )
        else:
            with model.disable_adapter():
                scores_base, _ = score_adapter(
                    model,
                    tokenizer,
                    tokenized,
                    ids,
                    batch_size=1,
                    decision_head_mode="token_logits",
                )
            model.load_adapter(directory / "causal_adapter", adapter_name=variant)
        model.set_adapter(variant)
        scores_adapter, _ = score_adapter(
            model,
            tokenizer,
            tokenized,
            ids,
            batch_size=1,
            decision_head_mode="token_logits",
        )
        write_json(
            directory / "parity_reference.json",
            {
                "ids": [r["id"] for r in rows],
                "base": scores_base,
                "adapter": scores_adapter,
                "prompt_sha256": [digest(p) for p in prompts],
                "kernels": kernels,
                "master_sha256": file_hash(
                    directory / "causal_adapter/adapter_model.safetensors"
                ),
            },
        )
        print(f"reference_complete {size} {variant}", flush=True)


def serving(config: dict, size: str) -> None:
    import numpy as np
    import torch
    import vllm
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    from gleipnir.binary_evaluation import score_from_output

    spec = config["models"][size]
    engine = config["engine"]
    manifest = json.loads((DATA / "manifest.json").read_text())
    tokenizer = AutoTokenizer.from_pretrained(spec["id"], revision=spec["revision"])
    ids = binary_token_ids(tokenizer)
    sampling = SamplingParams(
        max_tokens=1,
        temperature=0.0,
        logprobs=2,
        logprob_token_ids=ids,
        allowed_token_ids=ids,
    )
    llm = LLM(
        model=spec["id"],
        revision=spec["revision"],
        dtype="bfloat16",
        language_model_only=True,
        tensor_parallel_size=1,
        max_model_len=engine["max_model_len"],
        gpu_memory_utilization=engine["gpu_memory_utilization"],
        max_num_seqs=engine["max_num_seqs"],
        max_num_batched_tokens=engine["max_num_batched_tokens"],
        enable_prefix_caching=True,
        enable_lora=True,
        max_lora_rank=128,
        max_loras=1,
        seed=0,
        gdn_prefill_backend="flashinfer",
    )
    for number, variant in enumerate(VARIANTS, start=1):
        directory = OUTPUT / size / variant
        rebase = json.loads((directory / "model/rebase_manifest.json").read_text())
        for path, expected in [
            (
                directory / "causal_adapter/adapter_model.safetensors",
                rebase["source_sha256"],
            ),
            (
                directory / "model/adapter_model.safetensors",
                rebase["destination_sha256"],
            ),
        ]:
            if file_hash(path) != expected:
                raise ValueError("adapter checksum drift")
        request = LoRARequest(f"{size}-{variant}", number, str(directory / "model"))
        rows = canary(variant, config["parity"]["rows_per_source_label"])
        prompts = render(tokenizer, rows)
        eager = json.loads((directory / "parity_reference.json").read_text())
        if (
            eager["ids"] != [r["id"] for r in rows]
            or eager["prompt_sha256"] != [digest(p) for p in prompts]
            or eager["master_sha256"] != rebase["source_sha256"]
        ):
            raise ValueError("reference identity drift")
        served = {}
        for name, lora in [("base", None), ("adapter", request)]:
            served[name] = [
                score_from_output(o, ids)
                for o in llm.generate(prompts, sampling, lora_request=lora)
            ]
        parity = {}
        for name in ["base", "adapter"]:
            a, b = np.array(eager[name]), np.array(served[name])
            parity[name] = {
                "mean_absolute_difference": float(np.abs(a - b).mean()),
                "correlation": float(np.corrcoef(a, b)[0, 1]),
            }
        effects = {
            backend: max(
                abs(a - b)
                for a, b in zip(values["base"], values["adapter"], strict=True)
            )
            for backend, values in [("eager", eager), ("vllm", served)]
        }
        limits = config["parity"]
        passed = all(
            math.isfinite(p["correlation"])
            and p["correlation"] >= limits["min_correlation"]
            and p["mean_absolute_difference"] <= limits["max_mean_absolute_difference"]
            for p in parity.values()
        )
        passed = passed and all(
            v >= limits["min_adapter_effect"] for v in effects.values()
        )
        write_json(
            directory / "serving_parity.json",
            {
                "passed": passed,
                "comparisons": parity,
                "effects": effects,
                "reference": eager,
                "served": served,
                "limits": limits,
                "serving_sha256": rebase["destination_sha256"],
            },
        )
        if not passed:
            raise RuntimeError(f"serving parity failed {size} {variant}")
        print(f"serving_parity_passed {size} {variant}", flush=True)
        for split in ["id", "ood"]:
            relative = f"{variant}/{split}.jsonl"
            path = DATA / relative
            if (
                file_hash(path) != manifest["files_sha256"][relative]
                or templates()[variant].template_sha256
                != manifest["template_sha256"][variant]
            ):
                raise ValueError("evaluation input or instruction drift")
            rows = read_rows(path)
            prompts = render(tokenizer, rows)
            contract = digest(
                json.dumps(
                    {
                        "config": config,
                        "input_sha256": file_hash(path),
                        "serving_sha256": rebase["destination_sha256"],
                        "size": size,
                        "variant": variant,
                    },
                    sort_keys=True,
                )
            )
            target = directory / split
            predictions_path = target / "predictions.jsonl"
            existing = read_rows(predictions_path) if predictions_path.exists() else []
            done = {r["id"]: r for r in existing}
            expected = {r["id"]: r for r in rows}
            if len(done) != len(existing) or not set(done).issubset(expected):
                raise ValueError("resumable prediction membership drift")
            for identity, prediction in done.items():
                metadata = expected[identity]["metadata"]
                if (
                    prediction["config_sha256"] != contract
                    or prediction["source_prompt_sha256"]
                    != metadata["rendered_prompt_sha256"]
                    or prediction["label"] != metadata["ground_truth"]
                ):
                    raise ValueError("resumable prediction contract drift")
            pending = [
                (r, p)
                for r, p in zip(rows, prompts, strict=True)
                if r["id"] not in done
            ]
            start = time.time()
            for offset in range(0, len(pending), engine["batch_rows"]):
                batch = pending[offset : offset + engine["batch_rows"]]
                outputs = llm.generate(
                    [p for _, p in batch], sampling, lora_request=request
                )
                for (row, prompt), generated in zip(batch, outputs, strict=True):
                    packed = prediction_row(
                        row, prompt, generated, ids, contract, engine["max_model_len"]
                    )
                    done[packed["id"]] = packed
                write_rows(
                    predictions_path, [done[r["id"]] for r in rows if r["id"] in done]
                )
                print(
                    f"evaluation_progress {size} {variant} {split} "
                    f"{len(done)}/{len(rows)}",
                    flush=True,
                )
            predictions = [done[r["id"]] for r in rows]
            if len(predictions) != len(rows) or not all(
                math.isfinite(r["score"]) for r in predictions
            ):
                raise ValueError("incomplete or nonfinite evaluation")
            views = {"pooled": predictions}
            views.update(
                {
                    source: [r for r in predictions if r["source"] == source]
                    for source in sorted({r["source"] for r in predictions})
                }
            )
            calibration = {
                name: binary_calibration(
                    [r["label"] for r in population],
                    [r["score"] for r in population],
                    10,
                )
                for name, population in views.items()
            }
            write_json(
                target / "result.json",
                {
                    "campaign_id": config["campaign_id"],
                    "size": size,
                    "variant": variant,
                    "model": spec,
                    "input_sha256": file_hash(path),
                    "config_sha256": contract,
                    "template_sha256": manifest["template_sha256"][variant],
                    "adapter_sha256": rebase,
                    "calibration": calibration,
                    "runtime": {
                        "vllm": vllm.__version__,
                        "torch": torch.__version__,
                        "gpu": torch.cuda.get_device_name(0),
                        "seconds_this_invocation": time.time() - start,
                    },
                    **summarize(predictions),
                },
            )
            print(f"evaluation_complete {size} {variant} {split}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", choices=("4b", "9b"), required=True)
    parser.add_argument("--backend", choices=("reference", "vllm"), required=True)
    args = parser.parse_args()
    config = yaml.safe_load(CONFIG.read_text())
    manifest = json.loads((DATA / "manifest.json").read_text())
    if manifest["config_sha256"] != file_hash(CONFIG):
        raise ValueError("prepared evaluation config drift")
    if args.backend == "reference":
        reference(config, args.size)
    else:
        serving(config, args.size)


if __name__ == "__main__":
    main()
