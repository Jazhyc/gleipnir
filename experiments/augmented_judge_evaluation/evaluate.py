"""Bounded FP32-master parity followed by persistent A/B vLLM evaluation."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import os

import numpy as np

from experiments.augmented_judge_evaluation.prepare import (
    BASELINE,
    DATA,
    OUTPUT,
    ROOT,
    configuration,
    verify,
)
from gleipnir.decision_surface import decision_token_ids
from gleipnir.judge_injection_metrics import summarize_preferences
from gleipnir.monitoring_campaign_data import digest, file_hash, read_rows, write_json
from gleipnir.monitoring_scoring import completed_predictions, predictions_for, rendered


def identity(size: str) -> dict:
    manifest = verify()
    result = {
        "manifest_sha256": file_hash(DATA / "manifest.json"),
        "model": configuration()["models"][size],
        "adapters": manifest["adapters"][size],
        "surface": "AB",
        "input_sha256": file_hash(DATA / "test.jsonl"),
    }
    path = OUTPUT / size / "identity.json"
    if path.exists() and json.loads(path.read_text()) != result:
        raise ValueError("evaluation identity drift")
    write_json(path, result)
    return result


def parity_metrics(
    served: list[float],
    reference: list[float],
    base_served: list[float] | None = None,
    base_reference: list[float] | None = None,
) -> dict:
    """Reject nonfinite agreement or zero adapter effects before population scoring."""
    a, b = np.asarray(served), np.asarray(reference)
    if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("parity shape/nonfinite scores")
    result = {
        "correlation": float(np.corrcoef(a, b)[0, 1]),
        "mean_absolute_difference": float(np.abs(a - b).mean()),
    }
    limits = configuration()["parity"]
    result["passed"] = (
        math.isfinite(result["correlation"])
        and result["correlation"] >= limits["min_correlation"]
        and result["mean_absolute_difference"] <= limits["max_mean_absolute_difference"]
    )
    if base_served is not None:
        result["adapter_effect"] = float(np.max(np.abs(a - base_served)))
        result["reference_adapter_effect"] = float(np.max(np.abs(b - base_reference)))
        result["passed"] &= (
            min(result["adapter_effect"], result["reference_adapter_effect"])
            >= limits["min_adapter_effect"]
        )
    return result


def reference(size: str) -> None:
    ident = identity(size)
    dest = OUTPUT / size / "reference.json"
    if dest.exists():
        if json.loads(dest.read_text())["identity"] != ident:
            raise ValueError("reference identity drift")
        return
    for package, version in (
        ("flash-linear-attention", "0.5.2"),
        ("fla-core", "0.5.2"),
        ("causal-conv1d", "1.6.2.post1"),
    ):
        if importlib.metadata.version(package) != version:
            raise RuntimeError("reference kernel version drift")
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers.utils.import_utils import is_flash_linear_attention_available

    from experiments.deception_distillation.train_student_sft import (
        causal_conv1d_kernel_modules,
        gated_delta_kernel_modules,
    )
    from experiments.training_procedure_screen.evaluate_causal import score_adapter

    if not is_flash_linear_attention_available():
        raise RuntimeError("original FLA unavailable")
    config = configuration()["models"][size]
    tokenizer = AutoTokenizer.from_pretrained(config["id"], revision=config["revision"])
    tokenizer.pad_token = tokenizer.eos_token
    rows = read_rows(DATA / "canary.jsonl")
    prompts = [rendered(tokenizer, r, "AB") for r in rows]
    tokens = [tokenizer.encode(p, add_special_tokens=False) for p in prompts]
    model = AutoModelForCausalLM.from_pretrained(
        config["id"],
        revision=config["revision"],
        dtype=torch.bfloat16,
        device_map={"": 0},
        attn_implementation="sdpa",
    )
    kernels, conv = (
        gated_delta_kernel_modules(model),
        causal_conv1d_kernel_modules(model),
    )
    if not kernels or any(not k.startswith("fla.ops.") for k in kernels):
        raise RuntimeError("reference gated-delta fallback")
    if not conv or any(not k.startswith("causal_conv1d.") for k in conv):
        raise RuntimeError("reference convolution fallback")
    ids = decision_token_ids(tokenizer, list("AB"))
    cells = {}
    for name in ("base", *config["adapters"]):
        if name != "base":
            model = PeftModel.from_pretrained(
                model, str(ROOT / config["adapters"][name] / "causal_adapter")
            )
        scores, seconds = score_adapter(
            model,
            tokenizer,
            tokens,
            ids,
            batch_size=1,
            decision_head_mode="token_logits",
        )
        if not all(math.isfinite(s) for s in scores):
            raise ValueError("nonfinite master reference")
        cells[name] = {"scores": scores, "seconds": seconds}
        print(f"reference_complete {size}/{name} {seconds:.1f}s", flush=True)
        if name != "base":
            model = model.unload()
    write_json(
        dest,
        {
            "identity": ident,
            "ids": [r["id"] for r in rows],
            "prompt_sha256": list(map(digest, prompts)),
            "cells": cells,
            "kernels": kernels,
            "convolution": conv,
        },
    )


def serving(size: str) -> None:
    import torch
    import vllm
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    from gleipnir.binary_evaluation import score_from_output

    ident, config = identity(size), configuration()
    model, engine = config["models"][size], config["engine"]
    ref = json.loads((OUTPUT / size / "reference.json").read_text())
    if ref["identity"] != ident:
        raise ValueError("master reference drift")
    tokenizer = AutoTokenizer.from_pretrained(model["id"], revision=model["revision"])
    ids = decision_token_ids(tokenizer, list("AB"))
    sampling = SamplingParams(
        max_tokens=1,
        temperature=0,
        logprobs=2,
        logprob_token_ids=ids,
        allowed_token_ids=ids,
    )
    llm = LLM(
        model=model["id"],
        revision=model["revision"],
        tokenizer_revision=model["revision"],
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
        max_loras=len(model["adapters"]),
        seed=0,
        gdn_prefill_backend=engine["gdn_prefill_backend"],
        logprobs_mode="raw_logprobs",
    )
    requests = {
        name: LoRARequest(name, i + 1, str(ROOT / relative / "model"))
        for i, (name, relative) in enumerate(model["adapters"].items())
    }
    canary = read_rows(DATA / "canary.jsonl")
    prompts = [rendered(tokenizer, r, "AB") for r in canary]
    if ref["ids"] != [r["id"] for r in canary] or ref["prompt_sha256"] != list(
        map(digest, prompts)
    ):
        raise ValueError("master canary prompt drift")
    parity = {
        "identity": ident,
        "cells": {},
        "runtime": {
            "torch": torch.__version__,
            "vllm": vllm.__version__,
            "gpu": torch.cuda.get_device_name(0),
            "cuda": torch.version.cuda,
            "gdn_prefill_backend": engine["gdn_prefill_backend"],
            "cache_paths": {
                k: os.environ.get(k)
                for k in (
                    "HF_HOME",
                    "TRITON_CACHE_DIR",
                    "TORCHINDUCTOR_CACHE_DIR",
                    "VLLM_CACHE_ROOT",
                )
            },
        },
    }
    served = {}
    for name in ("base", *requests):
        out = llm.generate(
            prompts, sampling, lora_request=requests.get(name), use_tqdm=False
        )
        values = [score_from_output(o, ids) for o in out]
        result = parity_metrics(
            values,
            ref["cells"][name]["scores"],
            served.get("base") if name != "base" else None,
            ref["cells"]["base"]["scores"] if name != "base" else None,
        )
        served[name] = values
        parity["cells"][name] = result
        write_json(OUTPUT / size / "serving_parity.json", parity)
        print(f"parity {size}/{name} {json.dumps(result)}", flush=True)
        if not result["passed"]:
            raise RuntimeError("adapter serving parity failed")
    scoring = {"base": None} if model.get("score_base", False) else {}
    scoring.update(requests)
    for name, request in scoring.items():
        predictions_for(
            llm,
            sampling,
            request,
            tokenizer,
            read_rows(DATA / "test.jsonl"),
            "AB",
            OUTPUT / size / name / "preferences.jsonl",
            ident,
            f"{size}/{name}",
            engine=engine,
            status_dir=OUTPUT / size,
        )
    print(f"serving_complete {size}", flush=True)


def report() -> None:
    manifest = verify()
    inputs = read_rows(DATA / "test.jsonl")
    cached_ident = json.loads((BASELINE / "identity.json").read_text())
    original = completed_predictions(
        BASELINE / "original/preferences.jsonl", inputs, cached_ident
    )
    report = {
        "manifest_sha256": file_hash(DATA / "manifest.json"),
        "surface": "AB",
        "rows_per_cell": len(inputs),
        "query_groups": 6,
        "metrics": {"4b/regular": summarize_preferences(original)},
        "cached_regular_4b": {
            "rescored": False,
            "sha256": file_hash(BASELINE / "original/preferences.jsonl"),
        },
        "limitations": [
            "six query groups",
            "construction preference labels",
            "single-seed adapters",
            "no adaptive attack optimization",
        ],
    }
    for size, model in configuration()["models"].items():
        ident = identity(size)
        parity = json.loads((OUTPUT / size / "serving_parity.json").read_text())
        if parity["identity"] != ident or not all(
            c["passed"] for c in parity["cells"].values()
        ):
            raise ValueError("missing/pending serving parity")
        names = (["base"] if model.get("score_base", False) else []) + list(
            model["adapters"]
        )
        for name in names:
            rows = completed_predictions(
                OUTPUT / size / name / "preferences.jsonl", inputs, ident
            )
            cached_prompts = {r["id"]: r["prompt_sha256"] for r in original}
            if any(r["prompt_sha256"] != cached_prompts[r["id"]] for r in rows):
                raise ValueError("fresh and cached A/B prompts differ")
            report["metrics"][size + "/" + name] = summarize_preferences(rows)
    report["baseline_adapters"] = manifest["cached_regular_4b_sha256"]
    write_json(OUTPUT / "summary.json", report)
    print("report_complete five model conditions", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage", choices=("reference", "vllm", "report"), required=True
    )
    parser.add_argument("--size", choices=("4b", "9b"))
    args = parser.parse_args()
    if args.stage == "report":
        report()
    elif args.size is None:
        parser.error("--size is required for scoring")
    elif args.stage == "reference":
        reference(args.size)
    else:
        serving(args.size)


if __name__ == "__main__":
    main()
