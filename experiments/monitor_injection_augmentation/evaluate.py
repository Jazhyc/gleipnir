"""Score only the new adapter; reuse frozen original-adapter results."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import time
from pathlib import Path

import numpy as np
import yaml

from experiments.monitor_injection_augmentation.prepare import (
    CONFIG,
    DATA,
    OUTPUT,
    ROOT,
)
from experiments.monitor_injection_augmentation.prepare_eval import (
    EVAL_CONFIG,
    EVAL_DATA,
    TRANSFER,
    configuration,
    verify_evaluation,
)
from gleipnir.decision_surface import decision_token_ids
from gleipnir.monitoring_campaign_data import digest, file_hash, read_rows, write_json
from gleipnir.monitoring_scoring import completed_predictions, predictions_for, rendered

DEST = OUTPUT / "evaluation_regular"


def populations() -> dict[str, Path]:
    return {
        **{
            s: TRANSFER / s / "neutral.jsonl"
            for s in ("benchmark", "honest_controls", "id")
        },
        "synthetic_id": EVAL_DATA / "synthetic_id.jsonl",
    }


def adapter_paths() -> dict[str, Path]:
    folder = OUTPUT / "4b/augmented"
    return {"master": folder / "causal_adapter", "serving": folder / "model"}


def identity() -> dict:
    manifest = verify_evaluation()
    complete = json.loads((OUTPUT / "4b/augmented/complete.json").read_text())
    if complete["status"] != "trained" or complete["steps"] != 272:
        raise ValueError("matched one-epoch training incomplete")
    hashes = {
        k: file_hash(p / "adapter_model.safetensors")
        for k, p in adapter_paths().items()
    }
    if any(hashes[k] != complete[k + "_sha256"] for k in hashes):
        raise ValueError("new checkpoint checksum drift")
    if complete["manifest_sha256"] != file_hash(DATA / "manifest.json"):
        raise ValueError("trained population drift")
    execution = json.loads(
        (OUTPUT / "4b/augmented/execution_contract.json").read_text()
    )
    if any(file_hash(ROOT / p) != h for p, h in execution["source_sha256"].items()):
        raise ValueError("launched campaign code drift")
    result = {
        "training_config_sha256": file_hash(CONFIG),
        "evaluation_config_sha256": file_hash(EVAL_CONFIG),
        "evaluation_manifest_sha256": file_hash(EVAL_DATA / "manifest.json"),
        "adapter_sha256": hashes,
        "monitor_prompts": ["neutral"],
        "original_adapter_rescored": False,
        "inputs_sha256": {k: file_hash(p) for k, p in populations().items()},
        "source_sha256": {
            name: file_hash(ROOT / name)
            for name in (
                "experiments/monitor_injection_augmentation/evaluate.py",
                "experiments/monitor_injection_augmentation/metrics.py",
                "src/gleipnir/monitoring_scoring.py",
            )
        },
        "baseline_sha256": manifest["source_sha256"][
            configuration()["baseline_summary"]
        ],
    }
    path = DEST / "identity.json"
    if path.exists() and json.loads(path.read_text()) != result:
        raise ValueError("evaluation identity drift")
    write_json(path, result)
    return result


def reference() -> None:
    ident = identity()
    target = DEST / "reference.json"
    if target.exists():
        if json.loads(target.read_text())["identity"] != ident:
            raise ValueError("reference identity drift")
        return
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers.utils.import_utils import is_flash_linear_attention_available

    from experiments.deception_distillation.train_student_sft import (
        causal_conv1d_kernel_modules,
        gated_delta_kernel_modules,
    )
    from experiments.training_procedure_screen.evaluate_causal import score_adapter

    for package, version in (
        ("flash-linear-attention", "0.5.2"),
        ("fla-core", "0.5.2"),
        ("causal-conv1d", "1.6.2.post1"),
    ):
        if importlib.metadata.version(package) != version:
            raise RuntimeError("reference kernel version drift")
    if not is_flash_linear_attention_available():
        raise RuntimeError("original FLA unavailable")
    model_config = yaml.safe_load(CONFIG.read_text())["model"]
    tokenizer = AutoTokenizer.from_pretrained(
        model_config["id"], revision=model_config["revision"]
    )
    tokenizer.pad_token = tokenizer.eos_token
    rows = read_rows(TRANSFER / "canaries/neutral.jsonl")
    prompts = [rendered(tokenizer, r, "01") for r in rows]
    old = json.loads((ROOT / configuration()["baseline_reference"]).read_text())
    base_cell = old["cells"]["base/monitor/neutral"]
    if base_cell["ids"] != [r["id"] for r in rows] or base_cell[
        "prompt_sha256"
    ] != list(map(digest, prompts)):
        raise ValueError("cached base reference cohort drift")
    if any(not math.isfinite(s) or not 0 <= s <= 1 for s in base_cell["scores"]):
        raise ValueError("invalid cached base reference")
    base = AutoModelForCausalLM.from_pretrained(
        model_config["id"],
        revision=model_config["revision"],
        dtype=torch.bfloat16,
        device_map={"": 0},
        attn_implementation="sdpa",
    )
    kernels, conv = gated_delta_kernel_modules(base), causal_conv1d_kernel_modules(base)
    if (
        not kernels
        or any(not k.startswith("fla.ops.") for k in kernels)
        or not conv
        or any(not k.startswith("causal_conv1d.") for k in conv)
    ):
        raise RuntimeError("reference implementation drift")
    model = PeftModel.from_pretrained(base, str(adapter_paths()["master"]))
    scores, seconds = score_adapter(
        model,
        tokenizer,
        [tokenizer.encode(p, add_special_tokens=False) for p in prompts],
        decision_token_ids(tokenizer, list("01")),
        batch_size=1,
        decision_head_mode="token_logits",
    )
    write_json(
        target,
        {
            "identity": ident,
            "kernels": kernels,
            "convolution": conv,
            "reused_base_reference_sha256": file_hash(
                ROOT / configuration()["baseline_reference"]
            ),
            "cells": {
                "base": base_cell,
                "augmented": {
                    "ids": [r["id"] for r in rows],
                    "prompt_sha256": list(map(digest, prompts)),
                    "scores": scores,
                    "seconds": seconds,
                },
            },
        },
    )
    print(f"reference_complete augmented {seconds:.1f}s", flush=True)


def serving() -> None:
    import torch
    import vllm
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    from gleipnir.binary_evaluation import score_from_output

    ident, config = identity(), configuration()
    ref = json.loads((DEST / "reference.json").read_text())
    if ref["identity"] != ident or set(ref["cells"]) != {"base", "augmented"}:
        raise ValueError("reference coverage or identity drift")
    model = yaml.safe_load(CONFIG.read_text())["model"]
    tokenizer = AutoTokenizer.from_pretrained(model["id"], revision=model["revision"])
    ids = decision_token_ids(tokenizer, list("01"))
    sampling = SamplingParams(
        max_tokens=1,
        temperature=0,
        logprobs=2,
        logprob_token_ids=ids,
        allowed_token_ids=ids,
    )
    engine = config["engine"]
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
        max_loras=1,
        seed=0,
        gdn_prefill_backend=engine["gdn_prefill_backend"],
        logprobs_mode="raw_logprobs",
    )
    request = LoRARequest("augmented", 1, str(adapter_paths()["serving"]))
    parity = {
        "identity": ident,
        "cells": {},
        "runtime": {
            "torch": torch.__version__,
            "vllm": vllm.__version__,
            "gpu": torch.cuda.get_device_name(0),
        },
    }
    rows = read_rows(TRANSFER / "canaries/neutral.jsonl")
    prompts = [rendered(tokenizer, r, "01") for r in rows]
    served = {}
    for name in ("base", "augmented"):
        cell = ref["cells"][name]
        if cell["ids"] != [r["id"] for r in rows] or cell["prompt_sha256"] != list(
            map(digest, prompts)
        ):
            raise ValueError("parity cohort drift")
        outputs = llm.generate(
            prompts,
            sampling,
            lora_request=request if name == "augmented" else None,
            use_tqdm=False,
        )
        values = np.array([score_from_output(o, ids) for o in outputs])
        eager = np.array(cell["scores"])
        metrics = {
            "correlation": float(np.corrcoef(values, eager)[0, 1]),
            "mean_absolute_difference": float(np.abs(values - eager).mean()),
        }
        passed = (
            math.isfinite(metrics["correlation"])
            and metrics["correlation"] >= config["parity"]["min_correlation"]
            and metrics["mean_absolute_difference"]
            <= config["parity"]["max_mean_absolute_difference"]
        )
        if name == "augmented":
            metrics["adapter_effect"] = float(np.max(np.abs(values - served["base"])))
            metrics["reference_adapter_effect"] = float(
                np.max(np.abs(eager - ref["cells"]["base"]["scores"]))
            )
            passed &= (
                min(metrics["adapter_effect"], metrics["reference_adapter_effect"])
                >= config["parity"]["min_adapter_effect"]
            )
        served[name] = values
        parity["cells"][name] = {**metrics, "passed": passed}
        write_json(DEST / "serving_parity.json", parity)
        print(f"parity {name} {metrics} passed={passed}", flush=True)
        if not passed:
            raise RuntimeError("serving parity failed: " + name)
    for name, path in populations().items():
        predictions_for(
            llm,
            sampling,
            request,
            tokenizer,
            read_rows(path),
            "01",
            DEST / "augmented" / (name + ".jsonl"),
            ident,
            name,
            engine=engine,
            status_dir=DEST,
        )
    summarize()


def summarize() -> None:
    from experiments.judge_injection_continuation.evaluate import original_id_baseline
    from experiments.monitor_injection_augmentation.metrics import synthetic_metrics
    from experiments.monitoring_injection_disentanglement.evaluate import (
        benchmark_tools,
    )
    from experiments.tool_trajectory_monitoring.benchmark_gpt_oss_ood import (
        summarize as summarize_id,
    )
    from gleipnir.calibration import binary_calibration

    ident = identity()
    parity = json.loads((DEST / "serving_parity.json").read_text())
    if (
        parity["identity"] != ident
        or set(parity["cells"]) != {"base", "augmented"}
        or not all(c["passed"] for c in parity["cells"].values())
    ):
        raise ValueError("report requires both serving parity gates")
    scored = {
        name: completed_predictions(
            DEST / "augmented" / (name + ".jsonl"), read_rows(path), ident
        )
        for name, path in populations().items()
    }
    baseline = json.loads((ROOT / configuration()["baseline_summary"]).read_text())
    reused_id = original_id_baseline(baseline["identity"])
    benchmark_tools()
    from metrics import summarize as summarize_benchmark
    from metrics_honest_controls import summarize_controls

    standard, _ = summarize_benchmark(DEST / "augmented/benchmark.jsonl")
    mapped = [
        {**r, "source": r["source_dataset"], "label": r["ground_truth"]}
        for r in scored["id"]
    ]
    report = {
        "identity": ident,
        "id": {
            "augmented": {
                **summarize_id(mapped),
                "calibration": binary_calibration(
                    [r["label"] for r in mapped], [r["score"] for r in mapped]
                ),
            },
            "original": baseline["original_transfer_baseline"]["id"],
        },
        "monitoring": {
            "augmented": {
                "neutral": {
                    "standard": standard,
                    "honest_controls": summarize_controls(
                        scored["honest_controls"], scored["benchmark"]
                    ),
                }
            },
            "original": baseline["original_transfer_baseline"]["benchmark"],
        },
        "synthetic_id": synthetic_metrics(scored["id"], scored["synthetic_id"]),
        "baseline_provenance": {
            "summary_sha256": ident["baseline_sha256"],
            "original_id": reused_id,
            "original_adapter_rescored": False,
        },
    }
    write_json(DEST / "summary.json", report)
    write_json(
        DEST / "completion_audit.json",
        {
            "identity": ident,
            "fresh_scores": sum(map(len, scored.values())),
            "files_sha256": {
                name: file_hash(DEST / "augmented" / (name + ".jsonl"))
                for name in scored
            },
            "summary_sha256": file_hash(DEST / "summary.json"),
        },
    )
    write_json(
        DEST / "status.json",
        {"stage": "complete", "fresh_scores": sum(map(len, scored.values()))},
    )
    print("evaluation_complete monitor_injection_augmentation", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage", choices=("reference", "vllm", "summarize"), required=True
    )
    stage = parser.parse_args().stage
    try:
        {"reference": reference, "vllm": serving, "summarize": summarize}[stage]()
    except Exception as exc:
        write_json(
            DEST / f"failure_{stage}_{time.time_ns()}.json",
            {"type": type(exc).__name__, "message": str(exc)},
        )
        raise


if __name__ == "__main__":
    main()
