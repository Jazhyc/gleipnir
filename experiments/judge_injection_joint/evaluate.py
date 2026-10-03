"""Evaluate the frozen joint adapter with regular-only monitoring and A/B judging.

Explicitly reuse the continuation scorer and its audited original/continued
baselines. Their code and artifacts enter this campaign's frozen identity.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import time
from pathlib import Path

import numpy as np

from experiments.judge_injection_continuation import evaluate as prior
from experiments.judge_injection_joint.prepare import (
    CONFIG,
    DATA,
    OUTPUT,
    PREFERENCES,
    ROOT,
    configuration,
    verify_preparation,
)
from gleipnir.decision_surface import decision_token_ids
from gleipnir.judge_injection import digest
from gleipnir.judge_injection_metrics import summarize_preferences
from gleipnir.monitoring_campaign_data import file_hash, read_rows, write_json

DEST = OUTPUT / "evaluation_regular"
BASELINE = ROOT / "results/judge_injection_continuation/evaluation_regular"
BASELINE_SHA = "621b681b1b78d40dd11fc7eb02d4f9c67cb561ac9d98e80eb4aaa6df5e517620"
BASELINE_REFERENCE_SHA = (
    "b1e985596811b3c685ab68bd293acc7c93701d417eeb4c720b385626e48e3bb8"
)


def adapter_paths() -> dict[str, dict[str, Path]]:
    model = configuration()["model"]
    return {
        "original": {
            "master": ROOT / model["original_master"],
            "serving": ROOT / model["original_serving"],
        },
        "joint": {
            "master": OUTPUT / "4b/joint/causal_adapter",
            "serving": OUTPUT / "4b/joint/model",
        },
    }


def identity() -> dict:
    manifest = verify_preparation()
    complete = json.loads((OUTPUT / "4b/joint/complete.json").read_text())
    if (
        complete["status"] != "trained"
        or complete["steps"] != manifest["expected_steps"]
    ):
        raise ValueError("joint training incomplete")
    if file_hash(BASELINE / "summary.json") != BASELINE_SHA:
        raise ValueError("frozen continuation baseline drift")
    if file_hash(BASELINE / "reference.json") != BASELINE_REFERENCE_SHA:
        raise ValueError("frozen original/base reference drift")
    frozen = json.loads((BASELINE / "summary.json").read_text())["identity"]
    for split in ("canaries", "benchmark", "honest_controls"):
        if (
            file_hash(prior.TRANSFER_DATA / split / "neutral.jsonl")
            != (frozen["transfer_input_sha256"][split + "/neutral"])
        ):
            raise ValueError("matched transfer input drift")
    if file_hash(prior.TRANSFER_DATA / "id/neutral.jsonl") != frozen["id_input_sha256"]:
        raise ValueError("matched ID input drift")
    hashes = {
        name: {
            kind: file_hash(p / "adapter_model.safetensors")
            for kind, p in paths.items()
        }
        for name, paths in adapter_paths().items()
    }
    if hashes["joint"] != {
        key: complete[key + "_sha256"] for key in ("master", "serving")
    }:
        raise ValueError("joint checkpoint identity drift")
    if configuration()["engine"] != prior.configuration()["engine"]:
        raise ValueError("shared scorer engine configuration mismatch")
    result = {
        "config_sha256": file_hash(CONFIG),
        "manifest_sha256": file_hash(DATA / "manifest.json"),
        "adapters": hashes,
        "monitor_prompts": ["neutral"],
        "source_sha256": {
            name: file_hash(ROOT / name)
            for name in (
                "experiments/judge_injection_joint/evaluate.py",
                "experiments/judge_injection_continuation/evaluate.py",
                "src/gleipnir/judge_injection_metrics.py",
            )
        },
        "baseline_summary_sha256": BASELINE_SHA,
        "baseline_reference_sha256": BASELINE_REFERENCE_SHA,
        "input_sha256": {
            str(p.relative_to(ROOT)): file_hash(p)
            for p in (
                PREFERENCES / "test.jsonl",
                PREFERENCES / "canary.jsonl",
                prior.TRANSFER_DATA / "canaries/neutral.jsonl",
                prior.TRANSFER_DATA / "benchmark/neutral.jsonl",
                prior.TRANSFER_DATA / "honest_controls/neutral.jsonl",
                prior.TRANSFER_DATA / "id/neutral.jsonl",
            )
        },
    }
    path = DEST / "identity.json"
    if path.exists() and json.loads(path.read_text()) != result:
        raise ValueError("joint evaluation identity drift")
    write_json(path, result)
    return result


def cohorts() -> dict:
    return {
        "preference": (read_rows(PREFERENCES / "canary.jsonl"), "AB"),
        "monitor/neutral": (
            read_rows(prior.TRANSFER_DATA / "canaries/neutral.jsonl"),
            "01",
        ),
    }


def reference() -> None:
    ident = identity()
    target = DEST / "reference.json"
    if target.exists():
        if json.loads(target.read_text())["identity"] != ident:
            raise ValueError("reference identity drift")
        print("Reused joint reference", flush=True)
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
    model_config = configuration()["model"]
    tokenizer = AutoTokenizer.from_pretrained(
        model_config["id"], revision=model_config["revision"]
    )
    tokenizer.pad_token = tokenizer.eos_token
    old = json.loads((BASELINE / "reference.json").read_text())
    frozen = json.loads((BASELINE / "summary.json").read_text())["identity"]
    if (
        old["identity"] != frozen
        or frozen["adapters"]["original"] != ident["adapters"]["original"]
    ):
        raise ValueError("original reference checkpoint drift")
    result = {
        "identity": ident,
        "cells": {},
        "reused_reference_sha256": file_hash(BASELINE / "reference.json"),
    }
    for name in ("base", "original"):
        for cohort, (rows, surface) in cohorts().items():
            cell = old["cells"][name + "/" + cohort]
            if (
                cell["ids"] != [r.get("id", r.get("index")) for r in rows]
                or cell["prompt_sha256"]
                != [digest(prior.rendered(tokenizer, r, surface)) for r in rows]
                or len(cell["scores"]) != len(rows)
                or any(not math.isfinite(s) or not 0 <= s <= 1 for s in cell["scores"])
            ):
                raise ValueError("original reference cohort drift")
            result["cells"][name + "/" + cohort] = cell
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
    result.update(kernels=kernels, convolution=conv)
    model = PeftModel.from_pretrained(base, str(adapter_paths()["joint"]["master"]))
    for cohort, (rows, surface) in cohorts().items():
        prompts = [prior.rendered(tokenizer, r, surface) for r in rows]
        encoded = [tokenizer.encode(p, add_special_tokens=False) for p in prompts]
        scores, seconds = score_adapter(
            model,
            tokenizer,
            encoded,
            decision_token_ids(tokenizer, list(surface)),
            batch_size=1,
            decision_head_mode="token_logits",
        )
        result["cells"]["joint/" + cohort] = {
            "ids": [r.get("id", r.get("index")) for r in rows],
            "prompt_sha256": list(map(digest, prompts)),
            "scores": scores,
            "seconds": seconds,
        }
        write_json(DEST / "reference_partial.json", result)
        print(f"reference_complete joint/{cohort} {seconds:.1f}s", flush=True)
    write_json(target, result)


def serving() -> None:
    import torch
    import vllm
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    from gleipnir.binary_evaluation import score_from_output

    ident, config = identity(), configuration()
    ref = json.loads((DEST / "reference.json").read_text())
    if ref["identity"] != ident or len(ref["cells"]) != 6:
        raise ValueError("reference identity or coverage drift")
    tokenizer = AutoTokenizer.from_pretrained(
        config["model"]["id"], revision=config["model"]["revision"]
    )
    sampling = {
        surface: SamplingParams(
            max_tokens=1,
            temperature=0,
            logprobs=2,
            logprob_token_ids=decision_token_ids(tokenizer, list(surface)),
            allowed_token_ids=decision_token_ids(tokenizer, list(surface)),
        )
        for surface in ("AB", "01")
    }
    engine = config["engine"]
    llm = LLM(
        model=config["model"]["id"],
        revision=config["model"]["revision"],
        tokenizer_revision=config["model"]["revision"],
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
        max_loras=2,
        seed=0,
        gdn_prefill_backend=engine["gdn_prefill_backend"],
        logprobs_mode="raw_logprobs",
    )
    requests = {
        name: LoRARequest(name, n, str(paths["serving"]))
        for n, (name, paths) in enumerate(adapter_paths().items(), 1)
    }
    parity = {
        "identity": ident,
        "cells": {},
        "runtime": {
            "torch": torch.__version__,
            "vllm": vllm.__version__,
            "gpu": torch.cuda.get_device_name(0),
        },
    }
    served = {}
    for name in ("base", "original", "joint"):
        for cohort, (rows, surface) in cohorts().items():
            prompts = [prior.rendered(tokenizer, r, surface) for r in rows]
            cell = name + "/" + cohort
            reference_cell = ref["cells"][cell]
            if reference_cell["ids"] != [
                r.get("id", r.get("index")) for r in rows
            ] or reference_cell["prompt_sha256"] != list(map(digest, prompts)):
                raise ValueError("parity cohort drift")
            outputs = llm.generate(
                prompts,
                sampling[surface],
                lora_request=requests.get(name),
                use_tqdm=False,
            )
            values = np.array(
                [
                    score_from_output(o, decision_token_ids(tokenizer, list(surface)))
                    for o in outputs
                ]
            )
            eager = np.array(reference_cell["scores"])
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
            if name != "base":
                metrics["adapter_effect"] = float(
                    np.max(np.abs(values - served["base/" + cohort]))
                )
                metrics["reference_adapter_effect"] = float(
                    np.max(np.abs(eager - ref["cells"]["base/" + cohort]["scores"]))
                )
                passed &= (
                    min(metrics["adapter_effect"], metrics["reference_adapter_effect"])
                    >= config["parity"]["min_adapter_effect"]
                )
            served[cell] = values
            parity["cells"][cell] = {**metrics, "passed": passed}
            write_json(DEST / "serving_parity.json", parity)
            print(f"parity {cell} {json.dumps(metrics)} passed={passed}", flush=True)
            if not passed:
                raise RuntimeError("serving parity failed: " + cell)
    populations = [("preferences", PREFERENCES / "test.jsonl", "AB")]
    populations += [
        (split + "_neutral", prior.TRANSFER_DATA / split / "neutral.jsonl", "01")
        for split in ("benchmark", "honest_controls", "id")
    ]
    for name, path, surface in populations:
        prior.predictions_for(
            llm,
            sampling[surface],
            requests["joint"],
            tokenizer,
            read_rows(path),
            surface,
            DEST / "joint" / (name + ".jsonl"),
            ident,
            "joint/" + name,
        )
    summarize()


def summarize() -> None:
    from experiments.monitoring_injection_disentanglement.evaluate import (
        benchmark_tools,
    )
    from experiments.tool_trajectory_monitoring.benchmark_gpt_oss_ood import (
        summarize as summarize_id,
    )
    from gleipnir.calibration import binary_calibration

    benchmark_tools()
    from metrics import summarize as summarize_benchmark
    from metrics_honest_controls import summarize_controls

    ident = identity()
    parity = json.loads((DEST / "serving_parity.json").read_text())
    if (
        parity["identity"] != ident
        or len(parity["cells"]) != 6
        or not all(c["passed"] for c in parity["cells"].values())
    ):
        raise ValueError("report requires all parity gates")
    populations = {
        "preferences": PREFERENCES / "test.jsonl",
        **{
            split + "_neutral": prior.TRANSFER_DATA / split / "neutral.jsonl"
            for split in ("benchmark", "honest_controls", "id")
        },
    }
    scored = {
        name: prior.completed_predictions(
            DEST / "joint" / (name + ".jsonl"), read_rows(path), ident
        )
        for name, path in populations.items()
    }
    baseline = json.loads((BASELINE / "summary.json").read_text())
    # Revalidate both complete preference caches with their frozen identities.
    for name in ("original", "continued"):
        verified = prior.completed_predictions(
            BASELINE / name / "preferences.jsonl",
            read_rows(PREFERENCES / "test.jsonl"),
            baseline["identity"],
        )
        if summarize_preferences(verified) != baseline["preference"][name]:
            raise ValueError("baseline preference metric drift")
    mapped = [
        {**r, "source": r["source_dataset"], "label": r["ground_truth"]}
        for r in scored["id_neutral"]
    ]
    for row in mapped:
        if row["prompt_tokens"] < 1:
            raise ValueError("ID scoring token audit missing")
    id_contract = json.loads((DEST / "joint/id_neutral.contract.json").read_text())
    if sum(r["prompt_tokens"] for r in mapped) != id_contract["total_tokens"]:
        raise ValueError("ID token count mismatch")
    benchmark_path = DEST / "joint/benchmark_neutral.jsonl"
    standard, _ = summarize_benchmark(benchmark_path)
    report = {
        "identity": ident,
        "preference": {
            **baseline["preference"],
            "joint": summarize_preferences(scored["preferences"]),
        },
        "monitoring": {
            "joint": {
                "neutral": {
                    "standard": standard,
                    "honest_controls": summarize_controls(
                        scored["honest_controls_neutral"], scored["benchmark_neutral"]
                    ),
                }
            },
            "continued": baseline["monitoring"],
            "original": baseline["original_transfer_baseline"]["benchmark"],
        },
        "id": {
            "joint": {
                **summarize_id(mapped),
                "calibration": binary_calibration(
                    [r["label"] for r in mapped], [r["score"] for r in mapped]
                ),
            },
            "continued": baseline["id"]["continued"],
            "original": baseline["original_transfer_baseline"]["id"],
        },
        "baseline_provenance": {
            "summary_sha256": BASELINE_SHA,
            "original_id": prior.original_id_baseline(baseline["identity"]),
        },
    }
    write_json(DEST / "summary.json", report)
    write_json(
        DEST / "status.json",
        {
            "stage": "complete",
            "monitor_prompt_cells": 1,
            "fresh_scores": sum(len(r) for r in scored.values()),
        },
    )
    print("evaluation_complete judge_injection_joint", flush=True)


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
