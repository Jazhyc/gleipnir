"""Matched three-prompt grid and standard-prompt ID for census-filtered weights."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import time

import numpy as np

from experiments.monitoring_injection_disentanglement import evaluate as baseline
from experiments.monitoring_injection_removal.prepare import (
    CONFIG,
    DATA,
    EVAL_DATA,
    HERE,
    OUTPUT,
    PROMPTS,
    ROOT,
    configuration,
    digest,
    file_hash,
    read_rows,
    verify_preparation,
    write_json,
)

DEST = OUTPUT / "evaluation/4b"


def evaluation_cells() -> list[tuple[str, str]]:
    return [("id", "neutral")] + [
        (split, prompt)
        for prompt in PROMPTS
        for split in ("benchmark", "honest_controls")
    ]


def identities() -> dict:
    verify_preparation()
    config = configuration()
    completed = json.loads((OUTPUT / "4b/filtered/complete.json").read_text())
    if (
        completed["status"] != "trained"
        or completed["steps"] != config["expected_steps"]
        or completed["training_rows"] != config["training_rows"]
    ):
        raise ValueError("filtered training incomplete")
    adapters = {
        layout: f"results/monitoring_injection_removal/4b/filtered/{directory}"
        for layout, directory in (("master", "causal_adapter"), ("serving", "model"))
    }
    for layout, path in tuple(adapters.items()):
        actual = file_hash(ROOT / path / "adapter_model.safetensors")
        if actual != completed[layout + "_sha256"]:
            raise ValueError("filtered adapter checksum drift")
        adapters[layout + "_sha256"] = actual
    previous_identity = baseline.identities()
    if previous_identity["model"] != config["model"]:
        raise ValueError("baseline backbone identity drift")
    identity = {
        "config_sha256": file_hash(CONFIG),
        "manifest_sha256": file_hash(DATA / "manifest.json"),
        "entrypoint_sha256": file_hash(HERE / "evaluate.py"),
        "baseline_identity": previous_identity,
        "baseline_summary_sha256": file_hash(ROOT / config["previous_summary"]),
        "model": config["model"],
        "adapter": adapters,
    }
    target = DEST / "identity.json"
    if target.exists() and json.loads(target.read_text()) != identity:
        raise ValueError("evaluation identity drift")
    write_json(target, identity)
    return identity


def baseline_cells(identity: dict) -> dict:
    """Check all twelve existing grid cells; retain their original score provenance."""
    for weight in baseline.WEIGHTS:
        for prompt in PROMPTS:
            for split in ("benchmark", "honest_controls"):
                if baseline.fresh_cell(weight, prompt):
                    baseline.completed_cell(
                        weight, split, prompt, identity["baseline_identity"]
                    )
                else:
                    baseline.historical_cell(
                        weight,
                        split,
                        prompt,
                        read_rows(EVAL_DATA / split / f"{prompt}.jsonl"),
                        identity["baseline_identity"],
                    )
    prior = json.loads((ROOT / configuration()["previous_summary"]).read_text())[
        "benchmark"
    ]
    return {
        key: {
            **cell,
            "reused": True,
            "source_campaign_reused": cell["reused"],
            "source_campaign": "monitoring-injection-disentanglement-v1",
        }
        for key, cell in prior.items()
    }


def reference() -> None:
    identity = identities()
    target = DEST / "reference.json"
    if target.exists():
        cached = json.loads(target.read_text())
        expected = {f"{w}/{p}" for w in ("base", "filtered") for p in PROMPTS}
        if (
            cached["identity"] == identity
            and set(cached["cells"]) == expected
            and all(
                len(c["scores"]) == 20 and np.isfinite(c["scores"]).all()
                for c in cached["cells"].values()
            )
        ):
            print("Reusing checksum-matched reference", flush=True)
            return
    _, _, binary_token_ids, margin_prompt = baseline.benchmark_tools()
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
    config = configuration()
    tokenizer = AutoTokenizer.from_pretrained(
        config["model"]["id"], revision=config["model"]["revision"]
    )
    tokenizer.pad_token = tokenizer.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        config["model"]["id"],
        revision=config["model"]["revision"],
        dtype=torch.bfloat16,
        device_map={"": 0},
        attn_implementation="sdpa",
    )
    kernels, convolution = (
        gated_delta_kernel_modules(base),
        causal_conv1d_kernel_modules(base),
    )
    if (
        not kernels
        or any(not k.startswith("fla.ops.") for k in kernels)
        or not convolution
        or any(not k.startswith("causal_conv1d.") for k in convolution)
    ):
        raise RuntimeError("reference kernel implementation drift")
    result = {
        "identity": identity,
        "kernels": kernels,
        "convolution": convolution,
        "cells": {},
    }
    tokens = binary_token_ids(tokenizer)
    for weight in ("base", "filtered"):
        model = (
            base
            if weight == "base"
            else PeftModel.from_pretrained(
                base, str(ROOT / identity["adapter"]["master"])
            )
        )
        for prompt in PROMPTS:
            rows = read_rows(EVAL_DATA / "canaries" / f"{prompt}.jsonl")
            rendered = [margin_prompt(tokenizer, r["prompt"]) for r in rows]
            encoded = [tokenizer.encode(p, add_special_tokens=False) for p in rendered]
            if max(map(len, encoded)) >= config["engine"]["max_model_len"]:
                raise ValueError("reference context overflow")
            scores, seconds = score_adapter(
                model,
                tokenizer,
                encoded,
                tokens,
                batch_size=1,
                decision_head_mode="token_logits",
            )
            result["cells"][f"{weight}/{prompt}"] = {
                "ids": [r["id"] for r in rows],
                "scores": scores,
                "rendered_sha256": [digest(p) for p in rendered],
                "seconds": seconds,
            }
            write_json(target, result)
            print(f"reference_complete {weight}/{prompt} {seconds:.1f}s", flush=True)


def completed_cell(split: str, prompt: str, identity: dict):
    _, grid, _, _ = baseline.benchmark_tools()
    path = DEST / "filtered" / f"{split}_{prompt}.jsonl"
    contract = json.loads(path.with_suffix(".contract.json").read_text())
    complete = json.loads(path.with_suffix(".complete.json").read_text())
    rows = read_rows(EVAL_DATA / split / f"{prompt}.jsonl")
    if (
        contract["identity"] != identity
        or (contract["split"], contract["prompt"]) != (split, prompt)
        or contract["input_sha256"] != file_hash(EVAL_DATA / split / f"{prompt}.jsonl")
        or contract["sha256"]
        != grid.digest({k: v for k, v in contract.items() if k != "sha256"})
        or not complete["passed"]
        or complete["contract_sha256"] != contract["sha256"]
        or complete["sha256"] != file_hash(path)
        or complete["rows"] != len(rows)
    ):
        raise ValueError("completed filtered prediction identity drift")
    predictions = read_rows(path)
    grid.validate_saved(predictions, rows, contract["sha256"])
    if len(predictions) != len(rows):
        raise ValueError("filtered prediction coverage drift")
    return path, predictions


def serving() -> None:
    identity = identities()
    reused = baseline_cells(identity)
    write_json(
        DEST / "baseline_reuse.json",
        {
            "summary_sha256": identity["baseline_summary_sha256"],
            "grid_cells": list(reused),
        },
    )
    config, reference_data = (
        configuration(),
        json.loads((DEST / "reference.json").read_text()),
    )
    if reference_data["identity"] != identity:
        raise ValueError("reference identity drift")
    _, grid, binary_token_ids, margin_prompt = baseline.benchmark_tools()
    import torch
    import transformers
    import vllm
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    tokenizer = AutoTokenizer.from_pretrained(
        config["model"]["id"], revision=config["model"]["revision"]
    )
    token_ids = binary_token_ids(tokenizer)
    settings, inputs, encoded = config["engine"], {}, {}
    for split, prompt in [("canaries", p) for p in PROMPTS] + evaluation_cells():
        key = f"{split}/{prompt}"
        rows = read_rows(EVAL_DATA / split / f"{prompt}.jsonl")
        if len({r["id"] for r in rows}) != len(rows):
            raise ValueError("duplicate evaluation identity")
        inputs[key], encoded[key] = rows, []
        for n, row in enumerate(rows, 1):
            encoded[key].append(
                tokenizer.encode(
                    margin_prompt(tokenizer, row["prompt"]), add_special_tokens=False
                )
            )
            if n % 1000 == 0:
                print(f"token_audit {key} {n}/{len(rows)}", flush=True)
        lengths = list(map(len, encoded[key]))
        if max(lengths) >= settings["max_model_len"]:
            raise ValueError(f"evaluation context overflow: {key}")
        write_json(
            DEST / "token_audits" / f"{split}_{prompt}.json",
            {
                "rows": len(rows),
                "maximum": max(lengths),
                "total": sum(lengths),
                "truncated": 0,
                "input_sha256": file_hash(EVAL_DATA / split / f"{prompt}.jsonl"),
            },
        )
    llm = LLM(
        model=config["model"]["id"],
        revision=config["model"]["revision"],
        tokenizer_revision=config["model"]["revision"],
        dtype="bfloat16",
        language_model_only=True,
        tensor_parallel_size=1,
        max_model_len=settings["max_model_len"],
        gpu_memory_utilization=settings["gpu_memory_utilization"],
        max_num_seqs=settings["max_num_seqs"],
        max_num_batched_tokens=settings["max_num_batched_tokens"],
        enable_prefix_caching=True,
        enable_lora=True,
        max_lora_rank=128,
        max_loras=1,
        seed=0,
        gdn_prefill_backend=settings["gdn_prefill_backend"],
        logprobs_mode="raw_logprobs",
    )
    sampling = SamplingParams(
        max_tokens=1,
        temperature=0.0,
        logprobs=2,
        logprob_token_ids=token_ids,
        allowed_token_ids=token_ids,
    )
    request = LoRARequest("filtered", 1, str(ROOT / identity["adapter"]["serving"]))
    runtime = {
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "vllm": vllm.__version__,
        "gpu": torch.cuda.get_device_name(0),
    }
    parity, served = {"runtime": runtime, "cells": {}}, {}
    for weight in ("base", "filtered"):
        for prompt in PROMPTS:
            key, tokens = f"{weight}/{prompt}", encoded[f"canaries/{prompt}"]
            ref = reference_data["cells"][key]
            if ref["ids"] != [r["id"] for r in inputs[f"canaries/{prompt}"]] or ref[
                "rendered_sha256"
            ] != [
                digest(margin_prompt(tokenizer, r["prompt"]))
                for r in inputs[f"canaries/{prompt}"]
            ]:
                raise ValueError("reference canary drift")
            out = llm.generate(
                [{"prompt_token_ids": t} for t in tokens],
                sampling,
                lora_request=None if weight == "base" else request,
                use_tqdm=False,
            )
            scores = [
                grid.extract_score(o, token_ids, t)["score"]
                for o, t in zip(out, tokens, strict=True)
            ]
            metrics = grid.parity_metrics(scores, ref["scores"])
            if weight == "filtered":
                for backend, actual, original in (
                    (
                        "reference",
                        ref["scores"],
                        reference_data["cells"][f"base/{prompt}"]["scores"],
                    ),
                    ("serving", scores, served[f"base/{prompt}"]),
                ):
                    metrics[backend + "_adapter_effect"] = float(
                        np.max(np.abs(np.array(actual) - original))
                    )
                metrics["passed"] &= (
                    min(
                        metrics["reference_adapter_effect"],
                        metrics["serving_adapter_effect"],
                    )
                    >= config["parity"]["min_adapter_effect"]
                )
            metrics["passed"] &= (
                metrics["correlation"] >= config["parity"]["min_correlation"]
                and metrics["mean_absolute_difference"]
                <= config["parity"]["max_mean_absolute_difference"]
            )
            served[key], parity["cells"][key] = scores, {**metrics, "scores": scores}
            write_json(DEST / "serving_parity.json", parity)
            print(f"parity {key} {json.dumps(metrics)}", flush=True)
            if not metrics["passed"]:
                write_json(DEST / f"failed_parity_{time.time_ns()}.json", parity)
                raise RuntimeError(f"serving parity failed: {key}")
    for split, prompt in evaluation_cells():
        key, path = f"{split}/{prompt}", DEST / "filtered" / f"{split}_{prompt}.jsonl"
        rows = inputs[key]
        spec = {
            "identity": identity,
            "runtime": runtime,
            "engine": settings,
            "weight": "filtered",
            "split": split,
            "prompt": prompt,
            "input_sha256": file_hash(EVAL_DATA / split / f"{prompt}.jsonl"),
            "reference_sha256": file_hash(DEST / "reference.json"),
            "scoring": (
                "raw decision logprobs; sigmoid(lp1-lp0); "
                "constrained one token; Prediction:"
            ),
            "decision_token_ids": token_ids,
        }
        contract = grid.digest(spec)
        saved = read_rows(path) if path.exists() else []
        grid.validate_saved(saved, rows, contract)
        write_json(path.with_suffix(".contract.json"), {"sha256": contract, **spec})
        done = {r["id"] for r in saved}
        remaining = [i for i, r in enumerate(rows) if r["id"] not in done]
        started, parity_hash = time.monotonic(), file_hash(DEST / "serving_parity.json")
        with path.open("a") as handle:
            for offset in range(0, len(remaining), settings["batch_rows"]):
                indices = remaining[offset : offset + settings["batch_rows"]]
                tokens = [encoded[key][i] for i in indices]
                outputs = llm.generate(
                    [{"prompt_token_ids": t} for t in tokens],
                    sampling,
                    lora_request=request,
                    use_tqdm=False,
                )
                for i, out, t in zip(indices, outputs, tokens, strict=True):
                    original = rows[i]
                    record = {
                        "id": original["id"],
                        **original["metadata"],
                        **grid.extract_score(out, token_ids, t),
                        "prompt_sha256": digest(original["prompt"]),
                        "contract_sha256": contract,
                        "model_variant": "filtered",
                        "prompt_variant": prompt,
                        "timestamp_unix": time.time(),
                        "raw_returned_logprobs": {
                            str(k): float(v.logprob)
                            for k, v in out.outputs[0].logprobs[0].items()
                        },
                        "completion_tokens": 1,
                        "parity_sha256": parity_hash,
                    }
                    handle.write(json.dumps(record, allow_nan=False) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
                status = {
                    "stage": "scoring",
                    "cell": key,
                    "rows": len(saved) + offset + len(indices),
                    "total": len(rows),
                    "seconds": time.monotonic() - started,
                }
                write_json(DEST / "status.json", status)
                print(json.dumps(status), flush=True)
        scored = read_rows(path)
        grid.validate_saved(scored, rows, contract)
        if len(scored) != len(rows):
            raise ValueError("incomplete fresh evaluation cell")
        write_json(
            path.with_suffix(".complete.json"),
            {
                "passed": True,
                "rows": len(rows),
                "sha256": file_hash(path),
                "contract_sha256": contract,
            },
        )
        print(f"cell_complete filtered/{key}", flush=True)
    summarize()


def summarize() -> None:
    identity = identities()
    report = {
        "identity": identity,
        "benchmark": baseline_cells(identity),
        "id": {},
        "primary_id_prompt": "neutral",
    }
    baseline.benchmark_tools()
    from metrics import summarize as summarize_benchmark
    from metrics_honest_controls import summarize_controls

    from experiments.tool_trajectory_monitoring.benchmark_gpt_oss_ood import (
        summarize as summarize_id,
    )
    from gleipnir.calibration import binary_calibration

    for prompt in PROMPTS:
        path, predictions = completed_cell("benchmark", prompt, identity)
        controls_path, controls = completed_cell("honest_controls", prompt, identity)
        metrics, curves = summarize_benchmark(path)
        honest = summarize_controls(controls, predictions)
        report["benchmark"][f"filtered/{prompt}"] = {
            "reused": False,
            "paths": {"benchmark": str(path), "honest_controls": str(controls_path)},
            "standard": metrics,
            "honest_controls": honest,
            "score_ties": {
                "benchmark": len(predictions) - len({r["score"] for r in predictions}),
                "honest_controls": len(controls) - len({r["score"] for r in controls}),
            },
        }
        write_json(path.with_suffix(".metrics.json"), {**metrics, "roc": curves})
        write_json(controls_path.with_suffix(".metrics.json"), honest)
    path, rows = completed_cell("id", "neutral", identity)
    mapped = [
        {**r, "source": r["source_dataset"], "label": r["ground_truth"]} for r in rows
    ]
    metrics = summarize_id(mapped)
    metrics["calibration"] = binary_calibration(
        [r["label"] for r in mapped], [r["score"] for r in mapped]
    )
    metrics["score_ties"] = len(rows) - len({r["score"] for r in rows})
    report["id"]["neutral"] = metrics
    write_json(path.with_suffix(".metrics.json"), metrics)
    write_json(DEST / "summary.json", report)
    write_json(
        DEST / "status.json",
        {
            "stage": "complete",
            "grid_cells": 15,
            "id_cells": 1,
            "fresh_predictions": 30354,
            "reused_predictions": 109368,
        },
    )
    print("evaluation_complete 15 grid cells, 1 standard-prompt ID cell", flush=True)


if __name__ == "__main__":
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
