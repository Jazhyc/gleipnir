"""Matched prompt grid, honest controls and frozen ID evaluation for the new 4B."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

from experiments.monitoring_injection_disentanglement.prepare import (
    DATA,
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

EVAL_CONFIG = HERE / "evaluation_config.json"
DEST = OUTPUT / "evaluation/4b"
WEIGHTS = ("base", "regular", "aggressive", "conservative")
OLD_NAMES = {"aggressive": "injection_aware"}


def benchmark_tools():
    root = (ROOT / configuration()["benchmark_root"]).resolve()
    sys.path.insert(0, str(root))
    sys.path.insert(0, str(root / "scripts"))
    import injeval.grid as grid
    from score_vllm import binary_token_ids, margin_prompt

    return root, grid, binary_token_ids, margin_prompt


def identities() -> dict:
    config = configuration()
    frozen = json.loads(EVAL_CONFIG.read_text())
    if (frozen["model"]["model"], frozen["model"]["revision"]) != (
        config["model"]["id"],
        config["model"]["revision"],
    ):
        raise ValueError("evaluation base identity drift")
    complete = json.loads((OUTPUT / "4b/conservative/complete.json").read_text())
    if complete["status"] != "trained" or complete["steps"] != 272:
        raise ValueError("training is incomplete")
    adapters = {
        ("aggressive" if k == "injection_aware" else k): v
        for k, v in frozen["model"]["adapters"].items()
    }
    adapters["conservative"] = {
        "master": (
            "results/monitoring_injection_disentanglement/4b/conservative/causal_adapter"
        ),
        "serving": "results/monitoring_injection_disentanglement/4b/conservative/model",
        "master_sha256": complete["master_sha256"],
        "serving_sha256": complete["serving_sha256"],
    }
    for adapter in adapters.values():
        for layout in ("master", "serving"):
            if (
                file_hash(ROOT / adapter[layout] / "adapter_model.safetensors")
                != adapter[layout + "_sha256"]
            ):
                raise ValueError("evaluation adapter drift")
    benchmark_root = (ROOT / config["benchmark_root"]).resolve()
    identity = {
        "config_sha256": file_hash(HERE / "config.yaml"),
        "evaluation_config_sha256": file_hash(EVAL_CONFIG),
        "preparation_sha256": file_hash(DATA / "manifest.json"),
        "entrypoint_sha256": file_hash(Path(__file__)),
        "benchmark_code_sha256": {
            p: file_hash(benchmark_root / p)
            for p in (
                "injeval/grid.py",
                "scripts/score_vllm.py",
                "scripts/metrics.py",
                "scripts/metrics_honest_controls.py",
            )
        },
        "model": config["model"],
        "adapters": adapters,
    }
    target = DEST / "identity.json"
    if target.exists() and json.loads(target.read_text()) != identity:
        raise ValueError("frozen evaluation identity drift")
    write_json(target, identity)
    return identity


def fresh_cell(weight: str, prompt: str) -> bool:
    return weight == "conservative" or prompt == "conservative"


def evaluation_cells() -> list[tuple[str, str, str]]:
    """ID uses the trained instruction; only injection populations get a sweep."""
    return [("conservative", "id", "conservative")] + [
        (weight, split, prompt)
        for weight in WEIGHTS
        for prompt in PROMPTS
        if fresh_cell(weight, prompt)
        for split in ("benchmark", "honest_controls")
    ]


def reference() -> None:
    verify_preparation()
    identity = identities()
    target = DEST / "reference.json"
    if target.exists():
        previous = json.loads(target.read_text())
        expected = {
            f"{w}/{p}"
            for w in WEIGHTS
            for p in PROMPTS
            if w == "base" or fresh_cell(w, p)
        }
        if previous["identity"] == identity and set(previous["cells"]) == expected:
            for cell in previous["cells"].values():
                if len(cell["scores"]) != 20 or not np.isfinite(cell["scores"]).all():
                    raise ValueError("cached reference coverage or finiteness drift")
            print("Reusing complete checksum-matched bounded reference", flush=True)
            return
    config = configuration()
    _, _, binary_token_ids, margin_prompt = benchmark_tools()
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
            raise RuntimeError(f"reference requires {package}=={version}")
    if not is_flash_linear_attention_available():
        raise RuntimeError("original FLA unavailable")
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
    kernels = gated_delta_kernel_modules(base)
    convolution = causal_conv1d_kernel_modules(base)
    if not kernels or any(not x.startswith("fla.ops.") for x in kernels):
        raise RuntimeError("reference GDN kernel drift")
    if not convolution or any(not x.startswith("causal_conv1d.") for x in convolution):
        raise RuntimeError("reference convolution drift")
    result = {
        "identity": identity,
        "kernels": kernels,
        "convolution": convolution,
        "cells": {},
    }
    token_ids = binary_token_ids(tokenizer)
    model = None
    for weight in WEIGHTS:
        if weight != "base":
            master = str(ROOT / identity["adapters"][weight]["master"])
            if model is None:
                model = PeftModel.from_pretrained(base, master, adapter_name=weight)
            else:
                model.load_adapter(master, adapter_name=weight)
            model.set_adapter(weight)
        for prompt in PROMPTS:
            if weight != "base" and not fresh_cell(weight, prompt):
                continue
            rows = read_rows(DATA / "canaries" / f"{prompt}.jsonl")
            rendered = [margin_prompt(tokenizer, r["prompt"]) for r in rows]
            tokens = [tokenizer.encode(p, add_special_tokens=False) for p in rendered]
            if max(map(len, tokens)) >= config["engine"]["max_model_len"]:
                raise ValueError("reference context overflow")
            scores, seconds = score_adapter(
                base if weight == "base" else model,
                tokenizer,
                tokens,
                token_ids,
                batch_size=1,
                decision_head_mode="token_logits",
            )
            result["cells"][f"{weight}/{prompt}"] = {
                "ids": [r["id"] for r in rows],
                "scores": scores,
                "rendered_sha256": [digest(p) for p in rendered],
                "seconds": seconds,
            }
            write_json(DEST / "reference.json", result)
            print(f"reference_complete {weight}/{prompt} {seconds:.1f}s", flush=True)


def historical_cell(
    weight: str, split: str, prompt: str, inputs: list[dict], identity: dict
):
    root, grid, _, _ = benchmark_tools()
    old_weight = OLD_NAMES.get(weight, weight)
    old_prompt = OLD_NAMES.get(prompt, prompt)
    key = f"{old_weight}/{split}/{old_prompt}"
    files = json.loads(EVAL_CONFIG.read_text())["historical"][key]
    for p, h in files.items():
        if file_hash(root / p) != h:
            raise ValueError(f"historical cache checksum drift: {p}")
    path = root / next(p for p in files if p.endswith(".jsonl"))
    contract = json.loads(path.with_suffix(".contract.json").read_text())
    complete = json.loads(path.with_suffix(".complete.json").read_text())
    actual_contract = {k: v for k, v in contract.items() if k != "sha256"}
    if grid.digest(actual_contract) != contract["sha256"]:
        raise ValueError("historical contract digest drift")
    if (
        not complete["passed"]
        or complete["sha256"] != file_hash(path)
        or complete["contract_sha256"] != contract["sha256"]
        or complete["rows"] != len(inputs)
        or contract["model_variant"] != old_weight
        or contract["prompt_variant"] != old_prompt
        or contract["base_revision"] != identity["model"]["revision"]
        or contract["serving_sha256"]
        != (
            None if weight == "base" else identity["adapters"][weight]["serving_sha256"]
        )
    ):
        raise ValueError("historical model/prompt/completion drift")
    rows = read_rows(path)
    grid.validate_saved(rows, inputs, contract["sha256"])
    if len(rows) != len(inputs):
        raise ValueError("historical coverage drift")
    return path, rows


def completed_cell(weight: str, split: str, prompt: str, identity: dict):
    """Reject stale, partial or mismatched fresh artifacts before reporting."""
    _, grid, _, _ = benchmark_tools()
    path = DEST / weight / f"{split}_{prompt}.jsonl"
    contract = json.loads(path.with_suffix(".contract.json").read_text())
    complete = json.loads(path.with_suffix(".complete.json").read_text())
    inputs = read_rows(DATA / split / f"{prompt}.jsonl")
    if (
        contract["identity"] != identity
        or (contract["weight"], contract["split"], contract["prompt"])
        != (weight, split, prompt)
        or contract["input_sha256"] != file_hash(DATA / split / f"{prompt}.jsonl")
        or grid.digest({k: v for k, v in contract.items() if k != "sha256"})
        != contract["sha256"]
        or not complete["passed"]
        or complete["contract_sha256"] != contract["sha256"]
        or complete["sha256"] != file_hash(path)
        or complete["rows"] != len(inputs)
    ):
        raise ValueError("fresh completed artifact identity drift")
    rows = read_rows(path)
    grid.validate_saved(rows, inputs, contract["sha256"])
    if len(rows) != len(inputs):
        raise ValueError("fresh completed artifact coverage drift")
    return path, rows


def serving() -> None:
    verify_preparation()
    identity = identities()
    config = configuration()
    settings = config["engine"]
    _, grid, binary_token_ids, margin_prompt = benchmark_tools()
    import torch
    import transformers
    import vllm
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    reference_data = json.loads((DEST / "reference.json").read_text())
    if reference_data["identity"] != identity:
        raise ValueError("reference identity drift")
    tokenizer = AutoTokenizer.from_pretrained(
        config["model"]["id"], revision=config["model"]["revision"]
    )
    token_ids = binary_token_ids(tokenizer)
    inputs, encoded = {}, {}
    for split in ("canaries", "id", "benchmark", "honest_controls"):
        for prompt in PROMPTS:
            if split == "id" and prompt != "conservative":
                continue
            key = f"{split}/{prompt}"
            rows = read_rows(DATA / split / f"{prompt}.jsonl")
            if len({r["id"] for r in rows}) != len(rows):
                raise ValueError("evaluation duplicate IDs")
            inputs[key] = rows
            encoded[key] = []
            for n, row in enumerate(rows, 1):
                encoded[key].append(
                    tokenizer.encode(
                        margin_prompt(tokenizer, row["prompt"]),
                        add_special_tokens=False,
                    )
                )
                if n % 1000 == 0:
                    print(f"token_audit {key} {n}/{len(rows)}", flush=True)
            lengths = list(map(len, encoded[key]))
            if max(lengths) >= settings["max_model_len"]:
                raise ValueError(f"context overflow: {key}; truncation forbidden")
            write_json(
                DEST / "token_audits" / f"{split}_{prompt}.json",
                {
                    "rows": len(rows),
                    "maximum": max(lengths),
                    "total": sum(lengths),
                    "truncated": 0,
                    "input_sha256": file_hash(DATA / split / f"{prompt}.jsonl"),
                },
            )
    # Validate every reused cell before allocating the evaluation engine.
    reused = {}
    for weight in WEIGHTS[:-1]:
        for prompt in PROMPTS[:2]:
            for split in ("benchmark", "honest_controls"):
                path, _ = historical_cell(
                    weight, split, prompt, inputs[f"{split}/{prompt}"], identity
                )
                reused[f"{weight}/{split}/{prompt}"] = {
                    "path": str(path),
                    "sha256": file_hash(path),
                }
    write_json(DEST / "historical_reuse.json", reused)
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
    runtime = {
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "vllm": vllm.__version__,
        "gpu": torch.cuda.get_device_name(0),
    }
    parity = {
        "runtime": runtime,
        "reference_sha256": file_hash(DEST / "reference.json"),
        "cells": {},
    }
    served = {}
    requests = {"base": None}
    for number, weight in enumerate(WEIGHTS[1:], 1):
        requests[weight] = LoRARequest(
            weight, number, str(ROOT / identity["adapters"][weight]["serving"])
        )
    # All gates precede any full evaluation; base canaries also measure adapter effects.
    for weight in WEIGHTS:
        for prompt in PROMPTS:
            if weight != "base" and not fresh_cell(weight, prompt):
                continue
            key = f"{weight}/{prompt}"
            ref = reference_data["cells"][key]
            rows = inputs[f"canaries/{prompt}"]
            if ref["ids"] != [r["id"] for r in rows] or ref["rendered_sha256"] != [
                digest(margin_prompt(tokenizer, r["prompt"])) for r in rows
            ]:
                raise ValueError("canary identity drift")
            tokens = encoded[f"canaries/{prompt}"]
            outputs = llm.generate(
                [{"prompt_token_ids": p} for p in tokens],
                sampling,
                lora_request=requests[weight],
                use_tqdm=False,
            )
            actual = [
                grid.extract_score(o, token_ids, p)["score"]
                for o, p in zip(outputs, tokens, strict=True)
            ]
            metrics = grid.parity_metrics(actual, ref["scores"])
            if weight != "base":
                metrics["reference_adapter_effect"] = float(
                    np.max(
                        np.abs(
                            np.array(ref["scores"])
                            - reference_data["cells"][f"base/{prompt}"]["scores"]
                        )
                    )
                )
                metrics["serving_adapter_effect"] = float(
                    np.max(np.abs(np.array(actual) - served[f"base/{prompt}"]))
                )
                metrics["passed"] &= (
                    min(
                        metrics["reference_adapter_effect"],
                        metrics["serving_adapter_effect"],
                    )
                    >= config["parity"]["min_adapter_effect"]
                )
            served[key] = actual
            parity["cells"][key] = {**metrics, "scores": actual}
            write_json(DEST / "serving_parity.json", parity)
            print(f"parity {key} {json.dumps(metrics)}", flush=True)
            if not metrics["passed"]:
                write_json(DEST / f"failed_parity_{time.time_ns()}.json", parity)
                raise RuntimeError(f"serving parity failed: {key}")
    # Score the trained-prompt ID sanity check first, then fresh benchmark cells.
    for weight, split, prompt in evaluation_cells():
        key = f"{split}/{prompt}"
        rows = inputs[key]
        cell = f"{weight}/{split}/{prompt}"
        contract_data = {
            "identity": identity,
            "runtime": runtime,
            "engine": settings,
            "weight": weight,
            "split": split,
            "prompt": prompt,
            "input_sha256": file_hash(DATA / split / f"{prompt}.jsonl"),
            "scoring": (
                "raw decision-token logprobs; sigmoid(logprob1-logprob0); "
                "one constrained token; nonthinking Prediction:"
            ),
            "reference_sha256": file_hash(DEST / "reference.json"),
        }
        contract = grid.digest(contract_data)
        target = DEST / weight / f"{split}_{prompt}.jsonl"
        saved = read_rows(target) if target.exists() else []
        grid.validate_saved(saved, rows, contract)
        write_json(
            target.with_suffix(".contract.json"), {"sha256": contract, **contract_data}
        )
        done = {r["id"] for r in saved}
        remaining = [i for i, r in enumerate(rows) if r["id"] not in done]
        started = time.monotonic()
        with target.open("a") as handle:
            for offset in range(0, len(remaining), settings["batch_rows"]):
                indices = remaining[offset : offset + settings["batch_rows"]]
                tokens = [encoded[key][i] for i in indices]
                outputs = llm.generate(
                    [{"prompt_token_ids": p} for p in tokens],
                    sampling,
                    lora_request=requests[weight],
                    use_tqdm=False,
                )
                for i, out, tokenized in zip(indices, outputs, tokens, strict=True):
                    original = rows[i]
                    returned = out.outputs[0].logprobs[0]
                    prediction = {
                        "id": original["id"],
                        **original["metadata"],
                        **grid.extract_score(out, token_ids, tokenized),
                        "prompt_sha256": digest(original["prompt"]),
                        "contract_sha256": contract,
                        "model_variant": weight,
                        "prompt_variant": prompt,
                        "timestamp_unix": time.time(),
                        "raw_returned_logprobs": {
                            str(t): float(v.logprob) for t, v in returned.items()
                        },
                        "completion_tokens": 1,
                        "parity_sha256": file_hash(DEST / "serving_parity.json"),
                    }
                    handle.write(json.dumps(prediction, allow_nan=False) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
                progress = {
                    "stage": "scoring",
                    "cell": cell,
                    "rows": len(saved) + offset + len(indices),
                    "total": len(rows),
                    "seconds": time.monotonic() - started,
                }
                write_json(DEST / "status.json", progress)
                print(json.dumps(progress), flush=True)
        complete = read_rows(target)
        grid.validate_saved(complete, rows, contract)
        if len(complete) != len(rows):
            raise ValueError(f"incomplete cell: {cell}")
        write_json(
            target.with_suffix(".complete.json"),
            {
                "passed": True,
                "rows": len(rows),
                "sha256": file_hash(target),
                "contract_sha256": contract,
            },
        )
        print(f"cell_complete {cell}", flush=True)
    summarize()


def summarize() -> None:
    verify_preparation()
    identity = identities()
    _, _, _, _ = benchmark_tools()
    from metrics import summarize as summarize_benchmark
    from metrics_honest_controls import summarize_controls

    from experiments.tool_trajectory_monitoring.benchmark_gpt_oss_ood import (
        summarize as summarize_id,
    )
    from gleipnir.calibration import binary_calibration

    report = {
        "identity": identity,
        "benchmark": {},
        "id": {},
        "primary_id_prompt": "conservative",
    }
    for weight in WEIGHTS:
        for prompt in PROMPTS:
            paths, predictions = {}, {}
            for split in ("benchmark", "honest_controls"):
                if fresh_cell(weight, prompt):
                    path, predictions[split] = completed_cell(
                        weight, split, prompt, identity
                    )
                else:
                    path, predictions[split] = historical_cell(
                        weight,
                        split,
                        prompt,
                        read_rows(DATA / split / f"{prompt}.jsonl"),
                        identity,
                    )
                paths[split] = str(path)
            metrics, curves = summarize_benchmark(paths["benchmark"])
            controls = summarize_controls(
                predictions["honest_controls"], predictions["benchmark"]
            )
            report["benchmark"][f"{weight}/{prompt}"] = {
                "reused": not fresh_cell(weight, prompt),
                "paths": paths,
                "standard": metrics,
                "honest_controls": controls,
                "score_ties": {
                    s: len(r) - len({x["score"] for x in r})
                    for s, r in predictions.items()
                },
            }
            write_json(
                DEST / weight / f"benchmark_{prompt}.metrics.json",
                {**metrics, "roc": curves},
            )
            write_json(
                DEST / weight / f"honest_controls_{prompt}.metrics.json", controls
            )
    for prompt in ("conservative",):
        path, rows = completed_cell("conservative", "id", prompt, identity)
        mapped = [
            {**r, "source": r["source_dataset"], "label": r["ground_truth"]}
            for r in rows
        ]
        metrics = summarize_id(mapped)
        metrics["calibration"] = binary_calibration(
            [r["label"] for r in mapped], [r["score"] for r in mapped]
        )
        metrics["score_ties"] = len(rows) - len({r["score"] for r in rows})
        report["id"][prompt] = metrics
        write_json(path.with_suffix(".metrics.json"), metrics)
    write_json(DEST / "summary.json", report)
    write_json(
        DEST / "status.json",
        {
            "stage": "complete",
            "grid_cells": 12,
            "id_cells": 1,
            "fresh_predictions": 57696,
            "reused_predictions": 54684,
        },
    )
    print("evaluation_complete 12 benchmark cells and 1 ID cell", flush=True)


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
