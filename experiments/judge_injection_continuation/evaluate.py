"""Compare original/continued preferences and continued monitoring transfer."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import os
import time
from pathlib import Path

import numpy as np

from experiments.judge_injection_continuation.prepare import (
    CONFIG,
    DATA,
    HERE,
    OUTPUT,
    ROOT,
    configuration,
    verify_preparation,
)
from gleipnir.decision_surface import decision_token_ids
from gleipnir.judge_injection import digest
from gleipnir.judge_injection_metrics import summarize_preferences
from gleipnir.monitoring_campaign_data import file_hash, read_rows, write_json

DEST = OUTPUT / "evaluation"
TRANSFER_DATA = ROOT / "data/monitoring_injection_disentanglement"
PROMPTS = ("neutral", "aggressive", "conservative")


def adapters() -> dict[str, dict[str, Path]]:
    config = configuration()
    return {
        "original": {
            "master": ROOT / config["model"]["initial_adapter"],
            "serving": ROOT / config["model"]["original_serving"],
        },
        "continued": {
            "master": OUTPUT / "4b/continued/causal_adapter",
            "serving": OUTPUT / "4b/continued/model",
        },
    }


def identity() -> dict:
    manifest = verify_preparation()
    complete = json.loads((OUTPUT / "4b/continued/complete.json").read_text())
    if (
        complete["status"] != "trained"
        or complete["steps"] != manifest["expected_steps"]
    ):
        raise ValueError("continued training incomplete")
    hashes = {
        name: {
            layout: file_hash(path / "adapter_model.safetensors")
            for layout, path in locations.items()
        }
        for name, locations in adapters().items()
    }
    if hashes["continued"] != {
        k: complete[k + "_sha256"] for k in ("master", "serving")
    }:
        raise ValueError("continued adapter identity drift")
    result = {
        "config_sha256": file_hash(CONFIG),
        "manifest_sha256": file_hash(DATA / "manifest.json"),
        "entrypoint_sha256": file_hash(HERE / "evaluate.py"),
        "adapters": hashes,
        "transfer_input_sha256": {
            f"{split}/{prompt}": file_hash(TRANSFER_DATA / split / f"{prompt}.jsonl")
            for split in ("canaries", "benchmark", "honest_controls")
            for prompt in PROMPTS
        },
        "id_input_sha256": file_hash(TRANSFER_DATA / "id/neutral.jsonl"),
        "original_transfer_summary_sha256": file_hash(
            ROOT / "results/monitoring_injection_removal/evaluation/4b/summary.json"
        ),
        "original_id_summary_sha256": file_hash(
            ROOT / "results/student_injection_awareness/4b/regular/id/result.json"
        ),
    }
    path = DEST / "identity.json"
    if path.exists() and json.loads(path.read_text()) != result:
        raise ValueError("evaluation identity drift")
    write_json(path, result)
    return result


def rendered(tokenizer, row: dict, surface: str) -> str:
    text = row["student_prompt"] if surface == "AB" else row["prompt"]
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": text}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    ) + ("" if surface == "AB" else "Prediction:")


def cohorts() -> dict[str, tuple[list[dict], str]]:
    return {
        "preference": (read_rows(DATA / "canary.jsonl"), "AB"),
        **{
            "monitor/" + p: (read_rows(TRANSFER_DATA / "canaries" / f"{p}.jsonl"), "01")
            for p in PROMPTS
        },
    }


def reference() -> None:
    ident = identity()
    target = DEST / "reference.json"
    if target.exists() and json.loads(target.read_text())["identity"] == ident:
        cached = json.loads(target.read_text())["cells"]
        expected_cells = {
            name + "/" + cohort: rows
            for name in ("base", "original", "continued")
            for cohort, (rows, _) in cohorts().items()
        }
        if set(cached) != set(expected_cells):
            raise ValueError("reference cell coverage drift")
        for cell, rows in expected_cells.items():
            if (
                cached[cell]["ids"] != [r.get("id", r.get("index")) for r in rows]
                or len(cached[cell]["scores"]) != len(rows)
                or len(cached[cell]["prompt_sha256"]) != len(rows)
                or any(
                    not math.isfinite(s) or not 0 <= s <= 1
                    for s in cached[cell]["scores"]
                )
            ):
                raise ValueError("reference score/coverage drift")
        print("Reused bounded reference", flush=True)
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
    spec = configuration()["model"]
    tokenizer = AutoTokenizer.from_pretrained(spec["id"], revision=spec["revision"])
    tokenizer.pad_token = tokenizer.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        spec["id"],
        revision=spec["revision"],
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
    result = {"identity": ident, "kernels": kernels, "convolution": conv, "cells": {}}
    model = None
    for name in ("base", "original", "continued"):
        if name != "base":
            path = str(adapters()[name]["master"])
            if model is None:
                model = PeftModel.from_pretrained(base, path, adapter_name=name)
            else:
                model.load_adapter(path, adapter_name=name)
            model.set_adapter(name)
        for cohort, (rows, surface) in cohorts().items():
            prompts = [rendered(tokenizer, r, surface) for r in rows]
            encoded = [tokenizer.encode(p, add_special_tokens=False) for p in prompts]
            scores, seconds = score_adapter(
                base if name == "base" else model,
                tokenizer,
                encoded,
                decision_token_ids(tokenizer, list(surface)),
                batch_size=1,
                decision_head_mode="token_logits",
            )
            result["cells"][name + "/" + cohort] = {
                "ids": [r.get("id", r.get("index")) for r in rows],
                "prompt_sha256": list(map(digest, prompts)),
                "scores": scores,
                "seconds": seconds,
            }
            write_json(target.with_name("reference_partial.json"), result)
            print(f"reference_complete {name}/{cohort} {seconds:.1f}s", flush=True)
    write_json(target, result)


def predictions_for(
    llm,
    sampling,
    request,
    tokenizer,
    rows: list[dict],
    surface: str,
    path: Path,
    ident: dict,
    cell: str,
) -> list[dict]:
    from gleipnir.binary_evaluation import logprob_value

    ids = decision_token_ids(tokenizer, list(surface))
    prompts = [rendered(tokenizer, row, surface) for row in rows]
    encoded = [tokenizer.encode(p, add_special_tokens=False) for p in prompts]
    lengths = list(map(len, encoded))
    if max(lengths) >= configuration()["engine"]["max_model_len"]:
        raise ValueError("evaluation input would truncate")
    expected = {str(r.get("id", r.get("index"))): r for r in rows}
    prompt_hashes = {
        str(r.get("id", r.get("index"))): digest(p)
        for r, p in zip(rows, prompts, strict=True)
    }
    if len(expected) != len(rows):
        raise ValueError("duplicate input identity")
    contract = digest(
        json.dumps(
            {
                "identity": ident,
                "cell": cell,
                "surface": surface,
                "input_prompt_hashes": list(map(digest, prompts)),
                "decision_ids": ids,
                "raw_logprobs": True,
            },
            sort_keys=True,
        )
    )
    saved = read_rows(path) if path.exists() else []
    done = {}
    for row in saved:
        key = str(row["id"])
        if (
            key not in expected
            or key in done
            or row["contract_sha256"] != contract
            or row["label"]
            != expected[key].get(
                "label", expected[key].get("metadata", {}).get("ground_truth")
            )
            or not math.isfinite(row["score"])
            or row["prompt_sha256"] != prompt_hashes[key]
        ):
            raise ValueError("cached prediction identity drift")
        raw = row["raw_decision_logprobs"]
        if any(str(t) not in raw or not math.isfinite(raw[str(t)]) for t in ids):
            raise ValueError("cached decision logprobs invalid")
        delta = raw[str(ids[1])] - raw[str(ids[0])]
        score = 1 / (1 + math.exp(-max(-80, min(80, delta))))
        if abs(score - row["score"]) > 1e-12:
            raise ValueError("cached score disagrees with logprobs")
        done[key] = row
    write_json(
        path.with_suffix(".contract.json"),
        {
            "sha256": contract,
            "identity": ident,
            "surface": surface,
            "rows": len(rows),
            "decision_ids": ids,
            "maximum_tokens": max(lengths),
            "total_tokens": sum(lengths),
            "truncated": 0,
        },
    )
    remaining = [
        i for i, r in enumerate(rows) if str(r.get("id", r.get("index"))) not in done
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        for offset in range(0, len(remaining), configuration()["engine"]["batch_rows"]):
            indices = remaining[
                offset : offset + configuration()["engine"]["batch_rows"]
            ]
            outputs = llm.generate(
                [{"prompt_token_ids": encoded[i]} for i in indices],
                sampling,
                lora_request=request,
                use_tqdm=False,
            )
            for i, out in zip(indices, outputs, strict=True):
                if (
                    len(out.outputs[0].token_ids) != 1
                    or out.outputs[0].token_ids[0] not in ids
                    or len(out.prompt_token_ids) != lengths[i]
                ):
                    raise ValueError(
                        "one-token response or no-truncation contract failed"
                    )
                raw = {
                    str(k): logprob_value(v)
                    for k, v in out.outputs[0].logprobs[0].items()
                }
                if any(
                    str(token) not in raw or not math.isfinite(raw[str(token)])
                    for token in ids
                ):
                    raise ValueError("missing/nonfinite decision logprobs")
                difference = raw[str(ids[1])] - raw[str(ids[0])]
                score = 1 / (1 + math.exp(-max(-80, min(80, difference))))
                original = rows[i]
                metadata = original.get("metadata", {})
                row = {
                    **{
                        k: v
                        for k, v in original.items()
                        if k
                        not in (
                            "prompt",
                            "student_prompt",
                            "student_target",
                            "metadata",
                        )
                    },
                    **metadata,
                    "id": str(original.get("id", original.get("index"))),
                    "label": original.get("label", metadata.get("ground_truth")),
                    "score": score,
                    "p_B" if surface == "AB" else "p_harmful": score,
                    "raw_decision_logprobs": {
                        str(token): raw[str(token)] for token in ids
                    },
                    "raw_returned_logprobs": raw,
                    "prompt_sha256": digest(prompts[i]),
                    "contract_sha256": contract,
                    "timestamp_unix": time.time(),
                }
                handle.write(json.dumps(row, allow_nan=False) + "\n")
                done[row["id"]] = row
            handle.flush()
            os.fsync(handle.fileno())
            write_json(
                DEST / "status.json",
                {
                    "stage": "scoring",
                    "cell": cell,
                    "rows": len(done),
                    "total": len(rows),
                },
            )
            print(f"evaluation_progress {cell} {len(done)}/{len(rows)}", flush=True)
    result = [done[str(r.get("id", r.get("index")))] for r in rows]
    write_json(
        path.with_suffix(".complete.json"),
        {
            "rows": len(result),
            "passed": True,
            "sha256": file_hash(path),
            "contract_sha256": contract,
        },
    )
    return result


def serving() -> None:
    import torch
    import vllm
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    from gleipnir.binary_evaluation import score_from_output

    ident, config = identity(), configuration()
    ref = json.loads((DEST / "reference.json").read_text())
    if ref["identity"] != ident:
        raise ValueError("reference identity drift")
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
        for n, (name, paths) in enumerate(adapters().items(), 1)
    }
    parity, served = (
        {
            "identity": ident,
            "cells": {},
            "runtime": {
                "torch": torch.__version__,
                "vllm": vllm.__version__,
                "gpu": torch.cuda.get_device_name(0),
            },
        },
        {},
    )
    for name in ("base", "original", "continued"):
        for cohort, (rows, surface) in cohorts().items():
            prompts = [rendered(tokenizer, r, surface) for r in rows]
            cell = name + "/" + cohort
            original = ref["cells"][cell]
            if original["ids"] != [
                r.get("id", r.get("index")) for r in rows
            ] or original["prompt_sha256"] != list(map(digest, prompts)):
                raise ValueError("parity cohort drift")
            out = llm.generate(
                prompts,
                sampling[surface],
                lora_request=requests.get(name),
                use_tqdm=False,
            )
            values = np.array(
                [
                    score_from_output(o, decision_token_ids(tokenizer, list(surface)))
                    for o in out
                ]
            )
            eager = np.array(original["scores"])
            metrics = {
                "correlation": float(np.corrcoef(values, eager)[0, 1]),
                "mean_absolute_difference": float(np.abs(values - eager).mean()),
            }
            if name != "base":
                metrics["adapter_effect"] = float(
                    np.max(np.abs(values - served["base/" + cohort]))
                )
                metrics["reference_adapter_effect"] = float(
                    np.max(np.abs(eager - ref["cells"]["base/" + cohort]["scores"]))
                )
            passed = (
                math.isfinite(metrics["correlation"])
                and metrics["correlation"] >= config["parity"]["min_correlation"]
                and metrics["mean_absolute_difference"]
                <= config["parity"]["max_mean_absolute_difference"]
            )
            if name != "base":
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
    for name in ("original", "continued"):
        rows = predictions_for(
            llm,
            sampling["AB"],
            requests[name],
            tokenizer,
            read_rows(DATA / "test.jsonl"),
            "AB",
            DEST / name / "preferences.jsonl",
            ident,
            name + "/preferences",
        )
        write_json(
            DEST / name / "preferences.metrics.json", summarize_preferences(rows)
        )
    for prompt in PROMPTS:
        for split in ("benchmark", "honest_controls"):
            rows = read_rows(TRANSFER_DATA / split / f"{prompt}.jsonl")
            predictions_for(
                llm,
                sampling["01"],
                requests["continued"],
                tokenizer,
                rows,
                "01",
                DEST / "continued" / f"{split}_{prompt}.jsonl",
                ident,
                f"continued/{split}/{prompt}",
            )
    predictions_for(
        llm,
        sampling["01"],
        requests["continued"],
        tokenizer,
        read_rows(TRANSFER_DATA / "id/neutral.jsonl"),
        "01",
        DEST / "continued/id_neutral.jsonl",
        ident,
        "continued/id/neutral",
    )
    summarize()


def summarize() -> None:
    from experiments.monitoring_injection_disentanglement import (
        evaluate as baseline_eval,
    )
    from experiments.monitoring_injection_disentanglement.evaluate import (
        benchmark_tools,
    )

    benchmark_tools()
    baseline_identity = baseline_eval.identities()
    for prompt in PROMPTS:
        for split in ("benchmark", "honest_controls"):
            if baseline_eval.fresh_cell("regular", prompt):
                baseline_eval.completed_cell(
                    "regular", split, prompt, baseline_identity
                )
            else:
                baseline_eval.historical_cell(
                    "regular",
                    split,
                    prompt,
                    read_rows(TRANSFER_DATA / split / f"{prompt}.jsonl"),
                    baseline_identity,
                )
    from metrics import summarize as summarize_benchmark
    from metrics_honest_controls import summarize_controls

    from experiments.tool_trajectory_monitoring.benchmark_gpt_oss_ood import (
        summarize as summarize_id,
    )
    from gleipnir.calibration import binary_calibration

    report = {"identity": identity(), "preference": {}, "monitoring": {}, "id": {}}
    for name in ("original", "continued"):
        report["preference"][name] = summarize_preferences(
            read_rows(DEST / name / "preferences.jsonl")
        )
    for prompt in PROMPTS:
        path = DEST / "continued" / f"benchmark_{prompt}.jsonl"
        rows = read_rows(path)
        controls = read_rows(DEST / "continued" / f"honest_controls_{prompt}.jsonl")
        standard, _ = summarize_benchmark(path)
        report["monitoring"][prompt] = {
            "standard": standard,
            "honest_controls": summarize_controls(controls, rows),
        }
    rows = read_rows(DEST / "continued/id_neutral.jsonl")
    mapped = [
        {**r, "source": r["source_dataset"], "label": r["ground_truth"]} for r in rows
    ]
    report["id"]["continued"] = {
        **summarize_id(mapped),
        "calibration": binary_calibration(
            [r["label"] for r in mapped], [r["score"] for r in mapped]
        ),
    }
    # Freeze and explicitly attribute the existing original-adapter transfer scores.
    baseline_path = (
        ROOT / "results/monitoring_injection_removal/evaluation/4b/summary.json"
    )
    baseline = json.loads(baseline_path.read_text())
    report["original_transfer_baseline"] = {
        "sha256": file_hash(baseline_path),
        "benchmark": {p: baseline["benchmark"][f"regular/{p}"] for p in PROMPTS},
        "id": json.loads(
            (
                ROOT / "results/student_injection_awareness/4b/regular/id/result.json"
            ).read_text()
        ),
    }
    write_json(DEST / "summary.json", report)
    write_json(
        DEST / "status.json",
        {
            "stage": "complete",
            "preference_checkpoints": 2,
            "monitor_prompt_cells": 3,
            "id_cells": 1,
        },
    )
    print("evaluation_complete judge_injection_continuation", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage", choices=("reference", "vllm", "summarize"), required=True
    )
    args = parser.parse_args()
    try:
        {"reference": reference, "vllm": serving, "summarize": summarize}[args.stage]()
    except Exception as exc:
        write_json(
            DEST / f"failure_{args.stage}_{time.time_ns()}.json",
            {"type": type(exc).__name__, "message": str(exc)},
        )
        raise
