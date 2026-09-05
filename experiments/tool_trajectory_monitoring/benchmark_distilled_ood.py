#!/usr/bin/env python3
"""Evaluate frozen soft-distilled Qwen adapters on the strict OOD suite."""

from __future__ import annotations

import argparse
import hashlib
import math
import time
from pathlib import Path
from typing import Any

from experiments.tool_trajectory_monitoring.benchmark_gpt_oss_ood import (
    balanced_canary_rows,
    batches,
    ordered_predictions,
    prediction_row,
    summarize,
)
from experiments.tool_trajectory_monitoring.benchmark_qwen_ood import (
    QWEN_NON_THINKING_ASSISTANT_SUFFIX,
    render_margin_prompt,
)
from experiments.tool_trajectory_monitoring.prompting import load_prompt_set
from experiments.tool_trajectory_monitoring.qwen_reasoning_core import (
    atomic_write_json,
    atomic_write_jsonl,
    load_json,
    load_jsonl,
)
from gleipnir.binary_evaluation import binary_token_ids

DEFAULT_CONFIG = Path(
    "experiments/tool_trajectory_monitoring/distillation_ood_benchmark.json"
)
DEFAULT_OUTPUT = Path("results/tool_trajectory_distillation_ood")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_config(config: dict[str, Any]) -> None:
    if config.get("status") != "frozen_evaluation":
        raise ValueError("distillation OOD config must be frozen")
    prompt = config.get("prompt", {})
    if prompt.get("role") not in {"student", "teacher"}:
        raise ValueError(
            "distilled adapters must use a frozen student or teacher prompt"
        )
    if prompt.get("enable_thinking") is not False:
        raise ValueError("distilled OOD evaluation must disable native thinking")
    if prompt.get("assistant_suffix") != QWEN_NON_THINKING_ASSISTANT_SUFFIX:
        raise ValueError("Qwen non-thinking assistant boundary drifted")
    if prompt.get("decision_prefix") != "Prediction:":
        raise ValueError("decision prefix must remain Prediction:")
    if config.get("scoring", {}).get("tokens") != ["0", "1"]:
        raise ValueError("decision tokens must remain literal 0 and 1")
    engine = config.get("engine", {})
    if int(engine.get("max_lora_rank", 0)) != 128:
        raise ValueError("serving rank must remain 128")
    if engine.get("gdn_prefill_backend", "auto") not in {
        "auto",
        "flashinfer",
        "triton",
        "cutedsl",
    }:
        raise ValueError("unknown GDN prefill backend")
    groups = config.get("model_groups", {})
    expected_groups = set(config.get("expected_model_groups", ["9b", "4b"]))
    if not expected_groups or set(groups) != expected_groups:
        raise ValueError(
            "model groups differ from the frozen contract: "
            f"{sorted(groups)} != {sorted(expected_groups)}"
        )
    for model_size, group in groups.items():
        if group.get("parity_job") not in group.get("expected_jobs", []):
            raise ValueError(f"{model_size} parity job is not selected for evaluation")


def validate_inputs(config: dict[str, Any]) -> list[dict[str, Any]]:
    scope = config["scope"]
    path = Path(scope["input"])
    if sha256_file(path) != scope["input_sha256"]:
        raise ValueError("OOD input checksum differs from frozen config")
    manifest_path = Path(scope["manifest"])
    if scope.get("manifest_sha256") and sha256_file(manifest_path) != scope[
        "manifest_sha256"
    ]:
        raise ValueError("OOD input manifest checksum differs from frozen config")
    manifest = load_json(manifest_path)
    prompt_role = str(config["prompt"]["role"])
    if prompt_role == "student":
        if manifest.get("output_sha256") != scope["input_sha256"]:
            raise ValueError("student OOD manifest and config disagree")
        if manifest.get("source_sha256") != scope["source_input_sha256"]:
            raise ValueError("teacher OOD source checksum differs from frozen config")
        manifest_template_sha256 = manifest.get("student_template_sha256")
    else:
        manifest_template_sha256 = manifest.get("prompt", {}).get("template_sha256")
        if manifest.get("output", {}).get("sha256") != scope["input_sha256"]:
            raise ValueError("teacher OOD manifest and config disagree")
    if manifest_template_sha256 != config["prompt"]["template_sha256"]:
        raise ValueError(f"{prompt_role} prompt hash differs from frozen config")
    rows = load_jsonl(path)
    if len(rows) != int(scope["rows"]):
        raise ValueError("student OOD row count differs from frozen config")
    identities = [str(row.get("id")) for row in rows]
    if len(identities) != len(set(identities)):
        raise ValueError("student OOD input contains duplicate IDs")
    sources = {str(row.get("metadata", {}).get("source_dataset")) for row in rows}
    if sources != set(scope["sources"]):
        raise ValueError("student OOD source membership drifted")
    for row in rows:
        metadata = row.get("metadata", {})
        prompt = row.get("prompt")
        if int(metadata.get("ground_truth", -1)) not in {0, 1}:
            raise ValueError(f"non-binary OOD label for id={row.get('id')!r}")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError(f"empty {prompt_role} prompt for id={row.get('id')!r}")
        if (
            metadata.get("prompt_template_sha256")
            != config["prompt"]["template_sha256"]
        ):
            raise ValueError(f"student template drift for id={row.get('id')!r}")
        if hashlib.sha256(prompt.encode()).hexdigest() != metadata.get(
            "rendered_prompt_sha256"
        ):
            raise ValueError(f"student prompt checksum drift for id={row.get('id')!r}")
    return rows


def validate_jobs(
    config: dict[str, Any], model_size: str, only_job: str | list[str] | None = None
) -> list[dict[str, Any]]:
    group = config["model_groups"][model_size]
    jobs_path = Path(group["jobs"])
    if group.get("jobs_sha256") and sha256_file(jobs_path) != group["jobs_sha256"]:
        raise ValueError(f"{model_size} evaluation job manifest checksum drifted")
    available_jobs = load_jsonl(jobs_path)
    expected = list(group["expected_jobs"])
    manifest_expected = list(group.get("jobs_manifest_expected_jobs", expected))
    available_names = [str(job.get("job_name")) for job in available_jobs]
    if available_names != manifest_expected:
        raise ValueError(f"{model_size} evaluation job manifest drifted")
    by_name = {str(job["job_name"]): job for job in available_jobs}
    if len(by_name) != len(available_jobs) or not set(expected).issubset(by_name):
        raise ValueError(f"{model_size} selected evaluation jobs are incomplete")
    jobs = [by_name[name] for name in expected]
    expected_target = group.get("expected_target", "kimi_soft")
    if expected_target not in {"kimi_soft", "kimi_soft_plus_auxiliary"}:
        raise ValueError(f"unsupported frozen training target: {expected_target!r}")
    for job in jobs:
        if (
            job.get("target") != expected_target
            or float(job.get("soft_loss_weight", -1)) != 1.0
            or float(job.get("direct_loss_weight", -1)) != 0.0
            or int(job.get("rank", -1)) != 128
            or int(job.get("seed", -1)) != 0
        ):
            raise ValueError(f"soft-only recipe drift for {job.get('job_name')!r}")
    if only_job is not None:
        selected = [only_job] if isinstance(only_job, str) else only_job
        if not selected or len(selected) != len(set(selected)):
            raise ValueError("evaluation job selection must be nonempty and unique")
        if not set(selected).issubset(expected):
            raise ValueError(f"unknown {model_size} evaluation job {only_job!r}")
        jobs = [job for job in jobs if job["job_name"] in selected]
    return jobs


def adapter_metadata(job: dict[str, Any]) -> dict[str, Any]:
    model_dir = Path(job["model_dir"])
    causal_dir = Path(job["causal_adapter_dir"])
    manifest_path = model_dir / "rebase_manifest.json"
    metadata_path = causal_dir / "training_metadata.json"
    if not manifest_path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError(f"incomplete adapter artifacts for {job['job_name']}")
    manifest = load_json(manifest_path)
    source = causal_dir / "adapter_model.safetensors"
    destination = model_dir / "adapter_model.safetensors"
    if sha256_file(source) != manifest.get("source_sha256"):
        raise ValueError(f"causal adapter checksum drift for {job['job_name']}")
    if sha256_file(destination) != manifest.get("destination_sha256"):
        raise ValueError(f"serving adapter checksum drift for {job['job_name']}")
    return {
        "rebase_manifest": manifest,
        "training_metadata": load_json(metadata_path),
        "training_metadata_sha256": sha256_file(metadata_path),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--model-size", required=True)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--only-job",
        action="append",
        help="Evaluate only this job; repeat to batch completed jobs in one engine.",
    )
    parser.add_argument("--include-base", action="store_true")
    parser.add_argument("--canary-only", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_json(args.config)
    validate_config(config)
    if args.model_size not in config["model_groups"]:
        raise ValueError(f"unknown model group {args.model_size!r}")
    rows = validate_inputs(config)
    jobs = validate_jobs(config, args.model_size, args.only_job)
    group = config["model_groups"][args.model_size]
    prompt_config = config["prompt"]
    engine = config["engine"]
    prompt_set = load_prompt_set()
    prompt_role = str(prompt_config["role"])
    prompt_template = getattr(prompt_set, prompt_role)
    if prompt_template.template_sha256 != prompt_config["template_sha256"]:
        raise ValueError(
            f"working-tree {prompt_role} prompt differs from frozen config"
        )
    if args.canary_only:
        rows = balanced_canary_rows(
            rows,
            list(config["scope"]["sources"]),
            rows_per_source_label=int(engine["canary_rows_per_source_label"]),
        )

    audited = {job["job_name"]: adapter_metadata(job) for job in jobs}
    config_sha256 = sha256_file(args.config)

    import torch
    import vllm
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    tokenizer = AutoTokenizer.from_pretrained(group["id"], revision=group["revision"])
    token_ids = binary_token_ids(tokenizer)
    sampling = SamplingParams(
        max_tokens=1,
        temperature=0.0,
        logprobs=len(token_ids),
        logprob_token_ids=token_ids,
        allowed_token_ids=token_ids,
    )
    llm_kwargs = {
        "model": group["id"],
        "tokenizer": group["id"],
        "revision": group["revision"],
        "tokenizer_revision": group["revision"],
        "dtype": group["dtype"],
        "tensor_parallel_size": int(engine["tensor_parallel_size"]),
        "gpu_memory_utilization": float(engine["gpu_memory_utilization"]),
        "max_model_len": int(engine["max_model_len"]),
        "max_num_seqs": int(engine["max_num_seqs"]),
        "max_num_batched_tokens": int(engine["max_num_batched_tokens"]),
        "enable_prefix_caching": bool(engine["enable_prefix_caching"]),
        "language_model_only": bool(engine["language_model_only"]),
        "enable_lora": True,
        "max_lora_rank": int(engine["max_lora_rank"]),
        "max_loras": int(engine["max_loras"]),
        "seed": int(engine["seed"]),
    }
    if engine.get("gdn_prefill_backend") not in {None, "auto"}:
        llm_kwargs["gdn_prefill_backend"] = str(engine["gdn_prefill_backend"])
    llm = LLM(
        **llm_kwargs,
    )
    prompts = [
        render_margin_prompt(
            tokenizer,
            str(row["prompt"]),
            enable_thinking=False,
            assistant_suffix=str(prompt_config["assistant_suffix"]),
            decision_prefix=str(prompt_config["decision_prefix"]),
        )
        for row in rows
    ]
    output_root = args.output_root / args.model_size

    def evaluate(
        name: str, request: Any | None, metadata: dict[str, Any]
    ) -> dict[str, Any]:
        output_dir = output_root / ("base" if request is None else "adapters") / name
        predictions_path = output_dir / "predictions.jsonl"
        existing = (
            []
            if args.force or not predictions_path.is_file()
            else load_jsonl(predictions_path)
        )
        completed = {}
        allowed = {str(row["id"]): row for row in rows}
        for prediction in existing:
            identity = str(prediction.get("id"))
            if identity not in allowed or identity in completed:
                raise ValueError(f"invalid resumable prediction id={identity!r}")
            if prediction.get("config_sha256") != config_sha256:
                raise ValueError(f"prediction config drift for id={identity!r}")
            completed[identity] = prediction
        started = time.time()
        pending = [row for row in rows if str(row["id"]) not in completed]
        prompt_by_id = dict(zip((str(row["id"]) for row in rows), prompts, strict=True))
        for batch_index, batch in enumerate(
            batches(pending, int(engine["batch_rows"])), start=1
        ):
            batch_prompts = [prompt_by_id[str(row["id"])] for row in batch]
            outputs = llm.generate(batch_prompts, sampling, lora_request=request)
            if len(outputs) != len(batch):
                raise RuntimeError("vLLM output count differs from request count")
            for record, margin_prompt, generated in zip(
                batch, batch_prompts, outputs, strict=True
            ):
                packed = prediction_row(
                    record,
                    margin_prompt,
                    generated,
                    token_ids,
                    config_sha256,
                    int(engine["max_model_len"]),
                )
                completed[packed["id"]] = packed
            atomic_write_jsonl(predictions_path, ordered_predictions(rows, completed))
            print(
                f"{name}: batch={batch_index} complete={len(completed)}/{len(rows)}",
                flush=True,
            )
        predictions = ordered_predictions(rows, completed)
        if len(predictions) != len(rows) or not all(
            math.isfinite(float(row["score"])) for row in predictions
        ):
            raise RuntimeError(f"incomplete or non-finite evaluation for {name}")
        result = {
            "campaign_id": config["campaign_id"],
            "config_sha256": config_sha256,
            "model_size": args.model_size,
            "model": {key: group[key] for key in ("id", "revision", "dtype")},
            "prompt": prompt_config,
            "engine": engine,
            "job_name": name,
            **metadata,
            "runtime": {
                "vllm_version": vllm.__version__,
                "torch_version": torch.__version__,
                "gpu_name": torch.cuda.get_device_name(0),
                "elapsed_seconds_this_invocation": time.time() - started,
            },
            "score": "normalized direct probability for literal 1 versus 0",
            **summarize(predictions),
        }
        atomic_write_json(output_dir / "result.json", result)
        macro = result["metrics"]["macro"]["macro"]
        print(
            f"{name}: rows={len(rows)} auroc={macro['auroc']:.6f} "
            f"pauroc_at_20={macro['pauroc_at_20']:.6f}",
            flush=True,
        )
        return result

    if args.include_base:
        evaluate("base", None, {"adapter_path": None})
    for lora_id, job in enumerate(jobs, start=1):
        name = str(job["job_name"])
        details = audited[name]
        manifest = details["rebase_manifest"]
        request = LoRARequest(name, lora_id, str(job["model_dir"]))
        evaluate(
            name,
            request,
            {
                "adapter_path": str(job["model_dir"]),
                "training_job": job,
                "training_metadata": details["training_metadata"],
                "training_metadata_sha256": details["training_metadata_sha256"],
                "adapter_rebase_source_sha256": manifest["source_sha256"],
                "adapter_rebase_destination_sha256": manifest["destination_sha256"],
            },
        )


if __name__ == "__main__":
    main()
