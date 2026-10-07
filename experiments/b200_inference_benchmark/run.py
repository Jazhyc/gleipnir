"""Prepare and benchmark one persistent local HTTP monitor serving process."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib.metadata
import json
import os
import shutil
import statistics
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import yaml

from gleipnir.inference_benchmark import (
    match_selection,
    measurement_summary,
    paired_score_summary,
    quick_workload,
    ranking_comparison,
    response_score,
)
from gleipnir.merged_lora import file_sha256

ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT = Path(__file__).parent
DATA = ROOT / "data/b200_inference_benchmark"
OUTPUT = ROOT / "results/b200_inference_benchmark"
LOGS = ROOT / "logs/runpod/b200_inference_benchmark"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def write(path: Path, value: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def prepare(config: dict) -> dict:
    from transformers import AutoTokenizer

    from experiments.tool_trajectory_monitoring.benchmark_qwen_ood import (
        QWEN_NON_THINKING_ASSISTANT_SUFFIX,
        render_margin_prompt,
    )
    from gleipnir.binary_evaluation import binary_token_ids

    hashes = {}
    for name in ("source", "selection", "canary_source", "parity"):
        path = ROOT / config[name]
        hashes[name] = sha(path)
        if hashes[name] != config[f"{name}_sha256"]:
            raise ValueError(f"input drift: {name}")
    if (
        sha(ROOT / config["adapter"] / "adapter_model.safetensors")
        != config["adapter_sha256"]
    ):
        raise ValueError("adapter checksum drift")
    tokenizer = AutoTokenizer.from_pretrained(
        config["model"], revision=config["revision"]
    )

    def render(row: dict) -> dict:
        prompt = render_margin_prompt(
            tokenizer,
            row["student_prompt"],
            enable_thinking=False,
            assistant_suffix=QWEN_NON_THINKING_ASSISTANT_SUFFIX,
            decision_prefix="Prediction:",
        )
        tokens = len(tokenizer.encode(prompt, add_special_tokens=False))
        if tokens + 1 > config["max_model_len"]:
            raise ValueError("prompt exceeds zero-truncation envelope")
        return {
            "id": f"{row['dataset']}:{row['index']}",
            "index": row["index"],
            "dataset": row["dataset"],
            "label": row["label"],
            "prompt": prompt,
            "prompt_tokens": tokens,
            "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        }

    parent = [
        render(row)
        for row in match_selection(
            read_rows(ROOT / config["source"]), read_rows(ROOT / config["selection"])
        )
    ]
    if len(parent) != 320:
        raise ValueError("training cohort must contain exactly 320 rows")
    parity = json.loads((ROOT / config["parity"]).read_text())
    if not parity["passed"]:
        raise ValueError("historical serving parity failed")
    canary_map = {
        row["index"]: row for row in read_rows(ROOT / config["canary_source"])
    }
    canaries = [render(canary_map[index]) for index in parity["reference"]["ids"]]
    if [row["prompt_sha256"] for row in canaries] != parity["reference"][
        "prompt_sha256"
    ]:
        raise ValueError("serving canary prompt drift")
    quick = quick_workload(parent, config["quick_rows"], config["seed"])
    full = quick_workload(parent, len(parent), config["seed"])
    for name, rows in [("quick", quick), ("full", full), ("canary", canaries)]:
        write(DATA / f"{name}.json", rows)
    manifest = {
        "inputs": hashes,
        "token_ids": binary_token_ids(tokenizer),
        "config_sha256": sha(EXPERIMENT / "config.yaml"),
        "files": {
            name: sha(DATA / f"{name}.json") for name in ("quick", "full", "canary")
        },
        "populations": {
            name: {
                "rows": len(rows),
                "prompt_tokens": {
                    "total": sum(r["prompt_tokens"] for r in rows),
                    "min": min(r["prompt_tokens"] for r in rows),
                    "max": max(r["prompt_tokens"] for r in rows),
                },
                "datasets": dict(Counter(r["dataset"] for r in rows)),
            }
            for name, rows in [("quick", quick), ("full", full)]
        },
    }
    write(DATA / "manifest.json", manifest)
    return manifest


def server_command(config: dict, merged_model: Path | None = None) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        config["model"],
        "--revision",
        config["revision"],
        "--host",
        "127.0.0.1",
        "--port",
        str(config["port"]),
        "--dtype",
        "bfloat16",
        "--language-model-only",
        "--enable-lora",
        "--max-lora-rank",
        "128",
        "--max-loras",
        "1",
        "--lora-modules",
        f"monitor={ROOT / config['adapter']}",
        "--served-model-name",
        "base",
        "--gdn-prefill-backend",
        config.get("gdn_prefill_backend", "flashinfer"),
        "--max-model-len",
        str(config["max_model_len"]),
        "--max-num-seqs",
        str(config["max_num_seqs"]),
        "--max-num-batched-tokens",
        str(config["max_num_batched_tokens"]),
        "--gpu-memory-utilization",
        str(config["gpu_memory_utilization"]),
        "--no-enable-prefix-caching",
        "--enable-chunked-prefill",
        "--logprobs-mode",
        "processed_logprobs",
        "--seed",
        str(config["seed"]),
    ]
    if merged_model is not None:
        command[command.index("--model") + 1] = str(merged_model)
        for flag in ("--revision", "--max-lora-rank", "--max-loras", "--lora-modules"):
            index = command.index(flag)
            del command[index : index + 2]
        command.remove("--enable-lora")
        command[command.index("--served-model-name") + 1] = "monitor"
    return command


def verify_merged_model(config: dict, path: Path) -> dict:
    """Bind disposable merged weights to the frozen source and file manifest."""
    receipt = json.loads((path / "merge_manifest.json").read_text())
    for key, expected in (
        ("model", config["model"]),
        ("revision", config["revision"]),
        ("adapter_sha256", config["adapter_sha256"]),
    ):
        if receipt[key] != expected:
            raise ValueError(f"merged artifact source drift: {key}")
    for name, checksum in receipt["files_sha256"].items():
        if Path(name).name != name or file_sha256(path / name) != checksum:
            raise ValueError(f"merged artifact file drift: {name}")
    return receipt


def prepared_manifest(config: dict) -> dict:
    """Reuse frozen rendered inputs without rereading the raw source population."""
    manifest = json.loads((DATA / "manifest.json").read_text())
    if manifest["config_sha256"] != sha(EXPERIMENT / "config.yaml"):
        raise ValueError("prepared workload configuration drift")
    for name, checksum in manifest["inputs"].items():
        if checksum != config[f"{name}_sha256"]:
            raise ValueError(f"prepared source binding drift: {name}")
    for name, checksum in manifest["files"].items():
        if sha(DATA / f"{name}.json") != checksum:
            raise ValueError(f"prepared prompt drift: {name}")
    return manifest


def environment() -> dict[str, str]:
    env = dict(os.environ)
    env.update(
        PYTHONPATH=f"{ROOT / 'src'}:{ROOT}",
        PATH=f"{ROOT / '.venv/bin'}:/usr/local/cuda/bin:{env.get('PATH', '')}",
        CUDA_HOME="/usr/local/cuda",
        PYTHONUNBUFFERED="1",
        HF_HOME=str(ROOT / ".cache/huggingface"),
        HF_HUB_CACHE=str(ROOT / ".cache/huggingface/hub"),
        FLASHINFER_WORKSPACE_BASE=str(ROOT),
        TOKENIZERS_PARALLELISM="false",
        OMP_NUM_THREADS="4",
        VLLM_CACHE_ROOT=str(ROOT / ".cache/vllm/student_injection_awareness_v1"),
        TORCHINDUCTOR_CACHE_DIR=str(
            ROOT / ".cache/torchinductor/student_injection_awareness_v1"
        ),
    )
    for name in ("LD_LIBRARY_PATH", "TRITON_CACHE_DIR", "TILELANG_CACHE_DIR"):
        env.pop(name, None)
    return env


async def trial(
    client: Any,
    rows: list[dict],
    ids: list[int],
    concurrency: int,
    model: str = "monitor",
) -> tuple[list[dict], float]:
    queue = asyncio.Queue()
    for index, row in enumerate(rows):
        queue.put_nowait((index, row))
    results = [None] * len(rows)

    async def worker():
        while not queue.empty():
            index, row = queue.get_nowait()
            start = time.perf_counter()
            response = await client.post(
                "/v1/completions",
                json={
                    "model": model,
                    "prompt": row["prompt"],
                    "max_tokens": 1,
                    "temperature": 0.0,
                    "logprobs": 2,
                    "allowed_token_ids": ids,
                    "add_special_tokens": False,
                    "return_tokens_as_token_ids": True,
                },
            )
            response.raise_for_status()
            elapsed = time.perf_counter() - start
            results[index] = {
                "id": row["id"],
                "prompt_sha256": row["prompt_sha256"],
                "prompt_tokens": row["prompt_tokens"],
                "latency_seconds": elapsed,
                **response_score(response.json(), ids, row["prompt_tokens"]),
            }

    start = time.perf_counter()
    await asyncio.gather(*(worker() for _ in range(min(concurrency, len(rows)))))
    return results, time.perf_counter() - start


def resolve_kernel_baseline(condition: dict | None) -> dict | None:
    """Use the selected archived control for new kernel comparisons by default."""
    if condition is None or condition.get("baseline", "selected") != "selected":
        return condition
    selection_path = EXPERIMENT / "baseline.json"
    selection = json.loads(selection_path.read_text())
    summary_path = ROOT / selection["results"] / "summary.json"
    summary = json.loads(summary_path.read_text())
    if selection.get("quality_acceptance"):
        acceptance_path = ROOT / selection["quality_acceptance"]
        if (
            sha(acceptance_path) != selection["quality_acceptance_sha256"]
            or json.loads(acceptance_path.read_text())["status"]
            != "user_accepted_finite"
        ):
            raise ValueError("selected inference baseline identity drift: acceptance")
    if (
        sha(summary_path) != selection["summary_sha256"]
        or summary["status"] != "complete"
        or summary["manifest_sha256"] != selection["manifest_sha256"]
        or sha(DATA / "manifest.json") != selection["manifest_sha256"]
        or sha(ROOT / selection["validated_recipe_config"])
        != selection["validated_recipe_config_sha256"]
    ):
        raise ValueError("selected inference baseline identity drift")
    return {
        **condition,
        "baseline": selection["results"],
        "baseline_selection_sha256": sha(selection_path),
        "baseline_summary_sha256": selection["summary_sha256"],
        **(
            {
                "baseline_quality_acceptance_sha256": selection[
                    "quality_acceptance_sha256"
                ]
            }
            if selection.get("quality_acceptance")
            else {}
        ),
    }


def parity_status(
    comparisons: dict, limits: dict, effect: float, *, baseline_accepted: bool
) -> tuple[bool, bool]:
    """Retain strict master parity while gating changes against an accepted baseline."""

    def agrees(value: dict) -> bool:
        return (
            value["mean_absolute_difference"] <= limits["max_mean_absolute_difference"]
            and value["correlation"] >= limits["min_correlation"]
        )

    effect_passed = effect >= limits["min_adapter_effect"]
    strict = effect_passed and all(agrees(v) for v in comparisons.values())
    relative = (
        baseline_accepted
        and effect_passed
        and "kernel_baseline" in comparisons
        and agrees(comparisons["kernel_baseline"])
    )
    return bool(strict), bool(strict or relative)


def validate_preparation_usage(audit: dict, worker_pid: int, mode: str) -> None:
    """Reject stale, failed or incomplete preparation receipts before timing."""
    expected = (
        {
            "vendor": 64
            if audit.get("condition", {}).get("attention_projection_precision") == "fp4"
            else 48,
            "silu": 32,
            "norm": 32,
        }
        if mode == "combined"
        else {mode: 112 if mode == "vendor" else 32}
    )
    if (
        not audit.get("passed")
        or audit.get("worker_pid") != worker_pid
        or audit.get("condition", {}).get("fp4_preparation") != mode
        or Counter(c["stage"] for c in audit.get("calls", [])) != expected
    ):
        raise ValueError("native preparation usage audit failed")


def compatible_server_command(observed: list[str], requested: list[str]) -> bool:
    """Allow client-only reference changes while enforcing loaded kernel identity."""

    def identity(command: list[str]) -> list[str]:
        command = list(command)
        if "--additional-config" in command:
            index = command.index("--additional-config") + 1
            config = json.loads(command[index])
            condition = config.get("serving_condition")
            if condition is not None:
                condition.pop("high_reference", None)
            command[index] = json.dumps(config, sort_keys=True)
        return command

    try:
        return identity(observed) == identity(requested)
    except (ValueError, IndexError, AttributeError, TypeError):
        return False


async def benchmark(
    config: dict,
    manifest: dict,
    output: Path,
    count: int,
    reuse: bool,
    merged_model: Path | None = None,
    kernel_condition: dict | None = None,
    *,
    startup_only: bool = False,
    frontend: dict | None = None,
    host_wrapper: dict | None = None,
) -> None:
    import httpx
    import numpy as np

    kernel_condition = resolve_kernel_baseline(kernel_condition)
    command = server_command(config, merged_model)
    if kernel_condition:
        command.extend(kernel_condition["extra_server_args"])
        command[command.index("-m") + 1] = kernel_condition.get(
            "server_module", "vllm.entrypoints.openai.api_server"
        )
    metadata = (output.parent if kernel_condition else OUTPUT) / "server.json"
    logs = (
        ROOT
        / kernel_condition.get("log_directory", "logs/runpod/b200_inference_kernels")
        if kernel_condition
        else LOGS
    )
    base_url = f"http://127.0.0.1:{config['port']}"
    started = time.perf_counter()
    server_environment = environment()
    if kernel_condition and kernel_condition.get("training_fp4_environment"):
        from gleipnir.native_fp4_training import native_fp4_environment
        from gleipnir.qwen35_fast_training import (
            DEFAULT_TRITON_TARGET,
            triton_environment,
        )

        server_environment = native_fp4_environment(
            triton_environment(DEFAULT_TRITON_TARGET, server_environment), ROOT
        )
        server_environment["PYTHONPATH"] += f":{ROOT / '.cache/kernels/fa4'}"
    if kernel_condition and kernel_condition.get("prefill_graphs"):
        server_environment["VLLM_SERVER_DEV_MODE"] = "1"
    from gleipnir.serving_runtime import local_serving_runtime

    runtime = local_serving_runtime(ROOT, server_environment)
    if runtime is not None:
        command[0] = runtime["python"]
    frontend_receipt = None
    if frontend is not None:
        from gleipnir.serving_gigatoken import configure_frontend

        frontend_receipt = configure_frontend(
            ROOT, frontend, command, server_environment
        )
    if host_wrapper is not None:
        if frontend is None:
            raise ValueError("host wrapper trial requires the native registry frontend")
        validation_path = ROOT / host_wrapper["validation"]
        validation = json.loads(validation_path.read_text())
        helper = ROOT / "src/gleipnir/serving_frost_wrappers.py"
        if not validation.get("passed") or validation["helper_sha256"] != sha(helper):
            raise ValueError("FROST host wrapper admission drift")
        host_wrapper = {
            **host_wrapper,
            "validation_sha256": sha(validation_path),
            "source_sha256": sha(helper),
            "entrypoint_sha256": frontend_receipt["entrypoint_sha256"],
        }
        server_environment["GLEIPNIR_FROST_WRAPPER_VALIDATION"] = host_wrapper[
            "validation"
        ]
        server_environment["VLLM_SERVER_DEV_MODE"] = "1"
    from gleipnir.serving_cache_mirror import compiler_mirror, persist_compiler_mirror

    mirror = compiler_mirror(ROOT, server_environment) if runtime is not None else None
    if reuse:
        receipt = json.loads(metadata.read_text())
        if (
            not compatible_server_command(receipt["command"], command)
            or receipt["config_sha256"] != manifest["config_sha256"]
            or receipt.get("frontend") != frontend_receipt
            or receipt.get("host_wrapper") != host_wrapper
        ):
            raise ValueError("resident server configuration drift")
        os.kill(receipt["pid"], 0)
    else:
        if metadata.exists():
            raise ValueError(
                "preserve existing server; use --reuse-server if compatible"
            )
        async with httpx.AsyncClient(base_url=base_url, trust_env=False) as probe:
            try:
                await probe.get("/health")
            except httpx.ConnectError:
                pass
            else:
                raise ValueError("port already in use")
        logs.mkdir(parents=True, exist_ok=True)
        if shutil.which("ninja", path=server_environment["PATH"]) is None:
            raise RuntimeError("serving compiler executable ninja is unavailable")
        with (logs / "server.log").open("x") as handle:
            process = subprocess.Popen(
                command,
                cwd=ROOT,
                env=server_environment,
                stdout=handle,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        receipt = {
            "pid": process.pid,
            "command": command,
            "config_sha256": manifest["config_sha256"],
            "started_at_unix": time.time(),
            "local_runtime": runtime,
            "compiler_mirror": mirror,
            "frontend": frontend_receipt,
            "host_wrapper": host_wrapper,
            "status": "starting",
            "cache_paths": {
                **{k: v for k, v in server_environment.items() if "CACHE" in k},
                "FLASHINFER_CACHE_DIR": str(
                    Path(server_environment["FLASHINFER_WORKSPACE_BASE"])
                    / ".cache/flashinfer"
                ),
            },
            "log": str(logs / "server.log"),
        }
        write(metadata, receipt)
    async with httpx.AsyncClient(
        base_url=base_url,
        timeout=180,
        trust_env=False,
        limits=httpx.Limits(max_connections=32),
    ) as client:
        deadline = time.perf_counter() + 1200
        while True:
            if not reuse and process.poll() is not None:
                raise RuntimeError(f"serving process exited: {process.returncode}")
            try:
                health = await client.get("/health")
                if health.status_code == 200:
                    break
            except httpx.HTTPError:
                pass
            if time.perf_counter() > deadline:
                raise TimeoutError("server startup exceeded twenty minutes")
            await asyncio.sleep(2)
        receipt.update(status="ready", ready_at_unix=time.time())
        receipt["compiler_cache_persistence"] = persist_compiler_mirror(mirror)
        write(metadata, receipt)
        report = {
            "status": "running",
            "serving_mode": (
                kernel_condition["name"]
                if kernel_condition
                else "merged_bf16"
                if merged_model
                else "dynamic_lora_bf16"
            ),
            "kernel_condition": kernel_condition,
            "rows": count,
            "manifest_sha256": sha(DATA / "manifest.json"),
            "server": receipt,
            "ready_wait_seconds": time.perf_counter() - started,
            "models": (await client.get("/v1/models")).json(),
            "trials": [],
            "runtime": {
                name: importlib.metadata.version(name)
                for name in ("torch", "vllm", "transformers", "httpx")
            },
            "source_sha256": {
                str(p.relative_to(ROOT)): sha(p)
                for p in [Path(__file__), ROOT / "src/gleipnir/inference_benchmark.py"]
            },
        }
        write(output / "summary.json", report)
        print("server_ready", flush=True)
        if kernel_condition and kernel_condition.get("prefill_graphs"):
            from experiments.b200_attention_gdn_serving.prefill_graph_canary import (
                canary,
            )

            await canary(client, manifest["token_ids"], output)
            report["prefill_graph_canary_sha256"] = sha(
                output / "prefill_graph_canary.json"
            )
        canaries = json.loads((DATA / "canary.json").read_text())
        parity = json.loads((ROOT / config["parity"]).read_text())
        parity_start = time.perf_counter()
        comparisons, served = {}, {}
        for mode in ("adapter",) if merged_model else ("base", "adapter"):
            values, _ = await trial(
                client,
                canaries,
                manifest["token_ids"],
                4,
                "base" if mode == "base" else "monitor",
            )
            served[mode] = [r["score"] for r in values]
            reference = parity["reference"][mode]
            comparisons[mode] = {
                "mean_absolute_difference": float(
                    np.mean(np.abs(np.array(served[mode]) - reference))
                ),
                "correlation": float(np.corrcoef(served[mode], reference)[0, 1]),
            }
        if merged_model:
            baseline_parity = OUTPUT / "baseline01/http_parity.json"
            baseline = json.loads(baseline_parity.read_text())
            if not baseline["passed"]:
                raise ValueError("unmerged baseline parity failed")
            served["base"] = baseline["served"]["base"]
            reference = baseline["served"]["adapter"]
            comparisons["dynamic_lora"] = {
                "mean_absolute_difference": float(
                    np.mean(np.abs(np.array(served["adapter"]) - reference))
                ),
                "correlation": float(np.corrcoef(served["adapter"], reference)[0, 1]),
            }
            report["unmerged_baseline_parity_sha256"] = sha(baseline_parity)
            if kernel_condition:
                prior_path = ROOT / kernel_condition["baseline"] / "http_parity.json"
                prior = json.loads(prior_path.read_text())
                if not prior["passed"] and not kernel_condition.get(
                    "baseline_quality_acceptance_sha256"
                ):
                    raise ValueError("merged control parity failed")
                reference = prior["served"]["adapter"]
                comparisons["kernel_baseline"] = {
                    "mean_absolute_difference": float(
                        np.mean(np.abs(np.array(served["adapter"]) - reference))
                    ),
                    "correlation": float(
                        np.corrcoef(served["adapter"], reference)[0, 1]
                    ),
                }
                report["merged_baseline_parity_sha256"] = sha(prior_path)
        effect = float(np.max(np.abs(np.array(served["adapter"]) - served["base"])))
        strict_passed, accepted_passed = parity_status(
            comparisons,
            parity["limits"],
            effect,
            baseline_accepted=bool(
                kernel_condition
                and kernel_condition.get("baseline_quality_acceptance_sha256")
            ),
        )
        write(
            output / "http_parity.json",
            {
                "passed": strict_passed,
                "evaluation_passed": accepted_passed,
                "baseline_relative_acceptance": accepted_passed and not strict_passed,
                "comparisons": comparisons,
                "adapter_effect": effect,
                "served": served,
                "reference_sha256": sha(ROOT / config["parity"]),
                "base_scores_reused": bool(merged_model),
                "seconds": time.perf_counter() - parity_start,
            },
        )
        report["numerical_parity_passed"] = strict_passed
        report["baseline_relative_parity_passed"] = accepted_passed
        report["diagnostic_only"] = not accepted_passed
        if not accepted_passed and not (
            kernel_condition and kernel_condition.get("allow_finite_parity_diagnostic")
        ):
            raise ValueError("HTTP scoring parity failed")
        if not accepted_passed:
            print(
                "finite_parity_failure_preserved diagnostic_benchmark=true", flush=True
            )
        rows = json.loads(
            (
                DATA / ("quick.json" if count == config["quick_rows"] else "full.json")
            ).read_text()
        )
        warmup = quick_workload(rows, 4, config["seed"])
        _, report["warmup_seconds"] = await trial(
            client, warmup, manifest["token_ids"], 16
        )
        if kernel_condition and kernel_condition.get("startup_audit"):
            path = ROOT / kernel_condition["startup_audit"]
            if not json.loads(path.read_text())["passed"]:
                raise ValueError("native serving startup audit failed")
            report["startup_audit_sha256"] = sha(path)
        if kernel_condition and kernel_condition.get("fp4_preparation"):
            path = metadata.parent / "native_preparation.json"
            precision = json.loads(
                (metadata.parent / "loaded_precision.json").read_text()
            )
            validate_preparation_usage(
                json.loads(path.read_text()),
                precision["worker_pid"],
                kernel_condition["fp4_preparation"],
            )
            report["preparation_audit_sha256"] = sha(path)
        if kernel_condition and kernel_condition.get("gemm_tuning_validation"):
            from gleipnir.serving_fp4_tuning_validation import validate_runtime_usage

            path = metadata.parent / "native_gemm_tuning.json"
            validate_runtime_usage(
                json.loads(path.read_text()),
                precision["worker_pid"],
                kernel_condition["gemm_tuning_validation"],
            )
            report["gemm_tuning_audit_sha256"] = sha(path)
        if (
            kernel_condition
            and kernel_condition.get("attention_projection_precision") == "fp4"
        ):
            from gleipnir.serving_attention_fp4 import validate_usage

            path = metadata.parent / "native_attention_projections.json"
            validate_usage(
                json.loads(path.read_text()),
                precision["worker_pid"],
                kernel_condition["attention_projection_validation"],
            )
            report["attention_projection_audit_sha256"] = sha(path)
        if kernel_condition and kernel_condition.get("swiglu_fusion_validation"):
            from gleipnir.serving_fp4_swiglu_validation import validate_runtime

            path = metadata.parent / "native_swiglu.json"
            validate_runtime(
                json.loads(path.read_text()),
                precision["worker_pid"],
                kernel_condition["swiglu_fusion_validation"],
            )
            report["swiglu_fusion_audit_sha256"] = sha(path)
        if kernel_condition and kernel_condition.get("swiglu_overhead_validation"):
            from gleipnir.serving_fp4_swiglu_overhead_validation import validate_runtime

            path = metadata.parent / "native_swiglu_overhead.json"
            validate_runtime(
                json.loads(path.read_text()),
                precision["worker_pid"],
                kernel_condition["swiglu_overhead_validation"],
            )
            report["swiglu_overhead_audit_sha256"] = sha(path)
        if kernel_condition and kernel_condition.get("swiglu_native_output_validation"):
            from gleipnir.serving_fp4_swiglu_native_output_validation import (
                validate_runtime,
            )

            path = metadata.parent / "native_swiglu_output.json"
            validate_runtime(
                json.loads(path.read_text()),
                precision["worker_pid"],
                kernel_condition["swiglu_native_output_validation"],
            )
            report["swiglu_native_output_audit_sha256"] = sha(path)
        if kernel_condition and kernel_condition.get("gdn_direct_output_validation"):
            from gleipnir.serving_gdn_direct_output import validate_runtime

            path = metadata.parent / "native_gdn_direct_output.json"
            validate_runtime(
                json.loads(path.read_text()),
                precision["worker_pid"],
                kernel_condition["gdn_direct_output_validation"],
            )
            report["gdn_direct_output_audit_sha256"] = sha(path)
        print(
            f"http_canary_complete strict={strict_passed} accepted={accepted_passed}",
            flush=True,
        )
        if startup_only:
            report.update(
                status="complete", startup_only=True, timing_sweep_skipped=True
            )
            write(output / "summary.json", report)
            return
        for concurrency in config["concurrency"]:
            for repeat in range(config["repeats"]):
                results, elapsed = await trial(
                    client, rows, manifest["token_ids"], concurrency
                )
                write(output / f"c{concurrency}_repeat{repeat}.json", results)
                summary = {
                    "concurrency": concurrency,
                    "repeat": repeat,
                    **measurement_summary(results, elapsed),
                }
                report["trials"].append(summary)
                write(output / "summary.json", report)
                print(json.dumps(summary), flush=True)
        arrays = [
            np.array(
                [
                    r["score"]
                    for r in json.loads(
                        (
                            output / f"c{t['concurrency']}_repeat{t['repeat']}.json"
                        ).read_text()
                    )
                ]
            )
            for t in report["trials"]
        ]
        ranges = np.ptp(np.stack(arrays), axis=0)
        report.update(
            status="complete",
            score_variation={
                "mean_range": float(ranges.mean()),
                "max_range": float(ranges.max()),
                "threshold_unstable_rows": int(
                    np.sum(
                        np.any(np.stack(arrays) >= 0.5, axis=0)
                        & np.any(np.stack(arrays) < 0.5, axis=0)
                    )
                ),
            },
            gpu_health=subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=memory.used,utilization.gpu,temperature.gpu",
                    "--format=csv,noheader",
                ]
            )
            .decode()
            .strip(),
        )
        if merged_model:
            baseline_path = (
                ROOT / kernel_condition["baseline"]
                if kernel_condition
                else OUTPUT / "baseline01"
            )
            baseline_report = json.loads((baseline_path / "summary.json").read_text())
            if (
                baseline_report["status"] != "complete"
                or baseline_report["manifest_sha256"] != report["manifest_sha256"]
            ):
                raise ValueError("baseline workload binding drift")
            comparison = {}
            for concurrency in config["concurrency"]:
                runs = [
                    [t for t in source["trials"] if t["concurrency"] == concurrency]
                    for source in (baseline_report, report)
                ]
                scores = [
                    [
                        json.loads(
                            (
                                path / f"c{concurrency}_repeat{t['repeat']}.json"
                            ).read_text()
                        )
                        for t in trials
                    ]
                    for path, trials in zip((baseline_path, output), runs, strict=True)
                ]
                throughputs = [
                    statistics.median(t["prompt_tokens_per_second"] for t in trials)
                    for trials in runs
                ]
                comparison[str(concurrency)] = {
                    "baseline_prompt_tokens_per_second": throughputs[0],
                    "merged_prompt_tokens_per_second": throughputs[1],
                    "throughput_ratio": throughputs[1] / throughputs[0],
                    "paired_scores": paired_score_summary(*scores),
                    "ranking_metrics": ranking_comparison(rows, *scores),
                }
            write(
                output / "baseline_comparison.json",
                {
                    "baseline_summary_sha256": sha(baseline_path / "summary.json"),
                    "concurrency": comparison,
                    "baseline_score_variation": baseline_report["score_variation"],
                },
            )
        write(output / "summary.json", report)
        print("inference_baseline_complete server_retained=true", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--reuse-server", action="store_true")
    parser.add_argument("--merged-model", type=Path)
    parser.add_argument("--output", default="baseline01")
    parser.add_argument("--rows", type=int, choices=(64, 320), default=64)
    args = parser.parse_args()
    config = yaml.safe_load((EXPERIMENT / "config.yaml").read_text())
    manifest = (
        prepared_manifest(config)
        if args.reuse_server or args.merged_model
        else prepare(config)
    )
    print(json.dumps(manifest["populations"]), flush=True)
    if args.prepare_only:
        return
    if Path(args.output).name != args.output:
        raise ValueError("output must be a single directory name")
    output = OUTPUT / args.output
    output.mkdir(parents=True, exist_ok=False)
    if args.merged_model:
        args.merged_model = args.merged_model.resolve()
        write(
            output / "merged_artifact.json",
            verify_merged_model(config, args.merged_model),
        )
    for source in [
        *EXPERIMENT.glob("*.py"),
        EXPERIMENT / "config.yaml",
        EXPERIMENT / "README.md",
        ROOT / "src/gleipnir/inference_benchmark.py",
        ROOT / "src/gleipnir/merged_lora.py",
        ROOT / "experiments/tool_trajectory_monitoring/benchmark_qwen_ood.py",
    ]:
        destination = output / "executed_sources" / source.relative_to(ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(source.read_bytes())
    write(output / "manifest.json", manifest)
    try:
        asyncio.run(
            benchmark(
                config,
                manifest,
                output,
                args.rows,
                args.reuse_server,
                args.merged_model,
            )
        )
    except BaseException as error:
        write(output / "failure.json", {"error": f"{type(error).__name__}: {error}"})
        raise


if __name__ == "__main__":
    main()
