"""Prepare and benchmark one persistent local HTTP monitor serving process."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib.metadata
import json
import os
import shutil
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
    quick_workload,
    response_score,
)

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


def server_command(config: dict) -> list[str]:
    return [
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
        "flashinfer",
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


async def benchmark(
    config: dict, manifest: dict, output: Path, count: int, reuse: bool
) -> None:
    import httpx
    import numpy as np

    command = server_command(config)
    metadata = OUTPUT / "server.json"
    base_url = f"http://127.0.0.1:{config['port']}"
    started = time.perf_counter()
    if reuse:
        receipt = json.loads(metadata.read_text())
        if (
            receipt["command"] != command
            or receipt["config_sha256"] != manifest["config_sha256"]
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
        LOGS.mkdir(parents=True, exist_ok=True)
        if shutil.which("ninja", path=environment()["PATH"]) is None:
            raise RuntimeError("serving compiler executable ninja is unavailable")
        with (LOGS / "server.log").open("x") as handle:
            process = subprocess.Popen(
                command,
                cwd=ROOT,
                env=environment(),
                stdout=handle,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        receipt = {
            "pid": process.pid,
            "command": command,
            "config_sha256": manifest["config_sha256"],
            "started_at_unix": time.time(),
            "status": "starting",
            "cache_paths": {k: v for k, v in environment().items() if "CACHE" in k},
            "log": str(LOGS / "server.log"),
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
        write(metadata, receipt)
        report = {
            "status": "running",
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
        canaries = json.loads((DATA / "canary.json").read_text())
        parity = json.loads((ROOT / config["parity"]).read_text())
        parity_start = time.perf_counter()
        comparisons, served = {}, {}
        for mode in ("base", "adapter"):
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
        effect = float(np.max(np.abs(np.array(served["adapter"]) - served["base"])))
        passed = all(
            v["mean_absolute_difference"]
            <= parity["limits"]["max_mean_absolute_difference"]
            and v["correlation"] >= parity["limits"]["min_correlation"]
            for v in comparisons.values()
        )
        passed = passed and effect >= parity["limits"]["min_adapter_effect"]
        write(
            output / "http_parity.json",
            {
                "passed": bool(passed),
                "comparisons": comparisons,
                "adapter_effect": effect,
                "served": served,
                "reference_sha256": sha(ROOT / config["parity"]),
                "seconds": time.perf_counter() - parity_start,
            },
        )
        if not passed:
            raise ValueError("HTTP scoring parity failed")
        rows = json.loads(
            (
                DATA / ("quick.json" if count == config["quick_rows"] else "full.json")
            ).read_text()
        )
        warmup = quick_workload(rows, 4, config["seed"])
        _, report["warmup_seconds"] = await trial(
            client, warmup, manifest["token_ids"], 16
        )
        print("http_parity_and_warmup_passed", flush=True)
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
        write(output / "summary.json", report)
        print("inference_baseline_complete server_retained=true", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--reuse-server", action="store_true")
    parser.add_argument("--output", default="baseline01")
    parser.add_argument("--rows", type=int, choices=(64, 320), default=64)
    args = parser.parse_args()
    config = yaml.safe_load((EXPERIMENT / "config.yaml").read_text())
    manifest = prepared_manifest(config) if args.reuse_server else prepare(config)
    print(json.dumps(manifest["populations"]), flush=True)
    if args.prepare_only:
        return
    if Path(args.output).name != args.output:
        raise ValueError("output must be a single directory name")
    output = OUTPUT / args.output
    output.mkdir(parents=True, exist_ok=False)
    for source in [
        *EXPERIMENT.glob("*.py"),
        EXPERIMENT / "config.yaml",
        EXPERIMENT / "README.md",
        ROOT / "src/gleipnir/inference_benchmark.py",
        ROOT / "experiments/tool_trajectory_monitoring/benchmark_qwen_ood.py",
    ]:
        destination = output / "executed_sources" / source.relative_to(ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(source.read_bytes())
    write(output / "manifest.json", manifest)
    try:
        asyncio.run(benchmark(config, manifest, output, args.rows, args.reuse_server))
    except BaseException as error:
        write(output / "failure.json", {"error": f"{type(error).__name__}: {error}"})
        raise


if __name__ == "__main__":
    main()
