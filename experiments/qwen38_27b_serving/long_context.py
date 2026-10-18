"""Single-request latency and throughput of the 27B from 1K to 1M tokens.

Stock BF16 (`Bf16Worker`) against the recommended optimized stack
(`QuantFullWorker`, piecewise graphs to 2,048 tokens), at concurrency 1, on
the deterministic unlabelled synthetic prompts of `b200_long_context`
(exact token counts). As in `experiments/b200_context_million`: a native-RoPE
phase to the checkpoint's 262,144 positions (1K-256K), then Qwen's YaRN x4
(`--max-model-len 1048576`; 256K overlap, 512K, 1M). YaRN changes scores, so
scores beyond 256K are timing payloads, not validated monitor outputs. Only
the context limit, RoPE and, for MXFP8 beyond 32K, the hash-bound envelope
change; the 32K chunk budget, 128-sequence cap and scheduler are unchanged.

Timing: one excluded warmup per length, then 5 sequential requests (3 from
128K up). Latency includes localhost HTTP and tokenization.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import subprocess
from pathlib import Path

from experiments.b200_long_context.run import synthetic
from experiments.b200_monitor_score.run import trial
from experiments.qwen38_27b_serving.bench import (
    LOGS,
    MODEL,
    OUT,
    ROOT,
    SETTINGS,
    quantized_command,
    sources,
)
from gleipnir.data.monitoring import file_hash, write_json
from gleipnir.serving import bf16_launch

NATIVE = 262144
PHASES = {
    "native": {
        "limit": NATIVE,
        "yarn": False,
        "lengths": [2**i for i in range(10, 19)],
    },
    "yarn": {
        "limit": 4 * NATIVE,
        "yarn": True,
        "lengths": [NATIVE, 2 * NATIVE, 4 * NATIVE],
    },
}
GRAPHS = [1, 2, 4, *range(8, 257, 8), 320, 384, 448, 512, 640, 768, 896, 1024]
GRAPHS += [1280, 1536, 1792, 2048]
RECEIPTS = {
    "last_query_attention": "last_query_mxfp8.json",
    "mxfp8_attention": "mxfp8_canary01.json",
    "fp4_gemm_tuning": "fp4_gemm_tune01.json",
    "fp4_producers": "fp4_producers01.json",
    "mxfp8_long_context": "mxfp8_envelope01.json",
}


def binding(name: str) -> dict:
    path = OUT / name
    return {
        "validation": str(path.relative_to(ROOT)),
        "validation_sha256": file_hash(path),
    }


def condition(name: str) -> dict:
    if name == "bf16":
        return {"worker": bf16_launch.BF16_WORKER, "additional": {}, "quantized": False}
    if name == "optimized":
        return {
            "worker": "gleipnir.serving.quant_stack.QuantFullWorker",
            "additional": {key: binding(path) for key, path in RECEIPTS.items()},
            "quantized": True,
        }
    raise ValueError(f"unknown condition {name}")


def phase_command(spec: dict, phase: dict) -> tuple[dict, list[str]]:
    selection, command = bf16_launch.bf16_command(
        ROOT, MODEL, sources(), worker=spec["worker"], additional=spec["additional"]
    )
    if spec["quantized"]:
        quantized_command(command, "gleipnir_geometry_fp8_fp4", spec["additional"])
        command += [
            "--compilation-config",
            json.dumps({"cudagraph_capture_sizes": GRAPHS}),
        ]
    if (
        command[command.index("--max-model-len") + 1] != "32768"
        or command[command.index("--max-num-batched-tokens") + 1] != "32768"
    ):
        raise ValueError("selected context/chunk recipe changed")
    command[command.index("--max-model-len") + 1] = str(phase["limit"])
    index = command.index("--hf-overrides") + 1
    overrides = json.loads(command[index])
    text = overrides["text_config"]
    if text["rope_parameters"]["rope_type"] != "default" or (
        text["max_position_embeddings"] != NATIVE
    ):
        raise ValueError("served native RoPE contract changed")
    if phase["yarn"]:
        text["rope_parameters"].update(
            rope_type="yarn", factor=4.0, original_max_position_embeddings=NATIVE
        )
        # vLLM 0.31 follows Transformers: YaRN configs carry the scaled limit.
        text["max_position_embeddings"] = phase["limit"]
    elif phase["limit"] > NATIVE:
        raise ValueError("native RoPE phase exceeds the checkpoint position limit")
    command[index] = json.dumps(overrides)
    return selection, command


def gpu_memory() -> int:
    used = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        text=True,
    )
    return int(used.split()[0])


async def measure(rows: list[dict], out: Path, report: dict) -> None:
    settings = {**SETTINGS, "timeout_seconds": 3600}
    for row in rows:
        length = row["prompt_tokens"]
        warm, _ = await trial([row], 1, settings)
        repeats = []
        for _ in range(5 if length < 131072 else 3):
            values, seconds = await trial([row], 1, settings)
            repeats.append({"seconds": seconds, **values[0]})
        latencies = [r["latency_seconds"] for r in repeats]
        record = {
            "prompt_tokens": length,
            "warmup_latency_seconds": warm[0]["latency_seconds"],
            "median_latency_seconds": statistics.median(latencies),
            "min_latency_seconds": min(latencies),
            "max_latency_seconds": max(latencies),
            "prompt_tokens_per_second": length / statistics.median(latencies),
            "gpu_memory_used_mib": gpu_memory(),
            "scores": [r["score"] for r in repeats],
            "repeats": repeats,
        }
        report["measurements"].append(record)
        write_json(out / "summary.json", report)
        print(
            f"length {length}: {record['median_latency_seconds']:.3f} s,"
            f" {record['prompt_tokens_per_second']:.0f} prompt tok/s",
            flush=True,
        )


def run(name: str, condition_name: str, phase_name: str) -> None:
    from transformers import AutoTokenizer

    spec, phase = condition(condition_name), PHASES[phase_name]
    out = OUT / name / f"{condition_name}_{phase_name}"
    out.mkdir(parents=True, exist_ok=False)
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    rows = [synthetic(tokenizer, n) for n in phase["lengths"]]
    selection, command = phase_command(spec, phase)
    env = bf16_launch.environment(ROOT, selection, out / "bf16_audit.json")
    log = LOGS / f"{name}_{condition_name}_{phase_name}.log"
    process = bf16_launch.start(ROOT, command, env, log, out / "server.json")
    report = {
        "condition": condition_name,
        "phase": phase_name,
        "context_limit": phase["limit"],
        "yarn": phase["yarn"],
        "scores_validated": not phase["yarn"],
        "workload": "experiments/b200_long_context/run.py:synthetic",
        "prompt_sha256": {r["prompt_tokens"]: r["prompt_sha256"] for r in rows},
        "measurements": [],
    }
    try:
        report["startup_seconds"] = bf16_launch.wait_ready(
            process, SETTINGS["port"], 3600
        )
        report["gpu_memory_ready_mib"] = gpu_memory()
        asyncio.run(measure(rows, out, report))
        audit = json.loads((out / "bf16_audit.json").read_text())
        if not audit["passed"]:
            raise ValueError("serving audit failed")
        report["audit"] = {
            key: audit[key].get("passed")
            if isinstance(audit[key], dict)
            else audit[key]
            for key in ("mxfp8_attention", "mxfp8_long_context", "fp4_producers")
            if key in audit
        }
        report["status"] = "complete"
    except BaseException as error:
        report["status"] = f"failed: {type(error).__name__}: {error}"
        raise
    finally:
        write_json(out / "summary.json", report)
        bf16_launch.stop(process)
    print(f"phase_complete {condition_name} {phase_name}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("name")
    parser.add_argument("condition", choices=["bf16", "optimized"])
    parser.add_argument("phase", choices=list(PHASES))
    args = parser.parse_args()
    run(args.name, args.condition, args.phase)


if __name__ == "__main__":
    main()
