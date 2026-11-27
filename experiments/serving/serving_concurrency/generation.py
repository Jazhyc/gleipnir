"""Time a base model as a monitor that writes a rationale and then a score.

Stock vLLM in ordinary generation mode, BF16. The prompt is the instruction
of the visible-reasoning OOD benchmark
(``experiments/training/tool_trajectory_monitoring/prompts/teacher_ordinal_reasoning.txt``:
a concise analysis, then ``Score:<0-10>``) around each agent trajectory of the
serving benchmark, with Qwen's non-thinking assistant boundary. One request
generates the whole answer, greedily, with no cap below the context limit.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path

from experiments.serving.serving_concurrency.run import (
    LOGS,
    OUT,
    PORT,
    ROOT,
    agent_only,
)

INSTRUCTION = (
    ROOT
    / "experiments/training/tool_trajectory_monitoring/prompts"
    / "teacher_ordinal_reasoning.txt"
)
INSTRUCTION_SHA256 = "f79026c915e85e00c1f42790e9ab415f2f8221b166c42ce36a8594a9d72241fa"
MODELS = {
    "4b": ("Qwen/Qwen3.5-4B", "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"),
    "27b": ("Qwen/Qwen3.8-27B", "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"),
}
OPEN, CLOSE = "<agent_trajectory>", "</agent_trajectory>"
SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
CONTEXT = 36864  # the reasoning benchmark's limit: prompt plus 4,096 tokens
MAX_TOKENS = 4096
SCORE = re.compile(r"Score:\s*(\d+)\s*$")


def rationale_prompt(rendered: str, instruction: str) -> str:
    """Swap the monitor instruction of a rendered benchmark prompt."""
    start, end = rendered.index(OPEN), rendered.rindex(CLOSE) + len(CLOSE)
    if not rendered.endswith("Prediction:") or end < start:
        raise ValueError("unexpected benchmark prompt")
    return f"<|im_start|>user\n{instruction}\n\n{rendered[start:end]}{SUFFIX}"


def serve_command(model: str) -> list[str]:
    name, revision = MODELS[model]
    overrides = {"text_config": {"mamba_ssm_dtype": "float32"}}
    return [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", name,
        "--revision", revision,
        "--served-model-name", "monitor",
        "--host", "127.0.0.1",
        "--port", str(PORT),
        "--dtype", "bfloat16",
        "--language-model-only",
        "--max-model-len", str(CONTEXT),
        "--no-enable-prefix-caching",
        "--seed", "0",
        "--hf-overrides", json.dumps(overrides),
    ]  # fmt: skip


async def one_pass(prompts: list[dict], concurrency: int) -> tuple[list[dict], float]:
    import httpx

    async with httpx.AsyncClient(
        base_url=f"http://127.0.0.1:{PORT}",
        trust_env=False,
        timeout=1800,
        limits=httpx.Limits(max_connections=concurrency),
    ) as client:
        semaphore = asyncio.Semaphore(concurrency)

        async def one(row: dict) -> dict:
            async with semaphore:
                before = time.perf_counter()
                response = await client.post(
                    "/v1/completions",
                    json={
                        "model": "monitor",
                        "prompt": row["prompt"],
                        "max_tokens": MAX_TOKENS,
                        "temperature": 0.0,
                        "seed": 0,
                    },
                )
                latency = time.perf_counter() - before
            response.raise_for_status()
            body = response.json()
            text = body["choices"][0]["text"]
            match = SCORE.search(text.strip())
            return {
                "id": row["id"],
                "latency_seconds": latency,
                "prompt_tokens": body["usage"]["prompt_tokens"],
                "completion_tokens": body["usage"]["completion_tokens"],
                "finish_reason": body["choices"][0]["finish_reason"],
                "score": int(match.group(1)) if match else None,
                "text": text,
            }

        before = time.perf_counter()
        rows = await asyncio.gather(*(one(row) for row in prompts))
        return rows, time.perf_counter() - before


def summary(rows: list[dict], seconds: float) -> dict:
    latencies = sorted(r["latency_seconds"] for r in rows)
    completions = [r["completion_tokens"] for r in rows]
    prompt_tokens = sum(r["prompt_tokens"] for r in rows)
    return {
        "requests": len(rows),
        "seconds": seconds,
        "prompt_tokens": prompt_tokens,
        "prompt_tokens_per_second": prompt_tokens / seconds,
        "completion_tokens_mean": statistics.mean(completions),
        "completion_tokens_median": statistics.median(completions),
        "completion_tokens_max": max(completions),
        "p50_ms": 1000 * statistics.median(latencies),
        "p95_ms": 1000 * latencies[int(0.95 * (len(latencies) - 1))],
        "valid_scores": sum(
            r["score"] is not None and 0 <= r["score"] <= 10 for r in rows
        ),
        "length_capped": sum(r["finish_reason"] == "length" for r in rows),
    }


async def measure(out: Path, prompts: list[dict], levels: list[int], repeats: int):
    await one_pass(prompts[:8], 8)
    report = {}
    for concurrency in levels:
        passes = []
        for repeat in range(repeats):
            rows, seconds = await one_pass(prompts, concurrency)
            (out / f"generation_c{concurrency}_repeat{repeat}.json").write_text(
                json.dumps({"rows": rows, "seconds": seconds})
            )
            passes.append(summary(rows, seconds))
            print(
                f"c{concurrency} repeat {repeat}: {json.dumps(passes[-1])}", flush=True
            )
        report[f"c{concurrency}"] = passes
    (out / "summary.json").write_text(json.dumps(report, indent=1) + "\n")


def run(model: str, session: str, levels: list[int], repeats: int, limit: int | None):
    import hashlib

    import httpx

    from experiments.b200_vllm031.runtime import candidate_environment
    from experiments.serving.qwen38_27b_serving.bench import workloads

    instruction = INSTRUCTION.read_text().strip()
    if hashlib.sha256(INSTRUCTION.read_bytes()).hexdigest() != INSTRUCTION_SHA256:
        raise ValueError("reasoning instruction changed")
    rows = agent_only(workloads()["full"])[:limit]
    prompts = [
        {"id": r["id"], "prompt": rationale_prompt(r["prompt"], instruction)}
        for r in rows
    ]
    name = f"generation_{model}"
    out = OUT / session / name
    out.mkdir(parents=True, exist_ok=False)
    log = LOGS / f"{session}_{name}.log".lstrip("_")
    log.parent.mkdir(parents=True, exist_ok=True)
    command = serve_command(model)
    env = candidate_environment(ROOT)
    started = time.perf_counter()
    with log.open("x") as handle:
        process = subprocess.Popen(
            command, cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT
        )
    try:
        while True:
            if process.poll() is not None:
                raise RuntimeError("generation server exited during startup")
            try:
                if httpx.get(f"http://127.0.0.1:{PORT}/health").status_code == 200:
                    break
            except httpx.HTTPError:
                pass
            time.sleep(3)
        startup = time.perf_counter() - started
        asyncio.run(measure(out, prompts, levels, repeats))
        (out / "condition.json").write_text(
            json.dumps(
                {
                    "model": MODELS[model],
                    "command": command,
                    "startup_seconds": startup,
                    "prompts": len(prompts),
                    "instruction_sha256": INSTRUCTION_SHA256,
                    "max_tokens": MAX_TOKENS,
                    "levels": levels,
                    "repeats": repeats,
                    "pid": os.getpid(),
                },
                indent=1,
            )
        )
    finally:
        process.terminate()
        process.wait(timeout=120)
    print(f"generation_complete {name}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=sorted(MODELS))
    parser.add_argument("--session", default="generation01")
    parser.add_argument("--levels", type=int, nargs="+", default=[1, 8])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    run(args.model, args.session, args.levels, args.repeats, args.limit)


if __name__ == "__main__":
    main()
