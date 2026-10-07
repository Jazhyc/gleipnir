"""Measure the original 32K limit, then one validated long-context scorer."""

from __future__ import annotations

import argparse
import asyncio
import copy
import hashlib
import json
import statistics
import subprocess
import time
from pathlib import Path

import httpx
import numpy as np

from experiments.b200_context_scaling.run import bind_reference
from experiments.b200_inference_benchmark.run import DATA, ROOT, sha, write
from experiments.b200_monitor_score.run import SERVING, trial
from gleipnir.inference_benchmark import paired_score_summary, ranking_comparison

EXPERIMENT = Path(__file__).parent


def synthetic(tokenizer, length: int) -> dict:
    """Construct exact plain-text lengths; endpoint counts independently verify them."""
    body = "".join(
        f"Benchmark record {i}: amber cedar river glass, a neutral observation.\n"
        for i in range(256)
    )
    text = body * (length // 1024 + 2)
    for _ in range(8):
        ids = tokenizer.encode(text, add_special_tokens=False)
        if len(ids) == length:
            break
        text = tokenizer.decode(ids[:length])
    if len(tokenizer.encode(text, add_special_tokens=False)) != length:
        raise ValueError("synthetic prompt length did not round-trip")
    return {
        "id": f"synthetic{length}",
        "prompt": text,
        "prompt_tokens": length,
        "prompt_sha256": hashlib.sha256(text.encode()).hexdigest(),
    }


def extended_command(parent: list[str]) -> list[str]:
    """Keep the recipe and step budget fixed while extending total context."""
    if (
        parent[parent.index("--max-model-len") + 1] != "32768"
        or parent[parent.index("--max-num-batched-tokens") + 1] != "32768"
        or parent[parent.index("--max-num-seqs") + 1] != "128"
        or parent[parent.index("--runner") + 1] != "pooling"
        or "--enable-chunked-prefill" not in parent
        or "--no-enable-prefix-caching" not in parent
        or "--scheduler-cls" in parent
    ):
        raise ValueError("selected parent context/scheduler recipe changed")
    command = parent.copy()
    command[command.index("--max-model-len") + 1] = "262144"
    command[command.index("--worker-cls") + 1] = (
        "experiments.b200_long_context.worker.LongContextWorker"
    )
    return command


def gpu() -> dict:
    def query(option):
        return subprocess.check_output(
            ["nvidia-smi", option, "--format=csv,noheader"], text=True
        ).strip()

    return {
        "gpu": query("--query-gpu=name,uuid,memory.used,memory.total,utilization.gpu"),
        "apps": query("--query-compute-apps=pid,process_name,used_memory"),
    }


async def rpc(method: str, kwargs: dict | None = None) -> dict:
    async with httpx.AsyncClient(
        base_url="http://127.0.0.1:8010", trust_env=False, timeout=60
    ) as client:
        result = await client.post(
            "/collective_rpc",
            json={"method": method, "kwargs": kwargs or {}, "timeout": 60},
        )
        result.raise_for_status()
        return result.json()["results"][0]


async def canary(settings: dict, out: Path, phase: str) -> None:
    path = ROOT / settings["canary_reference"]
    control = json.loads(path.read_text())
    if not control["evaluation_passed"]:
        raise ValueError("accepted canary missing")
    values, _ = await trial(json.loads((DATA / "canary.json").read_text()), 4, settings)
    observed = np.array([r["score"] for r in values])
    expected = np.array(control["served"]["adapter"])
    mean = float(np.abs(observed - expected).mean())
    corr = float(np.corrcoef(observed, expected)[0, 1])
    effect = float(np.abs(observed - np.array(control["served"]["base"])).max())
    passed = (
        mean <= settings["canary_mean_error_limit"]
        and corr >= settings["canary_correlation_floor"]
        and effect > 0
    )
    write(
        out / f"{phase}_canary.json",
        {
            "passed": bool(passed),
            "mean_error": mean,
            "correlation": corr,
            "adapter_effect": effect,
            "reference_sha256": sha(path),
        },
    )
    write(out / f"{phase}_canary_predictions.json", values)
    if not passed:
        raise ValueError("real adapter canary failed")


async def measure_lengths(
    rows: list[dict], settings: dict, out: Path, report: dict, phase: str
) -> None:
    for row in rows:
        length = row["prompt_tokens"]
        before = gpu()
        if phase == "extended":
            await rpc("long_context_memory", {"reset": True})
        warm, seconds = await trial([row], 1, settings)
        write(
            out / f"{phase}_{length}_warmup.json", {"seconds": seconds, "values": warm}
        )
        timings = []
        for repeat in range(settings["repeats"]):
            memory_before = (
                await rpc("long_context_memory", {"reset": True})
                if phase == "extended"
                else None
            )
            values, seconds = await trial([row], 1, settings)
            memory = await rpc("long_context_memory") if phase == "extended" else None
            result = {
                "repeat": repeat,
                "seconds": seconds,
                "values": values,
                "memory_before": memory_before,
                "memory": memory,
            }
            write(out / f"{phase}_{length}_repeat{repeat}.json", result)
            timings.append(result)
        latencies = [t["values"][0]["latency_seconds"] for t in timings]
        throughputs = [length / t["seconds"] for t in timings]
        record = {
            "phase": phase,
            "prompt_tokens": length,
            "status": "complete",
            "median_latency_seconds": statistics.median(latencies),
            "min_latency_seconds": min(latencies),
            "max_latency_seconds": max(latencies),
            "prompt_tokens_per_second": statistics.median(throughputs),
            "requests_per_second": statistics.median(1 / t["seconds"] for t in timings),
            "throughput_min": min(throughputs),
            "throughput_max": max(throughputs),
            "before": before,
            "after": gpu(),
        }
        if phase == "extended":
            record["peak_allocated_bytes"] = max(
                t["memory"]["peak_allocated_bytes"] for t in timings
            )
            record["peak_reserved_bytes"] = max(
                t["memory"]["peak_reserved_bytes"] for t in timings
            )
            record["peak_incremental_allocated_bytes"] = max(
                t["memory"]["peak_allocated_bytes"]
                - t["memory_before"]["allocated_bytes"]
                for t in timings
            )
        report["measurements"].append(record)
        write(out / "summary.json", report)
        print(
            "length_complete",
            phase,
            length,
            record["median_latency_seconds"],
            record["prompt_tokens_per_second"],
            flush=True,
        )


async def run(name: str) -> None:
    settings = json.loads((EXPERIMENT / "config.json").read_text())
    out = ROOT / "results/b200_long_context" / name
    out.mkdir(parents=True, exist_ok=False)
    write(out / "settings.json", settings)
    for p in EXPERIMENT.iterdir():
        if p.suffix in {".py", ".json", ".md"}:
            destination = out / "executed_sources" / p.name
            destination.parent.mkdir(exist_ok=True)
            destination.write_bytes(p.read_bytes())
    report = {"status": "preparing", "measurements": [], "promoted": False}
    write(out / "summary.json", report)
    process = None
    try:
        parent = json.loads((SERVING / "server.json").read_text())
        bind_reference(
            parent,
            json.loads((ROOT / settings["reference_run"] / "summary.json").read_text()),
        )
        snapshot = gpu()
        if (
            settings["gpu_uuid"] not in snapshot["gpu"]
            or len(snapshot["apps"].splitlines()) != 1
        ):
            raise ValueError("exclusive B200 identity changed")
        write(out / "parent_server.json", parent)
        write(out / "initial_gpu.json", snapshot)
        command = extended_command(parent["command"])
        # Retain the existing authorized runtime in memory; never serialize env.
        env = dict(
            v.decode().split("=", 1)
            for v in Path(f"/proc/{parent['pid']}/environ").read_bytes().split(b"\0")
            if v
        )
        from transformers import AutoTokenizer

        model = Path(command[command.index("--model") + 1])
        config = json.loads((model / "config.json").read_text())
        if (
            config.get("text_config", config)["max_position_embeddings"]
            < settings["context_limit"]
        ):
            raise ValueError("requested length exceeds native model context")
        tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
        rows = [synthetic(tokenizer, n) for n in settings["lengths"]]
        write(out / "workload.json", rows)
        write(
            out / "workload_binding.json",
            {
                "sha256": sha(out / "workload.json"),
                "generator_sha256": sha(EXPERIMENT / "run.py"),
                "kind": "deterministic_synthetic_unlabelled",
                "tokenizer_files": {
                    p.name: sha(p)
                    for p in model.iterdir()
                    if p.name
                    in {"tokenizer.json", "tokenizer_config.json", "config.json"}
                },
            },
        )
        await canary(settings, out, "original")
        report["status"] = "running_original"
        await measure_lengths(rows[:3], settings, out, report, "original")
        async with httpx.AsyncClient(trust_env=False, timeout=60) as client:
            rejection = await client.post(
                "http://127.0.0.1:8010/v1/monitor/score",
                json={"model": "monitor", "prompt": rows[3]["prompt"]},
            )
            write(
                out / "original_64k_rejection.json",
                {"status_code": rejection.status_code, "body": rejection.text},
            )
            if rejection.status_code != 400:
                raise ValueError("original context limit behavior changed")
        subprocess.run(
            [
                command[0],
                "-m",
                "experiments.b200_attention_gdn_serving.stop_server",
                "--archive-name",
                name,
                "--reason",
                "User-requested C1 long-context capacity benchmark",
            ],
            env=env,
            cwd=ROOT,
            check=True,
        )
        if gpu()["apps"]:
            raise ValueError("GPU not empty after verified server retirement")
        native_path = out / "native.json"
        with (out / "native.log").open("x") as handle:
            subprocess.run(
                [
                    command[0],
                    "-m",
                    "experiments.b200_long_context.native",
                    "--output",
                    str(native_path),
                ],
                env=env,
                cwd=ROOT,
                stdout=handle,
                stderr=subprocess.STDOUT,
                check=True,
            )
        from experiments.b200_long_context.envelope import validate

        validate(json.loads(native_path.read_text()), ROOT)
        env["GLEIPNIR_LONG_CONTEXT_VALIDATION"] = str(native_path.relative_to(ROOT))
        env["GLEIPNIR_GIGATOKEN_RECEIPT"] = str(out / "frontend.json")
        index = command.index("--additional-config") + 1
        additional = json.loads(command[index])
        for filename in ("worker.py", "envelope.py"):
            path = str((EXPERIMENT / filename).relative_to(ROOT))
            additional["gleipnir_frost_fp4"][path] = sha(ROOT / path)
        command[index] = json.dumps(additional, sort_keys=True)
        server = copy.deepcopy(parent)
        log = ROOT / "logs/runpod/b200_attention_gdn_serving/server.log"
        started = time.perf_counter()
        with log.open("x") as handle:
            process = subprocess.Popen(
                command,
                env=env,
                cwd=ROOT,
                stdout=handle,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        server.update(
            pid=process.pid,
            command=command,
            status="starting",
            parent_pid=parent["pid"],
            started_at_unix=time.time(),
            context_limit=settings["context_limit"],
            long_context_validation=str(native_path.relative_to(ROOT)),
        )
        server["frontend"]["receipt_path"] = str(out / "frontend.json")
        write(SERVING / "server.json", server)
        write(out / "server.json", server)
        async with httpx.AsyncClient(
            base_url="http://127.0.0.1:8010", trust_env=False, timeout=30
        ) as client:
            while True:
                if process.poll() is not None:
                    raise RuntimeError(f"extended server exited: {process.returncode}")
                try:
                    if (await client.get("/health")).status_code == 200:
                        break
                except httpx.HTTPError:
                    pass
                if time.perf_counter() - started > settings["startup_timeout_seconds"]:
                    raise TimeoutError("extended server startup timed out")
                await asyncio.sleep(2)
        server.update(status="ready", ready_at_unix=time.time())
        from gleipnir.serving_cache_mirror import persist_compiler_mirror

        server["compiler_cache_persistence"] = persist_compiler_mirror(
            server.get("compiler_mirror")
        )
        write(SERVING / "server.json", server)
        write(out / "server.json", server)
        report["startup_seconds"] = time.perf_counter() - started
        print(
            "extended_server_ready", process.pid, report["startup_seconds"], flush=True
        )
        if (await rpc("frost_wrapper_state"))["mode"] != "direct":
            raise ValueError("direct FROST binding changed")
        await canary(settings, out, "extended")
        # Preserve full320 quality diagnostics on unchanged labelled inputs.
        full = json.loads((DATA / "full.json").read_text())
        await trial(full, 128, settings)
        predictions = []
        for repeat in range(3):
            values, _ = await trial(full, 128, settings)
            predictions.append(values)
            write(out / f"quality_repeat{repeat}.json", values)
        selection = json.loads(
            (ROOT / "experiments/b200_inference_benchmark/baseline.json").read_text()
        )
        baseline = ROOT / selection["results"]
        files = sorted(baseline.glob("c128_repeat*.json"))
        controls = [json.loads(p.read_text()) for p in files]
        if len(controls) != 6:
            raise ValueError("selected quality controls incomplete")
        write(
            out / "quality.json",
            {
                "controls": {str(p.relative_to(ROOT)): sha(p) for p in files},
                "scores": paired_score_summary(controls, predictions),
                "ranking": ranking_comparison(full, controls, predictions),
            },
        )
        report["status"] = "running_extended"
        await measure_lengths(rows, settings, out, report, "extended")
        for filename in (
            "monitor_score.json",
            "loaded_precision.json",
            "compile_identity.json",
            "native_attention.json",
            "native_preparation.json",
            "native_gemm_tuning.json",
            "native_attention_projections.json",
            "native_swiglu_output.json",
        ):
            write(out / filename, json.loads((SERVING / filename).read_text()))
        write(
            out / "closure.json", {**gpu(), "memory": await rpc("long_context_memory")}
        )
        report["status"] = "complete"
        write(out / "summary.json", report)
        print("long_context_complete", flush=True)
    except BaseException as error:
        report.update(
            status="failed", error=f"{type(error).__name__}: {error}", gpu=gpu()
        )
        write(out / "summary.json", report)
        # Retire only this trial's identified worker, never restore automatically.
        if process is not None and (SERVING / "server.json").exists():
            current = json.loads((SERVING / "server.json").read_text())
            if current["pid"] == process.pid and process.poll() is None:
                subprocess.run(
                    [
                        command[0],
                        "-m",
                        "experiments.b200_attention_gdn_serving.stop_server",
                        "--archive-name",
                        name + "_failed",
                        "--reason",
                        report["error"],
                    ],
                    env=env,
                    cwd=ROOT,
                    check=True,
                )
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    args = parser.parse_args()
    if args.name in {"", ".", ".."} or Path(args.name).name != args.name:
        raise ValueError("run name must be a stem")
    asyncio.run(run(args.name))


if __name__ == "__main__":
    main()
