"""Measure the selected FP8 scorer at C1 from 8K to 1M tokens on one B200."""

from __future__ import annotations

import argparse
import asyncio
import inspect
import json
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import httpx
import numpy as np

from experiments.b200_context_million.envelope import validate
from experiments.b200_context_million.environment import (
    GpuSampler,
    drift,
    snapshot,
    summarize_samples,
)
from experiments.b200_inference_benchmark.run import DATA, ROOT, sha, write
from experiments.b200_long_context.run import gpu, rpc, synthetic
from experiments.b200_monitor_score.run import SERVING, trial
from experiments.b200_vllm031.run import archive_audits
from experiments.b200_vllm031.runtime import candidate_environment
from experiments.b200_vllm031.source_bindings import restore_diagnostic_sources
from gleipnir.inference_benchmark import paired_score_summary, ranking_comparison
from gleipnir.serving.reference import selected_serving_default

EXPERIMENT = Path(__file__).parent
WORKER = "experiments.b200_context_million.worker.LongContextWorker"
# The selected default, before and after last-query attention was adopted.
PARENT_WORKERS = {
    "experiments.b200_attention_precision.worker.AttentionPrecisionWorker",
    "experiments.b200_last_layer_trim.worker.LastQueryPrecisionWorker",
}


@dataclass(frozen=True)
class Variant:
    """Server worker, bound sources and pre-launch checks for one sweep."""

    experiment: Path = EXPERIMENT
    result_group: str = "b200_context_million"
    worker: str = WORKER
    sources: tuple[str, ...] = (
        "experiments/b200_context_million/worker.py",
        "experiments/b200_context_million/envelope.py",
    )
    # (module, receipt filename, environment variable naming the receipt path)
    checks: tuple[tuple[str, str, str], ...] = field(default_factory=tuple)
    # Extra worker receipts under the shared serving directory to archive.
    audits: tuple[str, ...] = field(default_factory=tuple)
    # Server entry module replacing the selected precision server, if any.
    server: str | None = None
    # (worker RPC method, receipt filename) collected after measurement.
    rpc_audits: tuple[tuple[str, str], ...] = field(default_factory=tuple)
    # Extra server environment, e.g. GDN route settings.
    environment: dict[str, str] = field(default_factory=dict)


DEFAULT = Variant()


def phase_command(
    parent: list[str], phase: dict, chunk: int, variant: Variant = DEFAULT
) -> list[str]:
    """Change only context length, worker telemetry and, for YaRN, RoPE scaling."""
    if (
        parent[parent.index("--max-model-len") + 1] != "32768"
        or parent[parent.index("--max-num-batched-tokens") + 1] != str(chunk)
        or parent[parent.index("--max-num-seqs") + 1] != "128"
        or parent[parent.index("--runner") + 1] != "pooling"
        or "--enable-chunked-prefill" not in parent
        or "--no-enable-prefix-caching" not in parent
        or parent[parent.index("--worker-cls") + 1] not in PARENT_WORKERS
    ):
        raise ValueError("selected parent context/scheduler recipe changed")
    command = parent.copy()
    command[0] = sys.executable
    command[command.index("--max-model-len") + 1] = str(phase["context_limit"])
    command[command.index("--worker-cls") + 1] = variant.worker
    if variant.server is not None:
        if command[1:3] != ["-m", "experiments.b200_attention_precision.server"]:
            raise ValueError("selected precision server module changed")
        command[2] = variant.server
    index = command.index("--additional-config") + 1
    additional = json.loads(command[index])
    for path in variant.sources:
        additional["gleipnir_frost_fp4"][path] = sha(ROOT / path)
    command[index] = json.dumps(additional, sort_keys=True)
    override = phase["rope_override"]
    index = command.index("--hf-overrides") + 1
    overrides = json.loads(command[index])
    rope = overrides["text_config"]["rope_parameters"]
    native = overrides["text_config"]["max_position_embeddings"]
    if rope["rope_type"] != "default" or native != 262144:
        raise ValueError("selected native RoPE contract changed")
    if override is None:
        if phase["context_limit"] > native:
            raise ValueError("native RoPE phase exceeds the checkpoint position limit")
    else:
        if (
            override["rope_type"] != "yarn"
            or override["original_max_position_embeddings"] != native
            or phase["context_limit"] > native * override["factor"]
        ):
            raise ValueError("YaRN override does not cover the requested limit")
        rope.update(override)
        # vLLM 0.31 follows Transformers: YaRN configs carry the scaled limit.
        overrides["text_config"]["max_position_embeddings"] = int(
            native * override["factor"]
        )
        command[index] = json.dumps(overrides)
    return command


def server_environment(selection: dict, out: Path, binding: dict, native: Path) -> dict:
    env = candidate_environment(ROOT)
    parent = json.loads((ROOT / selection["host_parent"]).read_text())
    env.update(
        GLEIPNIR_FROST_WRAPPER_VALIDATION=parent["host_wrapper"]["validation"],
        GLEIPNIR_GIGATOKEN_RECEIPT=str(out / "frontend.json"),
        VLLM_GDN_DECODE_KERNEL="cuda",
        GLEIPNIR_FLASHINFER_GDN_CP="auto",
        GLEIPNIR_VLLM031_SCHEDULER_BINDING=json.dumps(binding),
        GLEIPNIR_LONG_CONTEXT_VALIDATION=str(native.relative_to(ROOT)),
    )
    return env


async def canary(selection: dict, settings: dict, out: Path, gate: bool) -> dict:
    values, _ = await trial(json.loads((DATA / "canary.json").read_text()), 4, settings)
    previous = json.loads((ROOT / selection["canary_predictions"]).read_text())
    if [(r["id"], r["prompt_sha256"], r["prompt_tokens"]) for r in values] != [
        (r["id"], r["prompt_sha256"], r["prompt_tokens"]) for r in previous
    ]:
        raise ValueError("canary identity changed")
    scores = np.array([r["score"] for r in values])
    expected = np.array([r["score"] for r in previous])
    master = json.loads((ROOT / selection["master_canary"]).read_text())
    if [r["prompt_sha256"] for r in values] != master["prompt_sha256"]:
        raise ValueError("master canary prompt identity changed")
    effect = float(np.abs(scores - np.array(master["base"])).max())
    limits = selection["reproduction_limits"]
    mae = float(np.abs(scores - expected).mean())
    correlation = float(np.corrcoef(scores, expected)[0, 1])
    finite = bool(np.isfinite(scores).all()) and bool(effect > 0)
    receipt = {
        "gate": gate,
        "finite_and_adapter_effect": finite,
        "reproduces_default": finite
        and mae <= limits["score_mae"]
        and correlation >= limits["correlation"],
        "score_mae": mae,
        "correlation": correlation,
        "adapter_effect": effect,
        "quality_status": "user_accepted_finite",
    }
    write(out / "canary.json", receipt)
    write(out / "canary_predictions.json", values)
    if not finite or (gate and not receipt["reproduces_default"]):
        raise ValueError(f"canary failed: {receipt}")
    return receipt


async def quality(settings: dict, out: Path) -> None:
    """Short-input batch diagnostic against the archived FP8 default repeats."""
    full = json.loads((DATA / "full.json").read_text())
    await trial(full, 128, settings)
    predictions = []
    for repeat in range(settings["quality_passes"]):
        values, _ = await trial(full, 128, settings)
        predictions.append(values)
        write(out / f"quality_repeat{repeat}.json", values)
    files = sorted((ROOT / settings["quality_controls"]).glob("c128_repeat*.json"))
    if len(files) != 6:
        raise ValueError("FP8 default quality controls incomplete")
    controls = [json.loads(p.read_text()) for p in files]
    write(
        out / "quality.json",
        {
            "controls": {str(p.relative_to(ROOT)): sha(p) for p in files},
            "scores": paired_score_summary(controls, predictions),
            "ranking": ranking_comparison(full, controls, predictions),
        },
    )


async def measure(rows: list[dict], settings: dict, out: Path, report: dict) -> None:
    for row in rows:
        length = row["prompt_tokens"]
        await rpc("long_context_memory", {"reset": True})
        warm, seconds = await trial([row], 1, settings)
        write(out / f"{length}_warmup.json", {"seconds": seconds, "values": warm})
        timings = []
        samples = out / f"{length}_gpu_samples.csv"
        with GpuSampler(samples):
            for repeat in range(settings["repeats"]):
                before = await rpc("long_context_memory", {"reset": True})
                values, seconds = await trial([row], 1, settings)
                memory = await rpc("long_context_memory")
                result = {
                    "repeat": repeat,
                    "seconds": seconds,
                    "values": values,
                    "memory_before": before,
                    "memory": memory,
                }
                write(out / f"{length}_repeat{repeat}.json", result)
                timings.append(result)
        latencies = [t["values"][0]["latency_seconds"] for t in timings]
        throughputs = [length / t["seconds"] for t in timings]
        record = {
            "prompt_tokens": length,
            "status": "complete",
            "median_latency_seconds": statistics.median(latencies),
            "min_latency_seconds": min(latencies),
            "max_latency_seconds": max(latencies),
            "prompt_tokens_per_second": statistics.median(throughputs),
            "requests_per_second": statistics.median(1 / t["seconds"] for t in timings),
            "throughput_min": min(throughputs),
            "throughput_max": max(throughputs),
            "peak_allocated_bytes": max(
                t["memory"]["peak_allocated_bytes"] for t in timings
            ),
            "peak_reserved_bytes": max(
                t["memory"]["peak_reserved_bytes"] for t in timings
            ),
            "peak_incremental_allocated_bytes": max(
                t["memory"]["peak_allocated_bytes"]
                - t["memory_before"]["allocated_bytes"]
                for t in timings
            ),
            "query_history_shapes": timings[-1]["memory"]["query_history_shapes"],
            "gpu_samples": summarize_samples(samples.read_text()),
            "scores": [t["values"][0]["score"] for t in timings],
        }
        report["measurements"].append(record)
        write(out / "summary.json", report)
        print(
            "length_complete",
            length,
            record["median_latency_seconds"],
            record["prompt_tokens_per_second"],
            flush=True,
        )


async def wait_ready(process: subprocess.Popen, timeout: float) -> float:
    started = time.perf_counter()
    async with httpx.AsyncClient(
        base_url="http://127.0.0.1:8010", trust_env=False, timeout=30
    ) as client:
        while True:
            if process.poll() is not None:
                raise RuntimeError(f"server exited: {process.returncode}")
            try:
                if (await client.get("/health")).status_code == 200:
                    return time.perf_counter() - started
            except httpx.HTTPError:
                pass
            if time.perf_counter() - started > timeout:
                raise TimeoutError("server startup timed out")
            await asyncio.sleep(2)


def retire(name: str, env: dict) -> None:
    subprocess.run(
        [sys.executable, "-m", "experiments.b200_vllm031.stop", "--archive-name", name],
        cwd=ROOT,
        env=env,
        check=True,
    )
    if gpu()["apps"]:
        raise ValueError("GPU not empty after verified server retirement")


async def run_phase(
    phase: dict,
    rows: list[dict],
    context: dict,
    settings: dict,
    out: Path,
    report: dict,
) -> None:
    selection, parent, binding, native, variant = (
        context[k] for k in ("selection", "command", "binding", "native", "variant")
    )
    out.mkdir()
    command = phase_command(parent, phase, settings["chunk_tokens"], variant)
    env = server_environment(selection, out, binding, native)
    env.update(context["check_env"])
    env.update(variant.environment)
    write(out / "command.json", command)
    log = (
        ROOT
        / "logs/runpod"
        / variant.result_group
        / f"{out.parent.name}_{out.name}.log"
    )
    log.parent.mkdir(parents=True, exist_ok=True)
    if gpu()["apps"] or (SERVING / "server.json").exists():
        raise ValueError("retire the existing server before a long-context phase")
    with log.open("x") as handle:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            env=env,
            stdout=handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    active = {
        "pid": process.pid,
        "command": command,
        "status": "starting",
        "runtime_migration": selection["runtime"],
        "log": str(log),
        "serving_default": selection["name"],
        "context_limit": phase["context_limit"],
        "rope_override": phase["rope_override"],
    }
    write(SERVING / "server.json", active)
    write(out / "server.json", active)
    try:
        record = {"phase": phase["name"], "measurements": []}
        report["phases"].append(record)
        record["startup_seconds"] = await wait_ready(
            process, settings["startup_timeout_seconds"]
        )
        active.update(status="ready", ready_at_unix=time.time())
        write(SERVING / "server.json", active)
        if (await rpc("frost_wrapper_state"))["mode"] != "direct":
            raise ValueError("direct FROST binding changed")
        rope = await rpc("rope_audit")
        write(out / "rope.json", rope)
        if not rope["covers_limit"]:
            raise ValueError("served rotary tables do not cover the context limit")
        record["canary"] = await canary(selection, settings, out, phase["canary_gate"])
        archive_audits(out)
        projections = json.loads(
            (out / "native_attention_projections.json").read_text()
        )
        if not projections["passed"] or projections["precision"] != "fp8":
            raise ValueError("actual FP8 projection dispatch changed")
        if phase["quality_diagnostic"]:
            await quality(settings, out)
        write(out / "environment_ready.json", snapshot(ROOT))
        report["status"] = f"measuring_{phase['name']}"
        await measure(rows, settings, out, record)
        for method, filename in variant.rpc_audits:
            write(out / filename, await rpc(method))
        write(
            out / "closure.json", {**gpu(), "memory": await rpc("long_context_memory")}
        )
        record["status"] = "complete"
        write(out / "summary.json", record)
    except BaseException as error:
        record["status"] = "failed"
        record["error"] = f"{type(error).__name__}: {error}"
        write(out / "summary.json", record)
        raise
    finally:
        archive_audits(out)
        for filename in variant.audits:
            source = SERVING / filename
            if source.exists():
                write(out / filename, json.loads(source.read_text()))
        if process.poll() is None and (SERVING / "server.json").exists():
            retire(f"{out.parent.name}_{out.name}", env)
        elif process.poll() is not None and (SERVING / "server.json").exists():
            if json.loads((SERVING / "server.json").read_text())["pid"] == process.pid:
                (SERVING / "server.json").unlink()


async def run(name: str, phases: list[str] | None, variant: Variant = DEFAULT) -> None:
    settings = json.loads((variant.experiment / "config.json").read_text())
    out = ROOT / "results" / variant.result_group / name
    out.mkdir(parents=True, exist_ok=False)
    write(out / "settings.json", settings)
    executed = {
        p.relative_to(ROOT)
        for directory in {EXPERIMENT, variant.experiment}
        for p in directory.iterdir()
        if p.suffix in {".py", ".json", ".md"}
    } | {Path(source) for source in variant.sources}
    for relative in sorted(executed):
        destination = out / "executed_sources" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((ROOT / relative).read_bytes())
    report = {"status": "preparing", "phases": [], "promoted": False}
    write(out / "summary.json", report)
    try:
        start = snapshot(ROOT, raw_directory=out / "environment_start")
        write(out / "environment_start.json", start)
        snapshot_gpu = gpu()
        if snapshot_gpu["apps"] or (SERVING / "server.json").exists():
            raise ValueError("long-context run requires an idle GPU and no server")
        # Variants supply their whole worker stack; start from the base recipe.
        selection, command = selected_serving_default(ROOT, last_query=False)
        from vllm.v1.core.sched import scheduler

        binding = {
            "upstream_sha256": sha(Path(inspect.getfile(scheduler))),
            "integration_sha256": sha(ROOT / "experiments/b200_vllm031/scheduler.py"),
        }
        if binding != selection["scheduler_binding"]:
            raise ValueError("accepted pooling scheduler source changed")
        archive = ROOT / "results/b200_vllm031/pre_migration_sources.tar.gz"
        additional = json.loads(command[command.index("--additional-config") + 1])
        write(
            out / "source_restoration.json",
            {
                "archive_sha256": sha(archive),
                "restored": restore_diagnostic_sources(
                    ROOT, additional["serving_condition"], archive
                ),
            },
        )
        model = Path(command[command.index("--model") + 1])
        merge = json.loads((ROOT / selection["merged_artifact"]).read_text())
        for path, expected in merge["files_sha256"].items():
            if sha(model / path) != expected:
                raise ValueError(f"accepted merged weight changed: {path}")
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
        lengths = sorted({n for p in settings["phases"] for n in p["lengths"]})
        rows = {n: synthetic(tokenizer, n) for n in lengths}
        write(out / "workload.json", list(rows.values()))
        historical = {
            r["prompt_tokens"]: r["prompt_sha256"]
            for r in json.loads((ROOT / settings["historical_workload"]).read_text())
        }
        write(
            out / "workload_binding.json",
            {
                "sha256": sha(out / "workload.json"),
                "generator": "experiments/b200_long_context/run.py:synthetic",
                "generator_sha256": sha(ROOT / "experiments/b200_long_context/run.py"),
                "kind": "deterministic_synthetic_unlabelled",
                "historical_prompt_matches": {
                    n: rows[n]["prompt_sha256"] == digest
                    for n, digest in historical.items()
                    if n in rows
                },
                "tokenizer_files": {
                    p.name: sha(p)
                    for p in model.iterdir()
                    if p.name
                    in {"tokenizer.json", "tokenizer_config.json", "config.json"}
                },
            },
        )
        native = out / "native.json"
        env = candidate_environment(ROOT)
        with (out / "native.log").open("x") as handle:
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "experiments.b200_context_million.native",
                    "--output",
                    str(native),
                ],
                cwd=ROOT,
                env=env,
                stdout=handle,
                stderr=subprocess.STDOUT,
                check=True,
            )
        validate(
            json.loads(native.read_text()),
            ROOT,
            max(p["context_limit"] for p in settings["phases"]),
        )
        check_env = {}
        for module, filename, variable in variant.checks:
            receipt = out / filename
            with (out / f"{Path(filename).stem}.log").open("x") as handle:
                subprocess.run(
                    [sys.executable, "-m", module, "--output", str(receipt)],
                    cwd=ROOT,
                    env=env,
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                    check=True,
                )
            check_env[variable] = str(receipt.relative_to(ROOT))
        context = {
            "selection": selection,
            "command": command,
            "binding": binding,
            "native": native,
            "variant": variant,
            "check_env": check_env,
        }
        for phase in settings["phases"]:
            if phases and phase["name"] not in phases:
                continue
            await run_phase(
                phase,
                [rows[n] for n in phase["lengths"]],
                context,
                settings,
                out / phase["name"],
                report,
            )
            write(out / "summary.json", report)
        end = snapshot(ROOT, raw_directory=out / "environment_end")
        write(out / "environment_end.json", end)
        report["environment_drift_during_run"] = drift(start, end)
        report["status"] = "complete"
        write(out / "summary.json", report)
        print(f"{variant.result_group}_complete", flush=True)
    except BaseException as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        report["gpu"] = gpu()
        write(out / "summary.json", report)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    parser.add_argument("--phase", action="append", choices=["native", "yarn"])
    args = parser.parse_args()
    if args.name in {"", ".", ".."} or Path(args.name).name != args.name:
        raise ValueError("run name must be a stem")
    asyncio.run(run(args.name, args.phase))


if __name__ == "__main__":
    main()
