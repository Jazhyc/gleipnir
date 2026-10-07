"""Replace the verified server, validate the head, and benchmark monitor scores."""

from __future__ import annotations

import argparse
import asyncio
import copy
import json
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path

import httpx
import numpy as np

from experiments.b200_inference_benchmark.run import DATA, ROOT, sha, write
from gleipnir.inference_benchmark import (
    measurement_summary,
    paired_score_summary,
    ranking_comparison,
)
from gleipnir.serving.monitor_score import (
    ENDPOINT,
    classification_overrides,
    validate_score_response,
)

EXPERIMENT = Path(__file__).parent
SERVING = ROOT / "results/b200_attention_gdn_serving"
SOURCES = [
    "src/gleipnir/serving/monitor_score.py",
    *[
        f"experiments/b200_monitor_score/{n}"
        for n in ["server.py", "worker.py", "run.py", "config.json", "README.md"]
    ],
]


def score_command(parent: list[str], hf_config: dict, hashes: dict) -> list[str]:
    """Preserve the admitted cached backbone, explicitly replacing its runner."""
    command = parent.copy()
    if (
        "--no-enable-prefix-caching" not in command
        or "--enable-chunked-prefill" not in command
    ):
        raise ValueError("monitor requires the selected cached parent")
    if any(
        flag in command for flag in ["--runner", "--hf-overrides", "--pooler-config"]
    ):
        raise ValueError("parent already overrides scoring")
    command[command.index("-m") + 1] = "experiments.b200_monitor_score.server"
    command[command.index("--worker-cls") + 1] = (
        "experiments.b200_monitor_score.worker.MonitorScoreAuditWorker"
    )
    if "--logprobs-mode" in command:
        index = command.index("--logprobs-mode")
        del command[index : index + 2]
    index = command.index("--additional-config") + 1
    additional = json.loads(command[index])
    additional["gleipnir_frost_fp4"].update(hashes)
    additional["monitor_score"] = {"token_ids": [15, 16], "head_dtype": "bfloat16"}
    command[index] = json.dumps(additional, sort_keys=True)
    command += [
        "--runner",
        "pooling",
        "--convert",
        "classify",
        "--pooler-config",
        json.dumps(
            {"task": "classify", "pooling_type": "LAST", "use_activation": False}
        ),
        "--hf-overrides",
        json.dumps(classification_overrides(hf_config)),
    ]
    return command


def selected_score_command(parent: list[str], template: list[str]) -> list[str]:
    """Restore the saved recipe without carrying experimental graph flags."""
    if (
        parent[0] != template[0]
        or parent[parent.index("--model") + 1]
        != template[template.index("--model") + 1]
        or template[template.index("--runner") + 1] != "pooling"
    ):
        raise ValueError("selected recipe runtime/model differs from live parent")
    return template.copy()


async def trial(
    rows: list[dict], concurrency: int, settings: dict
) -> tuple[list[dict], float]:
    """Time complete localhost score requests, including text encoding."""
    semaphore = asyncio.Semaphore(concurrency)
    async with httpx.AsyncClient(
        base_url=f"http://127.0.0.1:{settings['port']}",
        trust_env=False,
        timeout=settings["timeout_seconds"],
        limits=httpx.Limits(max_connections=256, max_keepalive_connections=128),
    ) as client:

        async def one(row):
            async with semaphore:
                before = time.perf_counter()
                response = await client.post(
                    ENDPOINT, json={"model": "monitor", "prompt": row["prompt"]}
                )
                response.raise_for_status()
                elapsed = time.perf_counter() - before
                return {
                    "id": row["id"],
                    "prompt_sha256": row["prompt_sha256"],
                    "prompt_tokens": row["prompt_tokens"],
                    "latency_seconds": elapsed,
                    **validate_score_response(response.json(), row["prompt_tokens"]),
                }

        before = time.perf_counter()
        values = await asyncio.gather(*(one(row) for row in rows))
        return values, time.perf_counter() - before


async def measure(
    name: str,
    *,
    experiment: Path = EXPERIMENT,
    result_group: str = "b200_monitor_score",
    native_prepare: Callable | None = None,
    retired_parent: Path | None = None,
    resumed_environment: dict | None = None,
    ready_prepare: Callable | None = None,
    command_prepare: Callable | None = None,
    environment_prepare: Callable | None = None,
    startup_only: bool = False,
) -> None:
    settings = json.loads((experiment / "config.json").read_text())
    assert not settings["promote"] and settings["endpoint"] == ENDPOINT
    out = ROOT / "results" / result_group / name
    out.mkdir(parents=True, exist_ok=False)
    write(out / "settings.json", settings)
    manifest = json.loads((DATA / "manifest.json").read_text())
    workloads = {}
    for key, digest in manifest["files"].items():
        assert sha(DATA / f"{key}.json") == digest, "frozen prompt drift"
        workloads[key] = json.loads((DATA / f"{key}.json").read_text())
    write(out / "manifest.json", manifest)
    sources = [*SOURCES, *settings.get("additional_sources", [])]
    hashes = {p: sha(ROOT / p) for p in sources}
    for p in sources:
        target = out / "executed_sources" / p
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT / p).read_bytes())
    parent = json.loads((retired_parent or (SERVING / "server.json")).read_text())
    if retired_parent is None:
        actual = [
            v.decode()
            for v in Path(f"/proc/{parent['pid']}/cmdline").read_bytes().split(b"\0")
            if v
        ]
        if parent["status"] != "ready" or actual != parent["command"]:
            raise ValueError("parent server identity changed")
    else:
        if (SERVING / "server.json").exists() or Path(
            f"/proc/{parent['pid']}"
        ).exists():
            raise ValueError("retired-parent recovery requires no live server")
        actual = parent["command"]
    # Retain the exact staged runtime environment in memory only; never emit it
    # into artifacts because inherited process environments can contain secrets.
    environment = (
        resumed_environment
        if retired_parent is not None
        else dict(
            v.decode().split("=", 1)
            for v in Path(f"/proc/{parent['pid']}/environ").read_bytes().split(b"\0")
            if v
        )
    )
    if environment is None:
        raise ValueError("retired parent requires a verified runtime environment")
    merged = Path(actual[actual.index("--model") + 1])
    hf_config = json.loads((merged / "config.json").read_text())
    compute_sources = {
        p: digest
        for p, digest in hashes.items()
        if p
        in {
            "src/gleipnir/serving/monitor_score.py",
            "src/gleipnir/serving/triton_mutation.py",
        }
        or p.endswith(("/server.py", "/worker.py"))
    }
    if settings.get("score_parent"):
        if actual[actual.index("--runner") + 1] != "pooling":
            raise ValueError("score trial requires the selected pooling parent")
        if settings.get("restore_selected_recipe"):
            from gleipnir.serving.reference import selected_score_reference

            selected = selected_score_reference(ROOT)
            recipe = json.loads((ROOT / selected["server_metadata"]).read_text())
            command = selected_score_command(actual, recipe["command"])
            environment["GLEIPNIR_STRIDE_VALIDATION"] = selected["mutation_validation"]
        else:
            command = actual.copy()
        index = command.index("--additional-config") + 1
        additional = json.loads(command[index])
        additional["gleipnir_frost_fp4"].update(compute_sources)
        command[index] = json.dumps(additional, sort_keys=True)
    elif settings.get("mutation_fix"):
        if actual[actual.index("--runner") + 1] != "pooling":
            raise ValueError("mutation trial requires the warm score parent")
        command = actual.copy()
        command[command.index("-m") + 1] = "experiments.b200_mutation_analysis.server"
        index = command.index("--additional-config") + 1
        additional = json.loads(command[index])
        additional["gleipnir_frost_fp4"].update(compute_sources)
        additional["qk_mutation_analysis"] = True
        command[index] = json.dumps(additional, sort_keys=True)
        environment["GLEIPNIR_STRIDE_VALIDATION"] = str(
            (out / "native.json").relative_to(ROOT)
        )
    else:
        command = score_command(actual, hf_config, compute_sources)
    if command_prepare is not None:
        command = command_prepare(command, settings, hashes)
    if environment_prepare is not None:
        environment_prepare(environment, settings, hashes, out)
    environment["GLEIPNIR_GIGATOKEN_RECEIPT"] = str(out / "frontend.json")
    write(out / "parent_server.json", parent)
    write(
        out / "merged_artifact.json",
        json.loads((merged / "merge_manifest.json").read_text()),
    )
    if settings["baseline"] == "selected":
        from gleipnir.serving.reference import selected_score_reference

        selection = selected_score_reference(ROOT)
        write(out / "reference_selection.json", selection)
        baseline = ROOT / selection["results"]
    else:
        baseline = ROOT / settings["baseline"]
    baseline_files = sorted(baseline.glob("c*_repeat*.json"))
    assert len(baseline_files) == 9, "cached repeat reference incomplete"
    canary_path = ROOT / settings["canary_reference"]
    control = json.loads(canary_path.read_text())
    assert control["evaluation_passed"], "accepted cached canary missing"
    write(
        out / "baseline_binding.json",
        {str(p.relative_to(ROOT)): sha(p) for p in [canary_path, *baseline_files]},
    )
    if retired_parent is None:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "experiments.b200_attention_gdn_serving.stop_server",
                "--archive-name",
                name,
                "--reason",
                settings.get(
                    "retirement_reason",
                    "Replace generation with a two-logit monitoring endpoint",
                ),
            ],
            cwd=ROOT,
            check=True,
        )
    if native_prepare is not None:
        native_prepare(environment, out)
    log = ROOT / "logs/runpod/b200_attention_gdn_serving/server.log"
    started = time.perf_counter()
    with log.open("x") as handle:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            env=environment,
            stdout=handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    server = copy.deepcopy(parent)
    server["frontend"]["receipt_path"] = str(out / "frontend.json")
    entrypoint = command[command.index("-m") + 1].replace(".", "/") + ".py"
    server["frontend"]["entrypoint_sha256"] = hashes[entrypoint]
    server.update(
        pid=process.pid,
        command=command,
        status="starting",
        started_at_unix=time.time(),
        runner="pooling",
        endpoint=ENDPOINT,
        parent_pid=parent["pid"],
        score_sources=hashes,
    )
    write(SERVING / "server.json", server)
    report = {"status": "starting", "trials": [], "server": server, "promoted": False}
    write(out / "summary.json", report)
    async with httpx.AsyncClient(
        base_url=f"http://127.0.0.1:{settings['port']}", trust_env=False, timeout=30
    ) as client:
        while True:
            if process.poll() is not None:
                raise RuntimeError(f"score server exited: {process.returncode}")
            try:
                if (await client.get("/health")).status_code == 200:
                    break
            except httpx.HTTPError:
                pass
            if time.perf_counter() - started > 1200:
                raise TimeoutError("score startup exceeded twenty minutes")
            await asyncio.sleep(2)
        server.update(status="ready", ready_at_unix=time.time())
        from gleipnir.serving_cache_mirror import persist_compiler_mirror

        server["compiler_cache_persistence"] = persist_compiler_mirror(
            server.get("compiler_mirror")
        )
        write(SERVING / "server.json", server)
        report.update(status="running", ready_seconds=time.perf_counter() - started)
        print("score_server_ready", report["ready_seconds"], flush=True)
        response = await client.post(
            "/collective_rpc",
            json={"method": "frost_wrapper_state", "kwargs": {}, "timeout": 60},
        )
        response.raise_for_status()
        assert response.json()["results"][0]["mode"] == "direct"
        if ready_prepare is not None:
            await ready_prepare(client, out)
    values, _ = await trial(workloads["canary"], 4, settings)
    observed = np.array([v["score"] for v in values])
    reference = np.array(control["served"]["adapter"])
    assert len(reference) == len(values) == 20
    mean = float(np.mean(np.abs(observed - reference)))
    correlation = float(np.corrcoef(observed, reference)[0, 1])
    effect = float(np.max(np.abs(observed - np.array(control["served"]["base"]))))
    canary = {
        "mean_absolute_difference": mean,
        "correlation": correlation,
        "adapter_effect": effect,
        "reference_sha256": sha(canary_path),
        "accepted_reference": True,
        "inherited_strict_master_passed": control["passed"],
        "passed": mean <= settings["canary_mean_error_limit"]
        and correlation >= settings["canary_correlation_floor"]
        and effect > 0,
    }
    write(out / "canary.json", canary)
    write(out / "canary_predictions.json", values)
    if not canary["passed"]:
        raise ValueError("score endpoint canary failed")
    print("score_canary_passed", mean, correlation, flush=True)
    passes = [
        (1, "quick", settings["c1_repeats"]),
        (128, "full", settings["c128_repeats"]),
    ]
    if startup_only:
        # Keep a restored reference warm without rerunning timing controls.
        for c, key, _ in passes:
            values, _ = await trial(workloads[key], c, settings)
            write(out / f"c{c}_warmup.json", values)
        report["startup_only"] = True
    for c, key, repeats in [] if startup_only else passes:
        rows = workloads[key]
        warm, seconds = await trial(rows, c, settings)
        write(out / f"c{c}_warmup.json", warm)
        candidate = []
        for index in range(repeats):
            values, seconds = await trial(rows, c, settings)
            candidate.append(values)
            write(out / f"c{c}_repeat{index}.json", values)
            report["trials"].append(
                {
                    "concurrency": c,
                    "repeat": index,
                    **measurement_summary(values, seconds),
                }
            )
            write(out / "summary.json", report)
            print("score_pass", c, index, seconds, flush=True)
        controls = [
            json.loads(p.read_text())
            for p in sorted(baseline.glob(f"c{c}_repeat*.json"))
        ]
        write(
            out / f"c{c}_comparison.json",
            {
                "scores": paired_score_summary(controls, candidate),
                "ranking": ranking_comparison(rows, controls, candidate),
            },
        )
    for p in [
        "monitor_score.json",
        "loaded_precision.json",
        "compile_identity.json",
        "native_attention.json",
        "native_preparation.json",
        "native_gemm_tuning.json",
        "native_attention_projections.json",
        "native_swiglu_output.json",
    ]:
        write(out / p, json.loads((SERVING / p).read_text()))
    report.update(
        status="complete",
        score_audit_passed=json.loads((out / "monitor_score.json").read_text())[
            "passed"
        ],
    )
    write(out / "summary.json", report)
    print(
        "score_restore_complete" if startup_only else "score_benchmark_complete",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    args = parser.parse_args()
    if Path(args.name).name != args.name:
        raise ValueError("run name must be a directory stem")
    try:
        asyncio.run(measure(args.name))
    except BaseException as error:
        write(
            ROOT / "results/b200_monitor_score" / args.name / "failure.json",
            {"error": f"{type(error).__name__}: {error}"},
        )
        raise


if __name__ == "__main__":
    main()
