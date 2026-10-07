"""Run fixed-ID score drift on the selected optimized pooling scorer, then retire it."""

from __future__ import annotations

import argparse
import asyncio
import copy
import hashlib
import inspect
import json
import subprocess
import time
from pathlib import Path

import httpx
import numpy as np

from experiments.b200_inference_benchmark.run import DATA, ROOT, sha, write
from experiments.b200_inference_benchmark.run import environment as base_environment
from experiments.b200_long_context.run import gpu, rpc
from experiments.b200_monitor_score.run import SERVING, trial
from experiments.b200_optimized_id.prepare import (
    BINDING,
    EXPERIMENT,
    align_inputs,
    source_rows,
)
from gleipnir.inference_benchmark import measurement_summary
from gleipnir.serving.reference import selected_score_reference
from gleipnir.serving.score_runtime import resume_score_environment


def materialize(
    tokenizer, rows: list[dict], control: list[dict]
) -> tuple[list[dict], list[dict]]:
    """Reproduce every frozen BF16 rendered input and usage count exactly."""
    import hashlib

    from experiments.tool_trajectory_monitoring.benchmark_qwen_ood import (
        QWEN_NON_THINKING_ASSISTANT_SUFFIX,
        render_margin_prompt,
    )

    align_inputs(rows, control)
    workload, reference = [], []
    for row, old in zip(rows, control, strict=True):
        text = render_margin_prompt(
            tokenizer,
            row["prompt"],
            enable_thinking=False,
            assistant_suffix=QWEN_NON_THINKING_ASSISTANT_SUFFIX,
            decision_prefix="Prediction:",
        )
        digest = hashlib.sha256(text.encode()).hexdigest()
        tokens = len(tokenizer.encode(text, add_special_tokens=False))
        if (
            digest != old["margin_prompt_sha256"]
            or tokens != old["prompt_tokens"]
            or not 0 < tokens < 32768
        ):
            raise ValueError(f"rendered ID/token identity drift: {row['id']}")
        item = {
            "id": row["id"],
            "dataset": old["source"],
            "label": old["label"],
            "prompt": text,
            "prompt_sha256": digest,
            "prompt_tokens": tokens,
            "source_prompt_sha256": old["source_prompt_sha256"],
        }
        workload.append(item)
        reference.append(
            {k: v for k, v in item.items() if k != "prompt"}
            | {"score": old["score"], "margin": old["logit_margin_1_minus_0"]}
        )
    return workload, reference


async def check_canary(settings: dict, out: Path) -> None:
    cached_path = ROOT / settings["cached_canary"]
    cached = json.loads(cached_path.read_text())
    master_path = ROOT / settings["master_canary"]
    master = json.loads(master_path.read_text())
    rows = json.loads((DATA / "canary.json").read_text())
    if (
        not cached["evaluation_passed"]
        or master["master_sha256"] != settings["master_sha256"]
        or [r["prompt_sha256"] for r in rows] != master["prompt_sha256"]
    ):
        raise ValueError("same-adapter canary source/prompt binding changed")
    values, _ = await trial(rows, 4, settings)
    scores = np.array([r["score"] for r in values])
    target = np.array(cached["served"]["adapter"])
    mean = float(np.abs(scores - target).mean())
    corr = float(np.corrcoef(scores, target)[0, 1])
    effect = float(np.abs(scores - np.array(master["base"])).max())
    master_scores = np.array(master["adapter"])
    receipt = {
        "passed": mean <= settings["canary_mean_error_limit"]
        and corr >= settings["canary_correlation_floor"]
        and effect > 0,
        "cached_recipe": {"mae": mean, "correlation": corr, "sha256": sha(cached_path)},
        "master": {
            "mae": float(np.abs(scores - master_scores).mean()),
            "correlation": float(np.corrcoef(scores, master_scores)[0, 1]),
            "sha256": sha(master_path),
        },
        "adapter_effect": effect,
        "inherited_strict_master_passed": cached["passed"],
    }
    write(out / "canary.json", receipt)
    write(out / "canary_predictions.json", values)
    if not receipt["passed"]:
        raise ValueError("selected optimized adapter canary failed")
    print("id_canary_passed", receipt["cached_recipe"], receipt["master"], flush=True)


class ExistingServer:
    """Track the exact independently launched API during a pre-evaluation takeover."""

    def __init__(self, server: dict) -> None:
        self.pid = server["pid"]
        self.command = server["command"]
        if self.poll() is not None:
            raise ValueError("prepared server is no longer live")

    def poll(self) -> int | None:
        path = Path(f"/proc/{self.pid}/cmdline")
        if not path.exists():
            return -1
        actual = path.read_bytes().split(b"\0")[:-1]
        if actual != [v.encode() for v in self.command]:
            return -1
        if Path(f"/proc/{self.pid}/stat").read_text().split(") ", 1)[1][0] == "Z":
            return -1
        return None


async def run(
    name: str, reuse_prepared: str | None = None, config_name: str = "config.json"
) -> None:
    config = EXPERIMENT / config_name
    settings = json.loads(config.read_text())
    admission = settings.get("admission", "grouped")
    if admission not in {"grouped", "continuous"}:
        raise ValueError("unknown client admission")
    out = ROOT / "results/b200_optimized_id" / name
    out.mkdir(parents=True, exist_ok=False)
    write(out / "settings.json", settings)
    binding = json.loads(
        (ROOT / settings.get("binding", str(BINDING.relative_to(ROOT)))).read_text()
    )
    if binding["config_sha256"] != sha(config):
        raise ValueError("frozen experiment settings changed")
    for filename, expected in binding["files"].items():
        if sha(ROOT / filename) != expected:
            raise ValueError(f"frozen ID/reference artifact changed: {filename}")
    write(out / "input_binding.json", binding)
    for p in EXPERIMENT.iterdir():
        if p.suffix in {".py", ".json", ".md"}:
            target = out / "executed_sources" / p.name
            target.parent.mkdir(exist_ok=True)
            target.write_bytes(p.read_bytes())
    report = {
        "status": "preparing",
        "passes": [],
        "promoted": False,
        "checkpoint_selection": binding["selection"],
    }
    write(out / "summary.json", report)
    process = None
    try:
        if reuse_prepared is not None:
            source = ROOT / "results/b200_optimized_id" / reuse_prepared
            previous = json.loads((source / "summary.json").read_text())
            old_settings = json.loads((source / "settings.json").read_text())
            if previous["status"] != "superseded_partial" or previous["passes"]:
                raise ValueError(
                    "takeover requires a retired attempt with no full ID passes"
                )
            if {k: v for k, v in settings.items() if k != "repeats"} != {
                k: v for k, v in old_settings.items() if k != "repeats"
            }:
                raise ValueError("only repetition count can change during takeover")
            old_binding = json.loads((source / "input_binding.json").read_text())
            if old_binding["files"] != binding["files"]:
                raise ValueError("prepared input/reference binding changed")
            rendered = json.loads((source / "rendered_binding.json").read_text())
            for filename in ("workload.json", "reference.json"):
                if sha(source / filename) != rendered[filename[:-5] + "_sha256"]:
                    raise ValueError("prepared rendered workload changed")
            for filename in (
                "workload.json",
                "reference.json",
                "rendered_binding.json",
                "merged_artifact.json",
                "reference_selection.json",
                "retired_parent.json",
                "pooling_boundary.json",
            ):
                (out / filename).write_bytes((source / filename).read_bytes())
            workload = json.loads((out / "workload.json").read_text())
            server = json.loads((SERVING / "server.json").read_text())
            original_server = json.loads((source / "server.json").read_text())
            if (
                server["pid"] != original_server["pid"]
                or server["command"] != original_server["command"]
            ):
                raise ValueError("prepared server identity changed")
            process = ExistingServer(server)
            command = server["command"]
            parent = json.loads((source / "retired_parent.json").read_text())
            base = base_environment()
            base["PYTHONPATH"] = (
                f"/tmp/gleipnir-serving-source-bootstrap:{base['PYTHONPATH']}"
            )
            env = resume_score_environment(ROOT, parent, base)
            started = time.perf_counter()
            write(out / "server.json", server)
            write(out / "initial_gpu.json", gpu())
            write(
                out / "takeover.json",
                {
                    "source": str(source),
                    "reason": "user requested one full pass; partial attempt excluded",
                    "server_restarted": False,
                    "original_config_sha256": old_binding["config_sha256"],
                    "current_config_sha256": binding["config_sha256"],
                    "source_summary_sha256": sha(source / "summary.json"),
                },
            )
        else:
            snapshot = gpu()
            if (
                (SERVING / "server.json").exists()
                or snapshot["apps"]
                or settings["gpu_uuid"] not in snapshot["gpu"]
            ):
                raise ValueError("evaluation requires the verified idle B200")
            write(out / "initial_gpu.json", snapshot)
            selection = selected_score_reference(ROOT)
            write(out / "reference_selection.json", selection)
            recipe = json.loads((ROOT / selection["server_metadata"]).read_text())
            parent = json.loads((ROOT / settings["retired_parent"]).read_text())
            write(out / "retired_parent.json", parent)
            command = recipe["command"].copy()
            model = Path(settings["merged_model"])
            merge = json.loads((model / "merge_manifest.json").read_text())
            if (
                merge["adapter_sha256"] != settings["serving_adapter_sha256"]
                or merge["model"] != settings["model"]
                or merge["revision"] != settings["revision"]
                or sha(ROOT / settings["serving_adapter"])
                != settings["serving_adapter_sha256"]
                or command[command.index("--model") + 1] != str(model)
            ):
                raise ValueError(
                    "optimized merged artifact is not this trained adapter"
                )
            for filename, expected in merge["files_sha256"].items():
                if sha(model / filename) != expected:
                    raise ValueError(f"merged checkpoint file changed: {filename}")
            write(out / "merged_artifact.json", merge)
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
            rows = source_rows(ROOT / settings["input"])
            old = source_rows(ROOT / settings["reference_predictions"])
            workload, reference = materialize(tokenizer, rows, old)
            write(out / "workload.json", workload)
            write(out / "reference.json", reference)
            if len(workload) != settings["rows"]:
                raise ValueError("ID row count changed")
            write(
                out / "rendered_binding.json",
                {
                    "workload_sha256": sha(out / "workload.json"),
                    "reference_sha256": sha(out / "reference.json"),
                    "prompt_tokens": sum(r["prompt_tokens"] for r in workload),
                    "batch_partitions": [
                        [
                            r["id"]
                            for r in workload[offset : offset + settings["batch_rows"]]
                        ]
                        for offset in range(0, len(workload), settings["batch_rows"])
                    ],
                },
            )
            if settings.get("grouped_control") is not None:
                old_rendered = json.loads(
                    (
                        ROOT
                        / settings["grouped_control"]["directory"]
                        / "rendered_binding.json"
                    ).read_text()
                )
                current = json.loads((out / "rendered_binding.json").read_text())
                if old_rendered != current:
                    raise ValueError("continuous workload differs from grouped control")
            base = base_environment()
            base["PYTHONPATH"] = (
                f"/tmp/gleipnir-serving-source-bootstrap:{base['PYTHONPATH']}"
            )
            env = resume_score_environment(ROOT, parent, base)
            env["GLEIPNIR_GIGATOKEN_RECEIPT"] = str(out / "frontend.json")
            from vllm.v1.core.sched import scheduler

            cpu_binding = {
                "scheduler_sha256": sha(Path(inspect.getfile(scheduler))),
                "integration_sha256": sha(
                    ROOT / "experiments/b200_long_context/scheduler.py"
                ),
                "generated_token_reservation": 0,
                "queue_policy": "unchanged_fcfs",
            }
            write(out / "pooling_boundary.json", cpu_binding)
            env["GLEIPNIR_POOLING_BOUNDARY_CONFIG"] = json.dumps(
                cpu_binding, sort_keys=True
            )
            command += [
                "--scheduler-cls",
                "experiments.b200_long_context.scheduler.PoolingContextScheduler",
            ]
            if (
                command[command.index("--max-model-len") + 1] != "32768"
                or command[command.index("--max-num-seqs") + 1] != "128"
                or command[command.index("--max-num-batched-tokens") + 1] != "32768"
                or "--no-enable-prefix-caching" not in command
            ):
                raise ValueError("selected optimized serving contract changed")
            if gpu()["apps"]:
                raise ValueError("GPU acquired by another process before launch")
            server = copy.deepcopy(recipe)
            started = time.perf_counter()
            log = ROOT / "logs/runpod/b200_attention_gdn_serving/server.log"
            with log.open("x") as handle:
                process = subprocess.Popen(
                    command,
                    cwd=ROOT,
                    env=env,
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
            server.update(
                pid=process.pid,
                command=command,
                status="starting",
                started_at_unix=time.time(),
                parent_pid=parent["pid"],
                context_limit=32768,
                pooling_boundary=cpu_binding,
            )
            server["frontend"]["receipt_path"] = str(out / "frontend.json")
            write(SERVING / "server.json", server)
            write(out / "server.json", server)
        async with httpx.AsyncClient(
            base_url=f"http://127.0.0.1:{settings['port']}", trust_env=False, timeout=30
        ) as client:
            while True:
                if process.poll() is not None:
                    raise RuntimeError(
                        f"optimized ID server exited: {process.returncode}"
                    )
                try:
                    if (await client.get("/health")).status_code == 200:
                        break
                except httpx.HTTPError:
                    pass
                if time.perf_counter() - started > 1200:
                    raise TimeoutError("optimized ID server startup timed out")
                await asyncio.sleep(2)
        from gleipnir.serving_cache_mirror import persist_compiler_mirror

        server.update(status="ready", ready_at_unix=time.time())
        server["compiler_cache_persistence"] = persist_compiler_mirror(
            server.get("compiler_mirror")
        )
        write(SERVING / "server.json", server)
        write(out / "server.json", server)
        report["startup_seconds"] = time.perf_counter() - started
        print(
            "optimized_id_server_ready",
            process.pid,
            report["startup_seconds"],
            flush=True,
        )
        if (await rpc("frost_wrapper_state"))["mode"] != "direct":
            raise ValueError("direct FROST host binding changed")
        await check_canary(settings, out)
        warm, seconds = await trial(
            json.loads((DATA / "quick.json").read_text()),
            settings["concurrency"],
            settings,
        )
        write(out / "training_warmup.json", {"seconds": seconds, "values": warm})
        report["status"] = "running"
        report["admission"] = admission
        write(out / "summary.json", report)
        for repeat in range(settings["repeats"]):
            if admission == "continuous":
                from gleipnir.serving import admission as admission_module

                (out / "client_source.py").write_bytes(
                    Path(admission_module.__file__).read_bytes()
                )
                last_progress = 0

                def progress(
                    completed: int, total: int, repeat_index: int = repeat
                ) -> None:
                    nonlocal last_progress
                    if (
                        completed - last_progress >= settings["batch_rows"]
                        or completed == total
                    ):
                        print("id_progress", repeat_index, completed, total, flush=True)
                        last_progress = completed

                (
                    values,
                    seconds,
                    persistence,
                ) = await admission_module.continuous_score_trial(
                    workload,
                    settings["concurrency"],
                    settings,
                    checkpoint=out / f"repeat{repeat}/responses.jsonl",
                    contract={
                        "config_sha256": sha(config),
                        "input_binding_sha256": sha(out / "input_binding.json"),
                        "merged_artifact_sha256": sha(out / "merged_artifact.json"),
                        "server_command_sha256": hashlib.sha256(
                            json.dumps(command, sort_keys=True).encode()
                        ).hexdigest(),
                    },
                    progress=progress,
                )
                if persistence["resumed_rows"]:
                    raise ValueError(
                        "resumed segment cannot count as a complete timed pass"
                    )
                exported_at = time.perf_counter()
                write(out / f"repeat{repeat}.json", values)
                export_seconds = time.perf_counter() - exported_at
                record = {
                    "repeat": repeat,
                    "admission": admission,
                    "first_id_shape_use_inclusive": repeat == 0,
                    "checkpoint": persistence,
                    "prediction_export_seconds": export_seconds,
                    **measurement_summary(values, seconds + export_seconds),
                }
            else:
                values, http_seconds, batch_records = [], 0.0, []
                started = time.perf_counter()
                for offset in range(0, len(workload), settings["batch_rows"]):
                    batch = workload[offset : offset + settings["batch_rows"]]
                    observed, elapsed = await trial(
                        batch, settings["concurrency"], settings
                    )
                    values.extend(observed)
                    http_seconds += elapsed
                    batch_index = offset // settings["batch_rows"]
                    batch_name = f"repeat{repeat}/batch{batch_index:03d}.json"
                    saved_at = time.perf_counter()
                    write(out / batch_name, observed)
                    save_seconds = time.perf_counter() - saved_at
                    lengths = [r["prompt_tokens"] for r in batch]
                    latencies = [r["latency_seconds"] for r in observed]
                    batch_records.append(
                        {
                            "batch": offset // settings["batch_rows"],
                            "rows": len(batch),
                            "prompt_tokens": sum(lengths),
                            "http_seconds": elapsed,
                            "save_seconds": save_seconds,
                            "prompt_token_min": min(lengths),
                            "prompt_token_max": max(lengths),
                            "request_latency_p50_seconds": float(
                                np.percentile(latencies, 50)
                            ),
                            "request_latency_p90_seconds": float(
                                np.percentile(latencies, 90)
                            ),
                            "request_latency_max_seconds": max(latencies),
                        }
                    )
                    print("id_progress", repeat, len(values), len(workload), flush=True)
                seconds = time.perf_counter() - started
                write(out / f"repeat{repeat}.json", values)
                record = {
                    "repeat": repeat,
                    "first_id_shape_use_inclusive": repeat == 0,
                    "http_seconds": http_seconds,
                    "batch_records": batch_records,
                    **measurement_summary(values, seconds),
                }
            report["passes"].append(record)
            write(out / "summary.json", report)
            print(
                "id_pass_complete",
                repeat,
                seconds,
                record["prompt_tokens_per_second"],
                flush=True,
            )
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
        frontend_path = Path(server["frontend"]["receipt_path"])
        if frontend_path.exists():
            (out / "frontend.json").write_bytes(frontend_path.read_bytes())
        report["status"] = "complete"
        write(out / "summary.json", report)
    except BaseException as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        write(out / "summary.json", report)
        raise
    finally:
        if process is not None and (SERVING / "server.json").exists():
            live = json.loads((SERVING / "server.json").read_text())
            if live["pid"] == process.pid and process.poll() is None:
                subprocess.run(
                    [
                        command[0],
                        "-m",
                        "experiments.b200_attention_gdn_serving.stop_server",
                        "--archive-name",
                        name + "_optimized_id",
                        "--reason",
                        "User-requested fixed-ID score-drift evaluation finished",
                    ],
                    cwd=ROOT,
                    env=env,
                    check=True,
                )
                retirement = SERVING / (name + "_optimized_id_server_retired.json")
                write(out / "retirement.json", json.loads(retirement.read_text()))
        write(out / "closure.json", gpu())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--reuse-prepared")
    args = parser.parse_args()
    if args.name in {"", ".", ".."} or Path(args.name).name != args.name:
        raise ValueError("run name must be a stem")
    if args.reuse_prepared is not None and (
        args.reuse_prepared in {"", ".", ".."}
        or Path(args.reuse_prepared).name != args.reuse_prepared
    ):
        raise ValueError("prepared run name must be a stem")
    if Path(args.config).name != args.config or not args.config.endswith(".json"):
        raise ValueError("config must be an experiment JSON filename")
    asyncio.run(run(args.name, args.reuse_prepared, args.config))


if __name__ == "__main__":
    main()
