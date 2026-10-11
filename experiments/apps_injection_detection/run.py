"""Prepare and execute the matched BF16 APPS injection-presence comparison."""

from __future__ import annotations

import argparse
import asyncio
import inspect
import json
import os
import shutil
import subprocess
import sys
import time
from importlib.metadata import version
from pathlib import Path

import httpx

from experiments.b200_vllm031.runtime import candidate_environment
from experiments.tool_trajectory_monitoring.prompting import load_prompt_set
from gleipnir.data.monitoring import (
    digest,
    file_hash,
    read_rows,
    write_json,
    write_rows,
)
from gleipnir.evaluation.concept_census import render_prompt
from gleipnir.evaluation.http_score import score_batch
from gleipnir.evaluation.injection_presence import (
    injection_label,
    paired_bootstrap,
    summarize,
    trajectory,
)
from gleipnir.serving.monitor_score import validate_score_response

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
CONFIG = HERE / "config.json"
ACTIVE = ROOT / "results/b200_attention_gdn_serving/server.json"


def checked_config() -> dict:
    """Bind source inputs, prompt and preserved master/merge receipts."""
    config = json.loads(CONFIG.read_text())
    for name, sha in config["files_sha256"].items():
        if file_hash(ROOT / name) != sha:
            raise ValueError(f"frozen input drift: {name}")
    return config


def prepare(config: dict, out: Path) -> None:
    """Render the same complete detection prompts for both model checkpoints."""
    from transformers import AutoTokenizer

    out.mkdir(parents=True, exist_ok=False)
    tokenizer = AutoTokenizer.from_pretrained(
        ROOT / config["tokenizer"], local_files_only=True
    )
    instruction = (
        ROOT / "experiments/training_injection_census/prompt.txt"
    ).read_text()
    workload = []
    source_template = load_prompt_set().student
    for split in ("benchmark", "honest_controls"):
        for row in read_rows(
            ROOT / f"data/monitoring_injection_disentanglement/{split}/neutral.jsonl"
        ):
            text = trajectory(row["prompt"], source_template=source_template)
            prompt = render_prompt(tokenizer, instruction, text)
            tokens = len(tokenizer.encode(prompt, add_special_tokens=False))
            if not 0 < tokens < 32768:
                raise ValueError("APPS detection prompt would truncate")
            metadata = row["metadata"]
            workload.append(
                {
                    "id": row["id"],
                    "prompt": prompt,
                    "prompt_sha256": digest(prompt),
                    "prompt_tokens": tokens,
                    "trajectory_sha256": digest(text),
                    "split": split,
                    "ground_truth": injection_label(metadata),
                    "behavior_label": metadata["ground_truth"],
                    "condition": metadata["condition"],
                    "task_id": metadata["task_id"],
                    "original_metadata": metadata,
                }
            )
    if len(workload) != config["expected_rows"] or len(
        {r["id"] for r in workload}
    ) != len(workload):
        raise ValueError("APPS detection coverage drift")
    write_rows(out / "workload.jsonl", workload)
    sources = {
        str(p.relative_to(ROOT)): file_hash(p)
        for p in [
            CONFIG,
            HERE / "run.py",
            HERE / "README.md",
            ROOT / "src/gleipnir/evaluation/injection_presence.py",
            ROOT / "src/gleipnir/evaluation/concept_census.py",
            ROOT / "src/gleipnir/evaluation/http_score.py",
        ]
    }
    for name in sources:
        target = out / "executed_sources" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, target)
    write_json(
        out / "contract.json",
        {
            "config": config,
            "sources": sources,
            "workload_sha256": file_hash(out / "workload.jsonl"),
            "rows": len(workload),
            "tokens": sum(r["prompt_tokens"] for r in workload),
            "maximum_tokens": max(r["prompt_tokens"] for r in workload),
            "positive_rows": sum(r["ground_truth"] for r in workload),
        },
    )
    print("presence_prepared", len(workload), flush=True)


def check_contract(out: Path) -> dict:
    contract = json.loads((out / "contract.json").read_text())
    if file_hash(out / "workload.jsonl") != contract["workload_sha256"]:
        raise ValueError("workload identity drift")
    for name, sha in contract["sources"].items():
        if file_hash(ROOT / name) != sha:
            raise ValueError(f"executed source drift: {name}")
    return contract


async def start(config: dict, out: Path, role: str) -> dict:
    """Reuse the archived BF16 command and prove fresh native/master agreement."""
    from vllm.v1.core.sched import scheduler

    from gleipnir.campaigns.monitoring.evaluation import agreement

    if {k: version(k) for k in config["runtime"]} != config["runtime"]:
        raise ValueError("serving runtime drift")
    gpu = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=uuid,driver_version", "--format=csv,noheader"],
        text=True,
    ).strip()
    apps = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True
    ).strip()
    if gpu != f"{config['gpu_uuid']}, {config['driver']}" or apps:
        raise ValueError(f"unexpected GPU or active workload: {gpu}; {apps}")
    model = str(ROOT / config["base_model"]) if role == "base" else config["sdpa_model"]
    merged = json.loads(
        (ROOT / "results/b200-augmented-sdpa-replay02/merged_artifact.json").read_text()
    )
    if not json.loads(
        (ROOT / "results/b200-augmented-sdpa-replay02/merged_parity.json").read_text()
    )["passed"]:
        raise ValueError("missing FP32-master merge parity")
    hashes = config["base_files_sha256"] if role == "base" else merged["files_sha256"]
    for name, sha in hashes.items():
        if file_hash(Path(model) / name) != sha:
            raise ValueError(f"model file drift: {name}")
    cell = out / role
    cell.mkdir(parents=True, exist_ok=True)
    parent = json.loads(
        (ROOT / "results/b200-augmented-sdpa-replay02/server.json").read_text()
    )
    command = list(parent["command"])
    command[0] = sys.executable
    command[command.index("--model") + 1] = model
    index = command.index("--additional-config") + 1
    additional = json.loads(command[index])
    condition = additional["serving_condition"]
    if condition["quantization"] is not None or any(
        condition[k] != "bf16"
        for k in (
            "attention_precision",
            "attention_projection_precision",
            "gdn_projection_precision",
            "mlp_precision",
        )
    ):
        raise ValueError("reference command is not plain BF16")
    condition["merged_model"] = model
    additional["gleipnir_frost_fp4"] = {
        p: file_hash(ROOT / p)
        for p in additional["gleipnir_frost_fp4"]
        if (ROOT / p).is_file()
    }
    command[index] = json.dumps(additional, sort_keys=True)
    env = candidate_environment(ROOT)
    env["PYTHONPATH"] += ":/tmp/gleipnir-gigatoken-0.10.0"
    env.update(
        VLLM_GDN_DECODE_KERNEL="cuda",
        GLEIPNIR_FLASHINFER_GDN_CP="auto",
        GLEIPNIR_BF16_AUDIT=str(cell / "bf16_audit.json"),
        GLEIPNIR_GIGATOKEN_RECEIPT=str(cell / "frontend.json"),
        GLEIPNIR_VLLM031_SCHEDULER_BINDING=json.dumps(
            {
                "upstream_sha256": file_hash(Path(inspect.getfile(scheduler))),
                "integration_sha256": file_hash(
                    ROOT / "experiments/b200_vllm031/scheduler.py"
                ),
            }
        ),
    )
    env.pop("GLEIPNIR_FROST_WRAPPER_VALIDATION", None)
    log = ROOT / f"logs/runpod/apps_injection_detection/{role}_server.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a") as handle:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            env=env,
            stdout=handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    server = parent | {
        "pid": process.pid,
        "command": command,
        "model_role": role,
        "status": "starting",
        "log": str(log),
        "launched_at_unix": time.time(),
    }
    if role == "base":
        server.pop("adapter_sha256", None)
    # Never signal stale PIDs from a terminated pod.
    if ACTIVE.exists():
        shutil.copyfile(ACTIVE, cell / "previous_inactive_server.json")
    write_json(ACTIVE, server)
    write_json(cell / "server.json", server)
    try:
        started = time.monotonic()
        async with httpx.AsyncClient(trust_env=False, timeout=10) as client:
            while True:
                if process.poll() is not None:
                    raise RuntimeError(f"{role} server exited: {process.returncode}")
                try:
                    if (
                        await client.get(f"http://127.0.0.1:{config['port']}/health")
                    ).status_code == 200:
                        break
                except httpx.HTTPError:
                    pass
                if time.monotonic() - started > 1200:
                    raise TimeoutError("BF16 startup exceeded twenty minutes")
                await asyncio.sleep(2)
        server.update(status="ready", ready_at_unix=time.time())
        write_json(ACTIVE, server)
        write_json(cell / "server.json", server)
        canary = read_rows(
            ROOT / "results/b200-augmented-sdpa-replay02/canary_workload.jsonl"
        )
        reference = json.loads(
            (
                ROOT
                / "results/b200-augmented-sdpa-replay02/4b/monitor"
                / "parity_reference.json"
            ).read_text()
        )
        if [r["prompt_sha256"] for r in canary] != reference["prompt_sha256"]:
            raise ValueError("master canary prompt identity drift")
        values, _ = await score_batch(canary, config | {"concurrency": 4})
        gate = agreement(
            [v["score"] for v in values],
            reference["base" if role == "base" else "adapter"],
            reference["base"] if role == "sdpa" else reference["adapter"],
            {
                "max_mean_absolute_difference": 0.02,
                "min_correlation": 0.99,
                "min_adapter_effect": 1e-6,
            },
        )
        write_json(cell / "canary_predictions.json", values)
        write_json(cell / "parity.json", gate)
        native = json.loads((cell / "bf16_audit.json").read_text())
        if (
            not gate["passed"]
            or not native["passed"]
            or len(native["attention_calls"]) < 8
        ):
            raise ValueError("BF16 master/native startup gate failed")
        write_json(
            cell / "binding.json",
            {
                "model": model,
                "files_sha256": hashes,
                "runtime": config["runtime"],
                "gpu": gpu,
                "contract_sha256": file_hash(out / "contract.json"),
            },
        )
        print("presence_server_passed", role, flush=True)
        return server
    except BaseException:
        if process.poll() is None:
            os.killpg(process.pid, 15)
        raise


async def evaluate(config: dict, out: Path) -> None:
    """Sequential model evaluation; retain identity-bound, resumable batches."""
    check_contract(out)
    workload = read_rows(out / "workload.jsonl")
    predictions = {}
    for role in ("base", "sdpa"):
        cell = out / role
        if (cell / "complete.json").exists():
            done = json.loads((cell / "complete.json").read_text())
            if done["contract_sha256"] != file_hash(out / "contract.json") or done[
                "predictions_sha256"
            ] != file_hash(cell / "predictions.jsonl"):
                raise ValueError("completed prediction drift")
            predictions[role] = read_rows(cell / "predictions.jsonl")
            continue
        server = await start(config, out, role)
        values, seconds = [], []
        binding_sha = file_hash(cell / "binding.json")
        for index in range(0, len(workload), config["batch_rows"]):
            batch = workload[index : index + config["batch_rows"]]
            path = cell / "batches" / f"{index:05d}.json"
            if path.exists():
                receipt = json.loads(path.read_text())
                if receipt["binding_sha256"] != binding_sha:
                    raise ValueError("resumable batch contract drift")
                scored, elapsed = receipt["predictions"], receipt["seconds"]
            else:
                scored, elapsed = await score_batch(batch, config)
                write_json(
                    path,
                    {
                        "binding_sha256": binding_sha,
                        "predictions": scored,
                        "seconds": elapsed,
                        "scored_at_unix": time.time(),
                    },
                )
            if [(r["id"], r["prompt_sha256"], r["prompt_tokens"]) for r in scored] != [
                (r["id"], r["prompt_sha256"], r["prompt_tokens"]) for r in batch
            ]:
                raise ValueError("scored prompt identity drift")
            for row, score in zip(batch, scored, strict=True):
                validate_score_response(score, row["prompt_tokens"])
            values.extend(
                {k: v for k, v in row.items() if k != "prompt"} | score
                for row, score in zip(batch, scored, strict=True)
            )
            seconds.append(elapsed)
            write_json(
                out / "status.json",
                {
                    "stage": "scoring",
                    "role": role,
                    "rows": len(values),
                    "total": len(workload),
                },
            )
            print("presence_progress", role, len(values), len(workload), flush=True)
        check_contract(out)
        actual = [
            v.decode()
            for v in Path(f"/proc/{server['pid']}/cmdline").read_bytes().split(b"\0")
            if v
        ]
        if actual != server["command"]:
            raise ValueError("server process identity drift")
        write_rows(cell / "predictions.jsonl", values)
        write_json(cell / "summary.json", summarize(values))
        tokens = sum(r["prompt_tokens"] for r in values)
        write_json(
            cell / "timing.json",
            {
                "seconds": sum(seconds),
                "rows": len(values),
                "prompt_tokens": tokens,
                "prompt_tokens_per_second": tokens / sum(seconds),
                "requests_per_second": len(values) / sum(seconds),
            },
        )
        write_json(
            cell / "complete.json",
            {
                "rows": len(values),
                "predictions_sha256": file_hash(cell / "predictions.jsonl"),
                "contract_sha256": file_hash(out / "contract.json"),
                "parity_passed": True,
            },
        )
        predictions[role] = values
        subprocess.run(
            [
                sys.executable,
                "-m",
                "experiments.b200_vllm031.stop",
                "--archive-name",
                f"apps_presence_{role}_{time.time_ns()}",
            ],
            check=True,
            cwd=ROOT,
        )
    write_json(
        out / "comparison.json",
        {
            "base": summarize(predictions["base"]),
            "sdpa": summarize(predictions["sdpa"]),
            "paired_task_bootstrap": paired_bootstrap(
                predictions["base"], predictions["sdpa"]
            ),
        },
    )
    write_json(
        out / "status.json", {"stage": "complete", "rows_per_model": len(workload)}
    )
    print("presence_complete", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("prepare", "evaluate"), required=True)
    args = parser.parse_args()
    config = checked_config()
    out = ROOT / "results/apps_injection_detection" / config["name"]
    try:
        if args.stage == "prepare":
            prepare(config, out)
        else:
            asyncio.run(evaluate(config, out))
    except BaseException as error:
        write_json(
            out / f"failure_{time.time_ns()}.json",
            {"type": type(error).__name__, "message": str(error)},
        )
        raise


if __name__ == "__main__":
    main()
