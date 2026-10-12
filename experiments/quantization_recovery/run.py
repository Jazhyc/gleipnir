"""Resumable QAD stages on the already authorized B200; never pod lifecycle."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import signal
import subprocess
import time
from pathlib import Path

import httpx
import yaml

from gleipnir.data.monitoring import file_hash, read_rows, write_json, write_rows
from gleipnir.evaluation.http_score import score_batch

ROOT = Path.cwd()
CONFIG = ROOT / "experiments/quantization_recovery/config.yaml"


def check(config):
    for name, spec in config["inputs"].items():
        if file_hash(ROOT / spec["path"]) != spec["sha256"]:
            raise ValueError(f"input drift: {name}")
    if (
        file_hash(ROOT / config["initial_adapter"] / "adapter_model.safetensors")
        != config["initial_adapter_sha256"]
    ):
        raise ValueError("initial FP32 adapter drift")
    gpu = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"], text=True
    ).strip()
    if gpu != config["gpu_uuid"]:
        raise ValueError("B200 UUID drift")


def prepare(config):
    from transformers import AutoTokenizer

    from experiments.tool_trajectory_monitoring.benchmark_qwen_ood import (
        QWEN_NON_THINKING_ASSISTANT_SUFFIX,
        render_margin_prompt,
    )

    directory = ROOT / "data" / config["experiment"]
    directory.mkdir(parents=True, exist_ok=True)
    manifest = directory / "workload_manifest.json"
    if manifest.exists():
        previous = json.loads(manifest.read_text())
        if previous["config_sha256"] != file_hash(CONFIG) or previous[
            "workload_sha256"
        ] != file_hash(directory / "workload.jsonl"):
            raise ValueError("teacher workload cache drift")
        return
    tokenizer = AutoTokenizer.from_pretrained(ROOT / config["base_model"])
    rows = read_rows(ROOT / config["inputs"]["training"]["path"])
    workload = []
    import hashlib

    for row in rows:
        prompt = render_margin_prompt(
            tokenizer,
            row["student_prompt"],
            enable_thinking=False,
            assistant_suffix=QWEN_NON_THINKING_ASSISTANT_SUFFIX,
            decision_prefix="Prediction:",
        )
        ids = tokenizer.encode(prompt, add_special_tokens=False)
        if len(ids) > config["max_length"]:
            raise ValueError("training teacher prompt would need truncation")
        workload.append(
            {
                "id": row["index"],
                "prompt": prompt,
                "input_ids": ids,
                "prompt_tokens": len(ids),
                "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                "dataset": row["dataset"],
                "label": row["label"],
                "source_dataset": row["source_dataset"],
            }
        )
    if (
        len(workload) != config["training_rows"]
        or sum(r["prompt_tokens"] for r in workload) != config["training_tokens"]
    ):
        raise ValueError("frozen training population/token count drift")
    write_rows(directory / "workload.jsonl", workload)
    write_json(
        manifest,
        {
            "config_sha256": file_hash(CONFIG),
            "training_sha256": config["inputs"]["training"]["sha256"],
            "workload_sha256": file_hash(directory / "workload.jsonl"),
            "rows": len(workload),
            "tokens": sum(r["prompt_tokens"] for r in workload),
            "rendering": "canonical_non_thinking_Prediction_prefix",
        },
    )


def server_command(config, output, *, bf16, model):
    from experiments.b200_vllm031.runtime import candidate_environment
    from gleipnir.serving.reference import selected_serving_default

    selection, command = selected_serving_default(ROOT)
    command[0] = config["serving_python"]
    command[command.index("--model") + 1] = str(model)
    data = json.loads(command[command.index("--additional-config") + 1])
    if bf16:
        from gleipnir.serving.bf16_monitor import command as bf16_command

        command = bf16_command(command, data["gleipnir_frost_fp4"])
    else:
        command[command.index("--worker-cls") + 1] = (
            "experiments.filtered_fp4_baseline.compiler_repair_worker.TypedCoordinateWorker"
        )
        data["serving_condition"].update(
            mxfp8_validation="results/filtered_fp4_compiler_repair/native02.json",
            mxfp8_coordinate_repair="typed_int32",
            merged_model=str(model),
        )
        from experiments.b200_vllm031.source_bindings import restore_diagnostic_sources

        archive = ROOT / "results/b200_vllm031/pre_migration_sources.tar.gz"
        restored = restore_diagnostic_sources(ROOT, data["serving_condition"], archive)
        write_json(
            output / "source_restoration.json",
            {
                "archive_sha256": file_hash(archive),
                "restored": restored,
            },
        )
        for path in [
            "experiments/filtered_fp4_baseline/compiler_repair.py",
            "experiments/filtered_fp4_baseline/compiler_repair_worker.py",
            "src/gleipnir/serving/mxfp8_compile_compat.py",
        ]:
            data["gleipnir_frost_fp4"][path] = file_hash(ROOT / path)
        command[command.index("--additional-config") + 1] = json.dumps(
            data, sort_keys=True
        )
    env = candidate_environment(ROOT)
    host = json.loads((ROOT / selection["host_parent"]).read_text())
    env.update(
        GLEIPNIR_GIGATOKEN_RECEIPT=str(output / "frontend.json"),
        VLLM_GDN_DECODE_KERNEL="cuda",
        GLEIPNIR_FLASHINFER_GDN_CP="auto",
        GLEIPNIR_VLLM031_SCHEDULER_BINDING=json.dumps(selection["scheduler_binding"]),
    )
    if bf16:
        env.pop("GLEIPNIR_FROST_WRAPPER_VALIDATION", None)
        env["GLEIPNIR_BF16_AUDIT"] = str(output / "bf16_audit.json")
    else:
        env["GLEIPNIR_FROST_WRAPPER_VALIDATION"] = host["host_wrapper"]["validation"]
    return command, env


async def wait_server(process, config):
    before = time.monotonic()
    async with httpx.AsyncClient(trust_env=False, timeout=10) as client:
        while True:
            if process.poll() is not None:
                raise RuntimeError(f"server exited: {process.returncode}")
            try:
                response = await client.get(
                    f"http://127.0.0.1:{config['evaluation']['port']}/health"
                )
                if response.status_code == 200:
                    return
            except httpx.HTTPError:
                pass
            if time.monotonic() - before > 900:
                raise TimeoutError("server startup")
            await asyncio.sleep(2)


def retire(process):
    if process.poll() is None:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()


def teacher(config, output):
    directory = ROOT / "data" / config["experiment"]
    manifest = json.loads((directory / "workload_manifest.json").read_text())
    # The old runner binds the exact teacher checkpoint shards and completed
    # master, export and merge. Its failed evaluation is not reused as a pass.
    runner = json.loads(
        (ROOT / "results/filtered-fp4-fa4-baseline01/runner.json").read_text()
    )
    for stage in ["train", "master-reference", "merge", "merged-reference"]:
        for path, expected in runner["stages"][stage]["files_sha256"].items():
            if file_hash(Path(path)) != expected:
                raise ValueError(f"teacher identity drift: {path}")
    identity = {
        "teacher_master_sha256": config["initial_adapter_sha256"],
        "workload_sha256": manifest["workload_sha256"],
        "temperature": config["temperature"],
        "merged_files_sha256": runner["stages"]["merge"]["files_sha256"],
    }
    identity_path = directory / "teacher_identity.json"
    if identity_path.exists() and json.loads(identity_path.read_text()) != identity:
        raise ValueError("BF16 teacher cache identity drift")
    write_json(identity_path, identity)
    complete = directory / "teacher_complete.json"
    if complete.exists():
        receipt = json.loads(complete.read_text())
        if receipt["teacher_sha256"] != file_hash(directory / "teacher.jsonl"):
            raise ValueError("completed teacher cache drift")
        return
    rows = read_rows(directory / "workload.jsonl")
    command, env = server_command(
        config, output, bf16=True, model=config["teacher_merge"]
    )
    write_json(
        output / "teacher_command.json", {"command": command, "identity": identity}
    )
    with (output / "teacher_server.log").open("a") as log:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    write_json(
        output / "teacher_launch.json",
        {"pid": process.pid, "started_at_unix": time.time()},
    )

    async def score():
        await wait_server(process, config)
        targets = []
        for start in range(0, len(rows), 128):
            batch = rows[start : start + 128]
            path = directory / "teacher_batches" / f"{start:05d}.json"
            if path.exists():
                saved = json.loads(path.read_text())
                values = saved["predictions"]
                if saved["identity"] != identity:
                    raise ValueError("cached teacher batch identity drift")
            else:
                values, seconds = await score_batch(batch, config["evaluation"])
                write_json(
                    path,
                    {
                        "identity": identity,
                        "predictions": values,
                        "seconds": seconds,
                        "received_at_unix": time.time(),
                    },
                )
            for row, value in zip(batch, values, strict=True):
                if (
                    row["id"] != value["id"]
                    or row["prompt_sha256"] != value["prompt_sha256"]
                ):
                    raise ValueError("teacher row/prompt mismatch")
                targets.append(
                    {
                        "index": row["id"],
                        "dataset": row["dataset"],
                        "label": row["label"],
                        "soft_target": value["score"],
                        "target_probs": {
                            "negative": 1 - value["score"],
                            "positive": value["score"],
                        },
                        "raw_logits": value["logits"],
                        "rendered_prompt_sha256": row["prompt_sha256"],
                        "teacher_model": "completed_strict_filtered_Qwen3.5-4B_BF16",
                        "teacher_master_sha256": config["initial_adapter_sha256"],
                    }
                )
            print("teacher_rows", len(targets), len(rows), flush=True)
        write_rows(directory / "teacher.jsonl", targets)
        audit = json.loads((output / "bf16_audit.json").read_text())
        if not audit["passed"]:
            raise ValueError("BF16 teacher precision/dispatch audit failed")
        write_json(
            complete,
            {
                "rows": len(targets),
                "teacher_sha256": file_hash(directory / "teacher.jsonl"),
                "identity": identity,
                "temperature": 1.0,
            },
        )

    try:
        asyncio.run(score())
    finally:
        retire(process)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage", choices=["teacher", "preflight", "all"], default="all"
    )
    args = parser.parse_args()
    config = yaml.safe_load(CONFIG.read_text())
    output = ROOT / "results" / config["experiment"]
    output.mkdir(parents=True, exist_ok=True)
    check(config)
    prepare(config)
    source_paths = [
        *sorted((ROOT / "experiments/quantization_recovery").glob("*.py")),
        CONFIG,
        ROOT / "experiments/quantization_recovery/README.md",
        ROOT / "src/gleipnir/training/quantization_recovery.py",
        ROOT / "src/gleipnir/kernels/fp4/decode.py",
    ]
    bindings = {str(p.relative_to(ROOT)): file_hash(p) for p in source_paths}
    contract = {"config": config, "source_bindings": bindings}
    if args.stage == "all":
        contract_path = output / "execution_contract.json"
        if contract_path.exists() and json.loads(contract_path.read_text()) != contract:
            raise ValueError("frozen QAD execution contract drift")
        write_json(contract_path, contract)

    def check_sources():
        for path, digest in bindings.items():
            if file_hash(ROOT / path) != digest:
                raise ValueError(f"QAD source drift: {path}")

    try:
        if args.stage in {"teacher", "all"}:
            teacher(config, output)
            check_sources()
        if args.stage in {"preflight", "all"}:
            from gleipnir.campaigns.runtime import training_environment

            env = training_environment(ROOT, config["experiment"], native_fp4_mlp=True)
            command = [
                config["training_python"],
                "-m",
                "experiments.quantization_recovery.train",
            ]
            if args.stage == "preflight":
                command.append("--preflight-only")
            write_json(output / "training_command.json", {"command": command})
            subprocess.run(command, cwd=ROOT, env=env, check=True)
            check_sources()
            if args.stage == "all":
                subprocess.run(
                    [
                        config["training_python"],
                        "-m",
                        "experiments.quantization_recovery.reference",
                    ],
                    cwd=ROOT,
                    env=env,
                    check=True,
                )
                import shutil

                from gleipnir.adapters.merge import merge_checkpoint
                from gleipnir.adapters.rebase import rebase_adapter

                complete = json.loads((output / "training_complete.json").read_text())
                master_weights = output / "causal_adapter/adapter_model.safetensors"
                if file_hash(master_weights) != complete["master_sha256"]:
                    raise ValueError("completed QAD master drift before rebase")
                rebased = rebase_adapter(
                    output / "causal_adapter", output / "merged_adapter"
                )

                # Replace only this explicitly owned, reconstructable ephemeral
                # slot after all teacher logits and original merge receipts exist.
                destination = Path(config["teacher_merge"])
                old_manifest = destination / "merge_manifest.json"
                old = json.loads(old_manifest.read_text())
                old_adapter_sha = (
                    "be3df0c241ac8daa551e77519ebb4c4d725cc8ebd7673ea43f2c9e9b7acb9b16"
                )
                if old["adapter_sha256"] != old_adapter_sha:
                    raise ValueError("ephemeral merge ownership/identity changed")
                if (
                    destination.is_symlink()
                    or str(destination)
                    != "/tmp/gleipnir-merged/filtered-fp4-fa4-baseline"
                ):
                    raise ValueError("unsafe merged checkpoint replacement")
                write_json(output / "replaced_merge_manifest.json", old)
                for name, digest in old["files_sha256"].items():
                    if file_hash(destination / name) != digest:
                        raise ValueError("predecessor merged checkpoint drift")
                shutil.rmtree(destination)
                report = merge_checkpoint(
                    ROOT / config["base_model"],
                    output / "merged_adapter",
                    destination,
                    expected_adapter_sha256=rebased["destination_sha256"],
                    model_id="Qwen/Qwen3.5-4B",
                    revision="851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
                )
                write_json(output / "merged_artifact.json", report)
                serving_command = [
                    config["serving_python"],
                    "-m",
                    "experiments.quantization_recovery.evaluate",
                ]
                from experiments.b200_vllm031.runtime import candidate_environment

                subprocess.run(
                    serving_command,
                    cwd=ROOT,
                    env=candidate_environment(ROOT),
                    check=True,
                )
        write_json(
            output / f"{args.stage}_status.json",
            {"status": "complete", "finished_at_unix": time.time()},
        )
    except BaseException as error:
        write_json(
            output / f"{args.stage}_status.json",
            {"status": "failed", "error": str(error), "finished_at_unix": time.time()},
        )
        raise


if __name__ == "__main__":
    main()
