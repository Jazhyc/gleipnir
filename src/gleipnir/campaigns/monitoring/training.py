"""Validated B200 training, FP32 export and BF16 merge stages."""

from __future__ import annotations

import json
import math
import os
import subprocess
import time
from pathlib import Path

from gleipnir.adapters.merge import merge_checkpoint
from gleipnir.adapters.rebase import rebase_adapter
from gleipnir.campaigns.training_command import training_command
from gleipnir.data.monitoring import file_hash, write_json
from gleipnir.evaluation.campaign import EvaluationContext, reference
from gleipnir.training.validation import validate_training_metadata

from .contract import Campaign


def validate_completion(metadata: dict, ctx: Campaign) -> None:
    config, job = ctx.config, ctx.job()
    validate_training_metadata(
        metadata,
        job,
        config["model"]["initial_tensor_sha256"],
        expected_steps=config["expected_steps"],
        profile=ctx.profile(),
    )
    adaptive = metadata["adaptive_microbatching"]
    batch = config["logical_batch_size"]
    sizes = [
        min(batch, config["training_rows"] - start)
        for start in range(0, config["training_rows"], batch)
    ]
    records = adaptive["records"]
    if (
        adaptive["logical_batch_sizes"] != sizes
        or metadata["train_metrics"]["epoch"] != 1.0
        or sum(r["tokens"] for r in records) != config["expected_training_tokens"]
        or any(r["tokens"] != r["padded_tokens"] for r in records)
    ):
        raise ValueError("epoch/token/packing coverage failed")
    if {r["update"] for r in records} != set(range(1, len(sizes) + 1)):
        raise ValueError("physical records contain missing or extra updates")
    for update, size in enumerate(sizes, 1):
        if sorted(
            i for r in records if r["update"] == update for i in r["logical_indices"]
        ) != list(range(size)):
            raise ValueError("missing or repeated logical training row")
    durations = metadata["optimizer_step_timing"]["durations_seconds"]
    if len(durations) != len(sizes) or not all(
        math.isfinite(t) and t > 0 for t in durations
    ):
        raise ValueError("nonfinite or incomplete training timings")


def train(ctx: Campaign) -> None:
    """Invoke the ordinary current Trainer, preserving its one-epoch contract."""
    config = ctx.config
    import torch
    from safetensors import safe_open

    from gleipnir.training.backends.flashqla import load_flashqla
    from gleipnir.training.backends.native_fp4 import (
        validate_kernel_sources,
        verify_native_fp4_runtime,
    )

    ctx.check()
    if (ctx.output / "training_launch.json").exists():
        raise ValueError(
            "training was already launched; inspect its receipt instead of restarting"
        )
    validate_kernel_sources()
    runtime = verify_native_fp4_runtime()
    gpu_uuid = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"], text=True
    ).strip()
    if gpu_uuid != config["gpu_uuid"]:
        raise ValueError("authorized B200 identity changed")
    _, flashqla = load_flashqla()
    write_json(
        ctx.output / "training_runtime.json", {"runtime": runtime, "flashqla": flashqla}
    )
    initial = (
        ctx.root / config["model"]["initial_adapter"] / "adapter_model.safetensors"
    )
    if file_hash(initial) != config["inputs"]["initial_weights"]["sha256"]:
        raise ValueError("original adapter initialization changed")
    with safe_open(str(initial), framework="pt") as weights:
        keys = [k for k in weights.keys() if "lora_B" in k]
        if len(keys) != 128 or any(
            torch.count_nonzero(weights.get_tensor(k)) for k in keys
        ):
            raise ValueError("replication requires the original fresh zero-B adapter")
    active = ctx.root / "results/b200_attention_gdn_serving/server.json"
    if active.exists():
        subprocess.run(
            [
                config["serving_python"],
                "-m",
                "experiments.b200_vllm031.runtime",
                "-m",
                "experiments.b200_vllm031.stop",
                "--archive-name",
                config["campaign_id"] + "_parent",
            ],
            cwd=ctx.root,
            env={**os.environ, "PYTHONPATH": f"{ctx.root / 'src'}:{ctx.root}"},
            check=True,
        )
    apps = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True
    ).strip()
    if any(int(pid) != os.getpid() for pid in apps.split()):
        raise ValueError("training launch requires no unrelated GPU process")
    job = ctx.job()
    command = training_command(job)
    command += [
        f"student.init_adapter={ctx.root / config['model']['initial_adapter']}",
        "++student.training.logging_steps=1",
        "student.training.lr_scheduler_type=linear",
        "student.training.warmup_ratio=0.03",
        "student.training.weight_decay=0.0",
    ]
    write_json(
        ctx.output / "execution_contract.json",
        {
            "job": job,
            "command": command,
            "manifest_sha256": file_hash(ctx.data / "manifest.json"),
            "config_sha256": file_hash(ctx.config_path),
        },
    )
    ctx.logs.mkdir(parents=True, exist_ok=True)
    with (ctx.logs / "train.log").open("x") as log:
        process = subprocess.Popen(
            command,
            cwd=ctx.root,
            env=dict(os.environ),
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    write_json(
        ctx.output / "training_launch.json",
        {"pid": process.pid, "command": command, "launched_at_unix": time.time()},
    )
    write_json(
        ctx.output / "status.json",
        {"stage": "training", "pid": process.pid, "steps": config["expected_steps"]},
    )
    print("campaign_training_launched", process.pid, flush=True)
    result = process.wait()
    if result:
        raise RuntimeError(f"campaign Trainer exited with status {result}")
    ctx.check()
    metadata = json.loads(
        (ctx.adapter / "causal_adapter/training_metadata.json").read_text()
    )
    validate_completion(metadata, ctx)
    rebase_adapter(ctx.adapter / "causal_adapter", ctx.adapter / "model")
    write_json(
        ctx.adapter / "complete.json",
        {
            "status": "trained",
            "steps": config["expected_steps"],
            "epochs": 1.0,
            "manifest_sha256": file_hash(ctx.data / "manifest.json"),
            "master_sha256": file_hash(
                ctx.adapter / "causal_adapter/adapter_model.safetensors"
            ),
            "serving_sha256": file_hash(
                ctx.adapter / "model/adapter_model.safetensors"
            ),
        },
    )
    write_json(
        ctx.output / "status.json",
        {"stage": "trained", "steps": config["expected_steps"]},
    )
    print("campaign_training_complete", flush=True)


def master_reference(ctx: Campaign) -> None:
    """Use original-FLA BF16 causal inference with FP32 masters on training rows."""
    config = ctx.config
    ctx.check()
    context = EvaluationContext(ctx.data, ctx.output, ("monitor",), {}, splits=("id",))
    reference(
        context, {"models": {"4b": config["model"]}, "parity": config["parity"]}, "4b"
    )


def merge(ctx: Campaign) -> None:
    """Verify the pinned source shards and retain all masters before merging."""
    config = ctx.config
    ctx.check()
    complete = json.loads((ctx.adapter / "complete.json").read_text())
    selection = json.loads(ctx.input("serving_selection").read_text())
    old = json.loads((ctx.root / selection["merged_artifact"]).read_text())
    base = ctx.root / config["base_model"]
    for name, expected in old["source_files_sha256"].items():
        if file_hash(base / name) != expected:
            raise ValueError("pinned base weight/config identity changed")
    report = merge_checkpoint(
        base,
        ctx.adapter / "model",
        Path(config["merged_model"]),
        expected_adapter_sha256=complete["serving_sha256"],
        model_id=config["model"]["id"],
        revision=config["model"]["revision"],
    )
    write_json(ctx.output / "merged_artifact.json", report)
    print(
        "campaign_model_merged",
        report["changed_projection_count"],
        report["seconds"],
        flush=True,
    )
