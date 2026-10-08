"""Run one fresh augmented epoch, merge its adapter, and evaluate fixed holdouts."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from experiments.b200_augmented_training.campaign import (
    ADAPTER,
    CONFIG,
    DATA,
    LOGS,
    OUTPUT,
    ROOT,
    check_binding,
    configuration,
    file_hash,
    input_path,
    make_job,
    prepare,
    profile,
    validate_completion,
    write_json,
)
from gleipnir.adapters.merge import merge_checkpoint
from gleipnir.adapters.rebase import rebase_adapter
from gleipnir.campaigns.runtime import training_environment
from gleipnir.campaigns.training_command import training_command
from gleipnir.evaluation.campaign import EvaluationContext, reference


def train(config: dict) -> None:
    """Invoke the ordinary current Trainer, preserving its one-epoch contract."""
    import torch
    from safetensors import safe_open

    from gleipnir.training.backends.flashqla import load_flashqla
    from gleipnir.training.backends.native_fp4 import (
        validate_kernel_sources,
        verify_native_fp4_runtime,
    )

    check_binding(config)
    validate_kernel_sources()
    runtime = verify_native_fp4_runtime()
    gpu_uuid = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"], text=True
    ).strip()
    if gpu_uuid != config["gpu_uuid"]:
        raise ValueError("authorized B200 identity changed")
    _, flashqla = load_flashqla()
    write_json(
        OUTPUT / "training_runtime.json", {"runtime": runtime, "flashqla": flashqla}
    )
    initial = ROOT / config["model"]["initial_adapter"] / "adapter_model.safetensors"
    if file_hash(initial) != config["inputs"]["initial_weights"]["sha256"]:
        raise ValueError("original adapter initialization changed")
    with safe_open(str(initial), framework="pt") as weights:
        keys = [k for k in weights.keys() if "lora_B" in k]
        if len(keys) != 128 or any(
            torch.count_nonzero(weights.get_tensor(k)) for k in keys
        ):
            raise ValueError("replication requires the original fresh zero-B adapter")
    active = ROOT / "results/b200_attention_gdn_serving/server.json"
    if active.exists():
        subprocess.run(
            [
                config["serving_python"],
                "-m",
                "experiments.b200_vllm031.runtime",
                "-m",
                "experiments.b200_vllm031.stop",
                "--archive-name",
                "augmented_training_parent01",
            ],
            cwd=ROOT,
            env={**os.environ, "PYTHONPATH": f"{ROOT / 'src'}:{ROOT}"},
            check=True,
        )
    apps = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True
    ).strip()
    if any(int(pid) != os.getpid() for pid in apps.split()):
        raise ValueError("training launch requires no unrelated GPU process")
    job = make_job(config, profile(config)["recipe"])
    command = training_command(job)
    command += [
        f"student.init_adapter={ROOT / config['model']['initial_adapter']}",
        "++student.training.logging_steps=1",
    ]
    write_json(
        OUTPUT / "execution_contract.json",
        {
            "job": job,
            "command": command,
            "manifest_sha256": file_hash(DATA / "manifest.json"),
            "config_sha256": file_hash(CONFIG),
        },
    )
    LOGS.mkdir(parents=True, exist_ok=True)
    with (LOGS / "train.log").open("x") as log:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            env=dict(os.environ),
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    write_json(
        OUTPUT / "training_launch.json",
        {"pid": process.pid, "command": command, "launched_at_unix": time.time()},
    )
    write_json(
        OUTPUT / "status.json",
        {"stage": "training", "pid": process.pid, "steps": config["expected_steps"]},
    )
    print("augmented_training_launched", process.pid, flush=True)
    result = process.wait()
    if result:
        raise RuntimeError(f"augmented Trainer exited with status {result}")
    check_binding(config)
    metadata = json.loads(
        (ADAPTER / "causal_adapter/training_metadata.json").read_text()
    )
    validate_completion(metadata, config, job)
    rebase_adapter(ADAPTER / "causal_adapter", ADAPTER / "model")
    write_json(
        ADAPTER / "complete.json",
        {
            "status": "trained",
            "steps": 272,
            "epochs": 1.0,
            "manifest_sha256": file_hash(DATA / "manifest.json"),
            "master_sha256": file_hash(
                ADAPTER / "causal_adapter/adapter_model.safetensors"
            ),
            "serving_sha256": file_hash(ADAPTER / "model/adapter_model.safetensors"),
        },
    )
    write_json(OUTPUT / "status.json", {"stage": "trained", "steps": 272})
    print("augmented_training_complete", flush=True)


def master_reference(config: dict) -> None:
    """Use original-FLA BF16 causal inference with FP32 masters on training rows."""
    check_binding(config)
    context = EvaluationContext(DATA, OUTPUT, ("augmented",), {}, splits=("id",))
    reference(
        context, {"models": {"4b": config["model"]}, "parity": config["parity"]}, "4b"
    )


def merge(config: dict) -> None:
    """Verify the pinned source shards and retain all masters before merging."""
    check_binding(config)
    complete = json.loads((ADAPTER / "complete.json").read_text())
    selection = json.loads(input_path(config, "serving_selection").read_text())
    old = json.loads((ROOT / selection["merged_artifact"]).read_text())
    base = ROOT / config["base_model"]
    for name, expected in old["source_files_sha256"].items():
        if file_hash(base / name) != expected:
            raise ValueError("pinned base weight/config identity changed")
    report = merge_checkpoint(
        base,
        ADAPTER / "model",
        Path(config["merged_model"]),
        expected_adapter_sha256=complete["serving_sha256"],
        model_id=config["model"]["id"],
        revision=config["model"]["revision"],
    )
    write_json(OUTPUT / "merged_artifact.json", report)
    print(
        "augmented_model_merged",
        report["changed_projection_count"],
        report["seconds"],
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=("all", "prepare", "train", "reference", "merge"),
        default="all",
    )
    args = parser.parse_args()
    config = configuration()
    # Metadata checks and training/reference imports must see the same pinned
    # overlays as the actual Trainer, not the serving process's environment.
    if os.environ.get("GLEIPNIR_AUGMENTED_TRAINING_ENV") != "1":
        env = training_environment(ROOT, config["campaign_id"], native_fp4_mlp=True)
        env.update(
            PYTHONPATH=env["PYTHONPATH"],
            GLEIPNIR_AUGMENTED_TRAINING_ENV="1",
            HF_HOME=str(ROOT / ".cache/huggingface"),
            HF_HUB_CACHE=str(ROOT / ".cache/huggingface/hub"),
            HF_HUB_OFFLINE="1",
            TRANSFORMERS_OFFLINE="1",
            TOKENIZERS_PARALLELISM="false",
            PATH=":".join(
                [
                    str(Path(config["training_python"]).parent),
                    "/usr/local/cuda/bin",
                    env.get("PATH", ""),
                ]
            ),
        )
        os.execve(
            config["training_python"],
            [
                config["training_python"],
                "-m",
                "experiments.b200_augmented_training.run",
                *sys.argv[1:],
            ],
            env,
        )
    if args.stage == "prepare":
        prepare()
        return
    if args.stage != "all":
        {"train": train, "reference": master_reference, "merge": merge}[args.stage](
            config
        )
        return
    try:
        prepare()
        train(config)
        write_json(OUTPUT / "status.json", {"stage": "master_reference"})
        with (LOGS / "master_reference.log").open("x") as log:
            subprocess.run(
                [
                    config["training_python"],
                    "-m",
                    "experiments.b200_augmented_training.run",
                    "--stage",
                    "reference",
                ],
                cwd=ROOT,
                env=dict(os.environ),
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
            )
        merge(config)
        # Merging is CPU work; each bounded reference uses a short-lived process
        # so no training/reference model stays resident beside the 90% scorer.
        with (LOGS / "merged_reference.log").open("x") as log:
            subprocess.run(
                [
                    config["training_python"],
                    "-m",
                    "experiments.b200_augmented_training.evaluate",
                    "--stage",
                    "merged-reference",
                ],
                cwd=ROOT,
                env=dict(os.environ),
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
            )
        write_json(OUTPUT / "status.json", {"stage": "optimized_evaluation"})
        with (LOGS / "evaluate.log").open("x") as log:
            evaluation_env = {
                **os.environ,
                "GLEIPNIR_AUGMENTED_DRIVER_PID": str(os.getpid()),
            }
            subprocess.run(
                [
                    config["serving_python"],
                    "-m",
                    "experiments.b200_vllm031.runtime",
                    "-m",
                    "experiments.b200_augmented_training.evaluate",
                    "--stage",
                    "optimized",
                ],
                cwd=ROOT,
                env=evaluation_env,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
            )
        write_json(
            OUTPUT / "status.json",
            {"stage": "complete", "steps": 272, "evaluated_rows": 12126},
        )
        print("augmented_replication_complete", flush=True)
    except BaseException as error:
        write_json(
            OUTPUT / "failure.json",
            {"type": type(error).__name__, "message": str(error)},
        )
        write_json(OUTPUT / "status.json", {"stage": "failed", "error": str(error)})
        raise


if __name__ == "__main__":
    main()
