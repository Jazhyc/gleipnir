"""Complete one frozen FP4 epoch and ID evaluation on the already allocated GPU."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

from experiments.b200_fp4_full_training.campaign import (
    CONFIG,
    DIRECTORY,
    OUTPUT,
    ROOT,
    configuration,
    file_hash,
    make_job,
    prepare,
    profile,
    write_json,
)
from gleipnir.monitoring_campaign_runtime import training_environment
from gleipnir.monitoring_training_command import training_command


def main() -> None:
    config = configuration()
    from gleipnir.native_fp4_training import validate_kernel_sources

    # Reject source drift before loading a model, using the same startup guard.
    validate_kernel_sources()
    manifest = prepare()
    job = make_job(config, profile(config)["recipe"])
    command = training_command(job)
    command[1:2] = ["-m", "experiments.b200_fp4_full_training.train_entry"]
    command.extend(
        [
            f"student.init_adapter={ROOT / config['initial_adapter']}",
            "++student.training.logging_steps=1",
        ]
    )
    environment = training_environment(ROOT, config["campaign_id"], native_fp4_mlp=True)
    environment.update(
        HF_HOME=str(ROOT / ".cache/huggingface"),
        HF_HUB_CACHE=str(ROOT / ".cache/huggingface/hub"),
        TOKENIZERS_PARALLELISM="false",
    )
    logroot = ROOT / "logs/runpod/b200_fp4_full_training"
    logroot.mkdir(parents=True, exist_ok=True)
    DIRECTORY.mkdir(parents=True, exist_ok=True)
    path = DIRECTORY / "job.json"
    if path.exists() and json.loads(path.read_text()) != job:
        raise ValueError("existing full-run job drift")
    write_json(path, job)
    if not (DIRECTORY / "complete.json").exists():
        if (OUTPUT / "launch.json").exists():
            raise ValueError("preserve existing startup; inspect before resuming")
        sources = [
            *Path(__file__).parent.glob("*.py"),
            CONFIG,
            Path(__file__).with_name("README.md"),
            *[
                ROOT / relative
                for relative in (
                    "experiments/deception_distillation/train_student_sft.py",
                    "src/gleipnir/native_fp4_training.py",
                    "src/gleipnir/cudnn_fp4_mlp.py",
                    "src/gleipnir/cudnn_fp4_gemm.py",
                    "src/gleipnir/cudnn_fp4_epilogue.py",
                    "src/gleipnir/nvfp4_pack.py",
                    "src/gleipnir/packed_training.py",
                    "src/gleipnir/packed_sequences.py",
                    "src/gleipnir/validated_startup.py",
                    "src/gleipnir/monitoring_campaign_evaluation.py",
                    "src/gleipnir/monitoring_campaign_runtime.py",
                    "src/gleipnir/monitoring_training_command.py",
                )
            ],
        ]
        source_sha256 = {}
        for source in sources:
            relative = source.resolve().relative_to(ROOT)
            source_sha256[str(relative)] = file_hash(source)
            target = OUTPUT / "executed_sources" / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(source.read_bytes())
        write_json(
            OUTPUT / "execution_contract.json",
            {
                "job": job,
                "command": command,
                "source_sha256": source_sha256,
                "manifest_sha256": file_hash(
                    ROOT / "data/b200_fp4_full_training/manifest.json"
                ),
                "runtime_shape_validation_sha256": (
                    file_hash(OUTPUT / "runtime_shape_validation.json")
                    if (OUTPUT / "runtime_shape_validation.json").exists()
                    else None
                ),
            },
        )
        with (logroot / "train.log").open("x") as log:
            process = subprocess.Popen(
                command,
                cwd=ROOT,
                env=environment,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        write_json(
            OUTPUT / "launch.json",
            {
                "pid": process.pid,
                "command": command,
                "manifest": manifest,
                "config_sha256": file_hash(CONFIG),
                "cache_paths": {k: v for k, v in environment.items() if "CACHE" in k},
            },
        )
        print(f"full_training_launched pid={process.pid}", flush=True)
        while not (DIRECTORY / "complete.json").exists():
            if process.poll() is not None:
                raise RuntimeError(f"full training exited with {process.returncode}")
            time.sleep(5)
    if not (DIRECTORY / "parity_reference.json").exists():
        with (logroot / "reference.log").open("a") as log:
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "experiments.b200_fp4_full_training.evaluate",
                    "--backend",
                    "reference",
                ],
                cwd=ROOT,
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
            )
    serving_env = dict(os.environ)
    serving_env.update(
        PYTHONPATH=f"{ROOT / 'src'}:{ROOT}",
        PATH=f"{ROOT / '.venv/bin'}:/usr/local/cuda/bin:{serving_env.get('PATH', '')}",
        CUDA_HOME="/usr/local/cuda",
        TOKENIZERS_PARALLELISM="false",
        HF_HOME=str(ROOT / ".cache/huggingface"),
        HF_HUB_CACHE=str(ROOT / ".cache/huggingface/hub"),
        VLLM_CACHE_ROOT=str(ROOT / ".cache/vllm/student_injection_awareness_v1"),
        TORCHINDUCTOR_CACHE_DIR=str(
            ROOT / ".cache/torchinductor/student_injection_awareness_v1"
        ),
        PYTHONUNBUFFERED="1",
    )
    for name in ("LD_LIBRARY_PATH", "TRITON_CACHE_DIR", "TILELANG_CACHE_DIR"):
        serving_env.pop(name, None)
    with (logroot / "id.log").open("a") as log:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "experiments.b200_fp4_full_training.evaluate",
                "--backend",
                "vllm",
            ],
            cwd=ROOT,
            env=serving_env,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
        )
    subprocess.run(
        [
            sys.executable,
            "-m",
            "experiments.b200_fp4_full_training.evaluate",
            "--backend",
            "summary",
        ],
        cwd=ROOT,
        env=serving_env,
        check=True,
    )
    print("full_replication_and_id_complete", flush=True)


if __name__ == "__main__":
    main()
