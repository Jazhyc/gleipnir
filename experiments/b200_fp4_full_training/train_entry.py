"""Run the ordinary Trainer once, then retain its loaded model and kernel plans."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import runpy
import time
from pathlib import Path
from unittest.mock import patch

import torch
from transformers import Trainer, TrainerCallback

from experiments.b200_fp4_full_training.campaign import (
    CONFIG,
    DIRECTORY,
    OUTPUT,
    ROOT,
    configuration,
    file_hash,
    prepare,
    validate_training,
    write_json,
)
from gleipnir.cudnn_fp4_mlp import cache_metadata
from gleipnir.qwen35_adapter_rebase import rebase_adapter


def worker_status(stage: str, **extra) -> None:
    write_json(
        OUTPUT / "worker.json",
        {
            "pid": os.getpid(),
            "status": stage,
            "updated_at_unix": time.time(),
            **extra,
        },
    )


class Progress(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        worker_status("training", update=state.global_step, total=state.max_steps)

    def on_log(self, args, state, control, logs=None, **kwargs):
        print(
            f"full_training_progress update={state.global_step}/{state.max_steps} "
            f"logs={logs}",
            flush=True,
        )


def main() -> None:
    prepare()
    trainer = None
    initial = None
    original = Trainer.train

    def capture(self, *args, **kwargs):
        nonlocal trainer, initial
        trainer = self
        initial = [
            p.detach().cpu().clone() for p in self.model.parameters() if p.requires_grad
        ]
        self.add_callback(Progress())
        worker_status("training", update=0, total=configuration()["expected_steps"])
        return original(self, *args, **kwargs)

    try:
        worker_status("starting")
        with patch.object(Trainer, "train", capture):
            runpy.run_path(
                str(ROOT / "experiments/deception_distillation/train_student_sft.py"),
                run_name="__main__",
            )
        if trainer is None:
            raise ValueError("ordinary training did not expose its Trainer")
        config = configuration()
        metadata = json.loads(
            (DIRECTORY / "causal_adapter/training_metadata.json").read_text()
        )
        validate_training(metadata, config)
        rebase_adapter(DIRECTORY / "causal_adapter", DIRECTORY / "model")
        write_json(
            DIRECTORY / "complete.json",
            {
                "status": "trained",
                "epochs": 1,
                "steps": config["expected_steps"],
                "training_rows": config["training_rows"],
                "config_sha256": file_hash(CONFIG),
                "master_sha256": file_hash(
                    DIRECTORY / "causal_adapter/adapter_model.safetensors"
                ),
                "serving_sha256": file_hash(
                    DIRECTORY / "model/adapter_model.safetensors"
                ),
            },
        )
        trainer.model.zero_grad(set_to_none=True)
        trainer.optimizer = trainer.lr_scheduler = None
        trainer.callback_handler.optimizer = trainer.callback_handler.lr_scheduler = (
            None
        )
        torch.cuda.empty_cache()
        worker_status(
            "idle",
            update=config["expected_steps"],
            total=config["expected_steps"],
            native_cache=cache_metadata(),
        )
        print(
            f"full_training_complete updates={config['expected_steps']} "
            "model_retained=true",
            flush=True,
        )
        # Future authorized optimization work can replace this fixed extension file
        # and submit a checksum-bound request without reloading the model.
        inbox = OUTPUT / "worker_requests"
        inbox.mkdir(exist_ok=True)
        while True:
            requests = sorted(inbox.glob("*.json"))
            if not requests:
                time.sleep(0.5)
                continue
            path = requests[0]
            request = json.loads(path.read_text())
            if request.get("action") == "snapshot":
                worker_status("idle", native_cache=cache_metadata())
                path.rename(path.with_suffix(".done"))
            elif request.get("action") == "extension":
                source_path = Path(__file__).with_name("resident_extension.py")
                source = source_path.read_bytes()
                if hashlib.sha256(source).hexdigest() != request.get("source_sha256"):
                    raise ValueError("resident extension source checksum drift")
                spec = importlib.util.spec_from_file_location(
                    "full_training_resident_extension", source_path
                )
                module = importlib.util.module_from_spec(spec)
                exec(compile(source, str(source_path), "exec"), module.__dict__)
                worker_status("extension")
                module.run(trainer, initial, request)
                path.rename(path.with_suffix(".done"))
                worker_status("idle", native_cache=cache_metadata())
            elif request.get("action") == "shutdown":
                path.rename(path.with_suffix(".done"))
                worker_status("stopped")
                return
            else:
                raise ValueError("unsupported resident full-run request")
    except BaseException as error:
        worker_status("failed", error=f"{type(error).__name__}: {error}")
        raise


if __name__ == "__main__":
    main()
