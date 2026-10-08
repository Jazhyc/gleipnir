"""Run an immutable monitoring campaign on existing authorized compute."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import subprocess
import time
from collections.abc import Callable
from pathlib import Path

from gleipnir.campaigns.runtime import training_environment
from gleipnir.data.monitoring import file_hash, read_rows, write_json

from .contract import REGISTRY, Campaign

STAGES = ("train", "master-reference", "merge", "merged-reference", "evaluate")


def products(ctx: Campaign, stage: str) -> list[Path]:
    """Bind the actual stage outputs, including merged shards and FP32 masters."""
    if stage == "train":
        return [
            ctx.adapter / "complete.json",
            ctx.adapter / "causal_adapter/training_metadata.json",
            ctx.adapter / "causal_adapter/adapter_config.json",
            ctx.adapter / "causal_adapter/adapter_model.safetensors",
            ctx.adapter / "model/adapter_config.json",
            ctx.adapter / "model/adapter_model.safetensors",
            ctx.adapter / "model/rebase_manifest.json",
        ]
    if stage == "master-reference":
        return [ctx.adapter / "parity_reference.json"]
    if stage == "merge":
        receipt = json.loads((ctx.output / "merged_artifact.json").read_text())
        return [
            ctx.output / "merged_artifact.json",
            Path(ctx.config["merged_model"]) / "merge_manifest.json",
            *[
                Path(ctx.config["merged_model"]) / name
                for name in receipt["files_sha256"]
            ],
        ]
    if stage == "merged-reference":
        return [ctx.output / "merged_parity.json", ctx.output / "canary_workload.jsonl"]
    return [
        ctx.output / "summary.json",
        ctx.output / "optimized_parity.json",
        ctx.output / "optimized_canary_predictions.json",
        ctx.output / "evaluation/complete.json",
        *[ctx.output / "evaluation" / f"{s}.jsonl" for s in ctx.splits],
    ]


def worker(ctx: Campaign, stage: str) -> None:
    """Launch each GPU stage in its pinned runtime, releasing models on exit."""
    ctx.check()
    config = ctx.config
    if stage == "evaluate":
        python = config["serving_python"]
        command = [
            python,
            "-m",
            "experiments.b200_vllm031.runtime",
            "-m",
            "gleipnir.campaigns.monitoring",
            "_worker",
        ]
        env = dict(os.environ)
    else:
        python = config["training_python"]
        command = [python, "-m", "gleipnir.campaigns.monitoring", "_worker"]
        env = training_environment(ctx.root, config["campaign_id"], native_fp4_mlp=True)
    env.update(
        PYTHONPATH=f"{ctx.source_root / 'src'}:{ctx.source_root}",
        HF_HOME=str(ctx.root / ".cache/huggingface"),
        HF_HUB_CACHE=str(ctx.root / ".cache/huggingface/hub"),
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        TOKENIZERS_PARALLELISM="false",
        GLEIPNIR_CAMPAIGN_DRIVER_PID=str(os.getpid()),
        PATH=f"{Path(python).parent}:/usr/local/cuda/bin:{env.get('PATH', '')}",
    )
    command += [
        "--config",
        str(ctx.config_path),
        "--root",
        str(ctx.root),
        "--stage",
        stage,
    ]
    ctx.logs.mkdir(parents=True, exist_ok=True)
    log_path = ctx.logs / "stages" / f"{stage}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("x") as log:
        process = subprocess.Popen(
            command,
            cwd=ctx.root,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        write_json(
            ctx.output / "stage_launches" / f"{stage}.json",
            {
                "pid": process.pid,
                "command": command,
                "log": str(log_path),
                "launched_at_unix": time.time(),
            },
        )
        result = process.wait()
    if result:
        raise RuntimeError(f"{stage} exited {result}; see {log_path}")


def execute(
    ctx: Campaign,
    *,
    through: str = "evaluate",
    resume: bool = False,
    launch: Callable[[Campaign, str], None] = worker,
) -> dict:
    """Skip only complete, hash-bound stages; never silently repeat training."""
    inventory = ctx.inventory(runtime=True)
    if not inventory["passed"]:
        raise ValueError(
            "campaign prerequisites failed:\n" + "\n".join(inventory["errors"])
        )
    ctx.output.mkdir(parents=True, exist_ok=True)
    with (ctx.output / "runner.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError(
                "another campaign driver holds this campaign lock"
            ) from error
        ctx.prepare()
        state_path = ctx.output / "runner.json"
        if state_path.exists() and not resume:
            raise ValueError("campaign already started; inspect status or use --resume")
        if resume and not state_path.exists():
            raise ValueError("no campaign receipt to resume")
        state = (
            json.loads(state_path.read_text())
            if resume
            else {"stages": {}, "status": "ready"}
        )
        for stage in STAGES[: STAGES.index(through) + 1]:
            ctx.check()
            previous = state["stages"].get(stage)
            if previous:
                if previous["status"] != "complete":
                    raise ValueError(
                        f"{stage} has an unfinished/failed attempt; "
                        "inspect its launch and log before recovery"
                    )
                for path, expected in previous["files_sha256"].items():
                    if file_hash(Path(path)) != expected:
                        raise ValueError(f"completed {stage} output changed: {path}")
                continue
            started = time.time()
            state["status"] = "running"
            state["stages"][stage] = {
                "status": "running",
                "started_at_unix": started,
                "driver_pid": os.getpid(),
            }
            write_json(state_path, state)
            try:
                launch(ctx, stage)
                ctx.check()
                paths = products(ctx, stage)
                missing = [str(p) for p in paths if not p.is_file()]
                if missing:
                    raise ValueError(f"{stage} missing required products: {missing}")
                hashes = {str(p): file_hash(p) for p in paths}
                if not hashes:
                    raise ValueError(f"{stage} produced no bound artifacts")
                state["stages"][stage].update(
                    status="complete", finished_at_unix=time.time(), files_sha256=hashes
                )
                write_json(state_path, state)
            except BaseException as error:
                state["status"] = "failed"
                state["stages"][stage].update(
                    status="failed",
                    finished_at_unix=time.time(),
                    error_type=type(error).__name__,
                    error=str(error),
                )
                write_json(state_path, state)
                raise
        state["status"] = "complete" if through == "evaluate" else "paused_after_stage"
        write_json(state_path, state)
        return state


def stage_worker(ctx: Campaign, stage: str) -> None:
    ctx.check()
    if stage in ("train", "master-reference", "merge"):
        from . import training

        {
            "train": training.train,
            "master-reference": training.master_reference,
            "merge": training.merge,
        }[stage](ctx)
    else:
        from . import evaluation

        if stage == "merged-reference":
            evaluation.merged_reference(ctx)
        else:
            import asyncio

            asyncio.run(evaluation.optimized(ctx))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action",
        choices=(
            "inventory",
            "prepare",
            "run",
            "status",
            "summarize",
            "registry",
            "_worker",
        ),
    )
    parser.add_argument("--config", type=Path)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--through", choices=STAGES, default="evaluate")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stage", choices=STAGES)
    args = parser.parse_args()
    if args.action == "registry":
        print(json.dumps(REGISTRY))
        return
    if args.config is None:
        parser.error("--config is required")
    ctx = Campaign.load(args.root, args.config)
    if args.action == "inventory":
        value = ctx.inventory()
        print(json.dumps(value, indent=2))
        if not value["passed"]:
            raise SystemExit(1)
    elif args.action == "prepare":
        value = ctx.prepare()
        print(
            json.dumps(
                {
                    "status": "prepared",
                    "files": len(value["files_sha256"]),
                    "manifest": str(ctx.data / "manifest.json"),
                }
            )
        )
    elif args.action == "run":
        value = execute(ctx, through=args.through, resume=args.resume)
        print(json.dumps({"status": value["status"], "stages": list(value["stages"])}))
    elif args.action == "status":
        print((ctx.output / "runner.json").read_text())
    elif args.action == "summarize":
        from gleipnir.serving.monitor_score import validate_score_response

        from .evaluation import attach_inputs, summarize

        complete = json.loads((ctx.output / "evaluation/complete.json").read_text())
        if complete["config_sha256"] != file_hash(ctx.config_path):
            raise ValueError("completed evaluation configuration drift")
        for split, expected in complete["files_sha256"].items():
            if file_hash(ctx.output / "evaluation" / f"{split}.jsonl") != expected:
                raise ValueError(f"completed predictions changed: {split}")
        populations = {
            s: read_rows(ctx.output / "evaluation" / f"{s}.jsonl") for s in ctx.splits
        }
        for split, rows in populations.items():
            for row in rows:
                validate_score_response(row, row["prompt_tokens"])
            attach_inputs(rows, read_rows(ctx.input(split)))
        write_json(ctx.output / "summary.json", summarize(populations, ctx))
    else:
        if args.stage is None:
            parser.error("_worker requires --stage")
        stage_worker(ctx, args.stage)


if __name__ == "__main__":
    main()
