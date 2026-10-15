"""Train the seed-1 SDPA teacher, then score seed 1 OOD and rescore seed 0."""

from __future__ import annotations

import argparse
import asyncio
import fcntl
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

import yaml

from experiments.teacher_disagreement_filtering.run import retire
from experiments.teacher_seed_variation.prepare import ROOT, SCORING, SEED1
from gleipnir.campaigns.monitoring.__main__ import execute
from gleipnir.campaigns.monitoring.contract import Campaign
from gleipnir.campaigns.monitoring.evaluation import optimized
from gleipnir.campaigns.monitoring.training import merge
from gleipnir.campaigns.runtime import training_environment
from gleipnir.data.monitoring import file_hash, write_json

STATUS = ROOT / "results/teacher_seed_variation/status.json"
STAGE_TIMEOUT = 3 * 3600


class ScoreContext(Campaign):
    """Score an existing adapter on fixed splits with the shared helpers."""

    @property
    def adapter(self) -> Path:
        return self.root / self.config["parent_campaign"] / "4b/monitor"

    @property
    def splits(self) -> tuple[str, ...]:
        return tuple(self.config["score_splits"])

    def check(self) -> dict:
        manifest = json.loads((self.data / "manifest.json").read_text())
        for path, expected in manifest["files_sha256"].items():
            if file_hash(self.root / path) != expected:
                raise ValueError(f"scoring input drift: {path}")
        for path, expected in manifest["source_sha256"].items():
            if file_hash(self.source_root / path) != expected:
                raise ValueError(f"scoring source drift: {path}")
        return manifest


def scoring_context(name: str) -> ScoreContext:
    path = ROOT / "data" / name / "config.yaml"
    return ScoreContext(ROOT, path, yaml.safe_load(path.read_text()))


def reconstruct(ctx: ScoreContext) -> None:
    """Merge the parent's FP32 master again; reuse parity only if identical."""
    ctx.check()
    parent = ROOT / ctx.config["parent_campaign"]
    if Path(ctx.config["merged_model"]).exists():
        raise ValueError("scoring merge slot already exists")
    merge(ctx)
    original = json.loads((parent / "merged_artifact.json").read_text())
    actual = json.loads((ctx.output / "merged_artifact.json").read_text())
    for key in ("files_sha256", "source_files_sha256", "adapter_sha256"):
        if original[key] != actual[key]:
            raise ValueError(f"reconstructed merge differs from parent: {key}")
    receipt = json.loads((parent / "merged_parity.json").read_text())
    if not receipt["passed"]:
        raise ValueError("parent merge parity did not pass")
    receipt.update(
        performed_this_run=False,
        policy="reuse_identical_merge_and_parent_fp32_reference",
        source_receipt_sha256=file_hash(parent / "merged_parity.json"),
        merged_artifact_sha256=file_hash(ctx.output / "merged_artifact.json"),
    )
    write_json(ctx.output / "merged_parity.json", receipt)
    shutil.copyfile(
        parent / "canary_workload.jsonl", ctx.output / "canary_workload.jsonl"
    )
    print("seed_variation_identical_merge", ctx.config["campaign_id"], flush=True)


def subprocess_stage(module_args: list[str], python: str, env: dict, log: Path) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    env = {
        **env,
        "PYTHONPATH": f"{ROOT / 'src'}:{ROOT}:{env.get('PYTHONPATH', '')}",
        "GLEIPNIR_CAMPAIGN_DRIVER_PID": str(os.getpid()),
        "HF_HOME": str(ROOT / ".cache/huggingface"),
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
    }
    with log.open("x") as handle:
        subprocess.run(
            [python, *module_args],
            cwd=ROOT,
            env=env,
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=True,
            timeout=STAGE_TIMEOUT,
        )


def score(name: str) -> None:
    ctx = scoring_context(name)
    ctx.output.mkdir(parents=True, exist_ok=True)
    module = ["-m", "experiments.teacher_seed_variation.run", "--stage"]
    subprocess_stage(
        [*module, "merge", name],
        ctx.config["training_python"],
        training_environment(ROOT, name),
        ctx.logs / "merge.log",
    )
    subprocess_stage(
        ["-m", "experiments.b200_vllm031.runtime", *module, "score", name],
        ctx.config["serving_python"],
        dict(os.environ),
        ctx.logs / "score.log",
    )
    retire(ctx)


def run() -> None:
    gpu = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"], text=True
    ).strip()
    seed = Campaign.load(ROOT, ROOT / "data" / SEED1 / "config.yaml")
    if gpu != seed.config["gpu_uuid"]:
        raise ValueError("authorized GPU identity changed")
    if STATUS.exists():
        raise ValueError("seed-variation queue already launched")
    state = {"status": "running", "driver_pid": os.getpid(), "stages": {}}
    write_json(STATUS, state)
    try:
        for name in [SEED1, *SCORING]:
            state["stages"][name] = {"status": "running", "started": time.time()}
            write_json(STATUS, state)
            if name == SEED1:
                inventory = seed.inventory(runtime=True)
                if not inventory["passed"]:
                    raise ValueError("\n".join(inventory["errors"]))
                execute(seed)
                retire(seed)
            else:
                score(name)
            state["stages"][name].update(status="complete", finished=time.time())
            write_json(STATUS, state)
            print("seed_variation_stage_complete", name, flush=True)
    except BaseException as error:
        state.update(status="failed", error=f"{type(error).__name__}: {error}")
        write_json(STATUS, state)
        raise
    state["status"] = "complete"
    write_json(STATUS, state)
    print("seed_variation_complete", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["merge", "score"])
    parser.add_argument("name", nargs="?")
    args = parser.parse_args()
    if args.stage == "merge":
        reconstruct(scoring_context(args.name))
    elif args.stage == "score":
        asyncio.run(optimized(scoring_context(args.name)))
    else:
        STATUS.parent.mkdir(parents=True, exist_ok=True)
        with (STATUS.parent / "queue.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            run()
