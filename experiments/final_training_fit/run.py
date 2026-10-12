"""Queue full training-set scoring for fixed regular 4B/9B checkpoints."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import subprocess
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml

from experiments.monitor_score_resolution.run import install_head_override
from experiments.teacher_disagreement_filtering.run import retire, terminate_owned
from gleipnir.adapters.merge import merge_checkpoint
from gleipnir.campaigns.monitoring import evaluation
from gleipnir.campaigns.monitoring.contract import Campaign
from gleipnir.campaigns.runtime import training_environment
from gleipnir.data.monitoring import file_hash, read_rows, write_json

from .metrics import row_fit

ROOT = Path(__file__).resolve().parents[2]
NAME = "final-training-fit02"
SLOT = Path("/tmp/gleipnir-merged/teacher-disagreement-sweep")


class FitContext(Campaign):
    @property
    def source_adapter(self):
        return (
            ROOT
            / "results/student_injection_awareness"
            / self.config["size"]
            / "regular"
        )

    @property
    def adapter(self):
        return self.output / "adapter"

    def check(self):
        manifest = json.loads((self.data / "manifest.json").read_text())
        for section in ("files_sha256", "source_sha256"):
            for name, digest in manifest[section].items():
                if file_hash(ROOT / name) != digest:
                    raise ValueError(f"frozen fit input/source drift: {name}")
        return manifest


def context(size):
    path = ROOT / "data" / f"{NAME}-{size}" / "config.yaml"
    return FitContext(ROOT, path, yaml.safe_load(path.read_text()))


def prepare():
    original = ROOT / "data/student_injection_awareness"
    teacher = original / "soft_targets.jsonl"
    training = original / "regular/student_rows.jsonl"
    rows = read_rows(training)
    targets = {(r["dataset"], str(r["index"])): r for r in read_rows(teacher)}
    assert len(rows) == len(targets) == 8688
    for size in ("4b", "9b"):
        prior = ROOT / "data" / f"final-training-fit01-{size}"
        old = json.loads((prior / "manifest.json").read_text())
        for name, digest in old["files_sha256"].items():
            if file_hash(ROOT / name) != digest:
                raise ValueError("prepared workload/input drift")
        config = yaml.safe_load((prior / "config.yaml").read_text())
        config.update(
            campaign_id=f"{NAME}-{size}",
            extra_sources=[
                str(p.relative_to(ROOT))
                for p in Path(__file__).parent.iterdir()
                if p.is_file()
            ],
        )
        data = ROOT / "data" / config["campaign_id"]
        if data.exists():
            raise ValueError("preserve prepared fit attempt")
        data.mkdir(parents=True)
        (data / "monitor").mkdir()
        (data / "monitor/student_rows.jsonl").symlink_to(training)
        (data / "workload.jsonl").symlink_to(prior / "workload.jsonl")
        config["inputs"]["id"] = {
            "path": str((prior / "inputs.jsonl").relative_to(ROOT)),
            "sha256": file_hash(prior / "inputs.jsonl"),
        }
        ctx = FitContext(ROOT, data / "config.yaml", config)
        ctx.adapter.mkdir(parents=True)
        for name in ("causal_adapter", "model", "parity_reference.json"):
            (ctx.adapter / name).symlink_to(ctx.source_adapter / name)
        master = ctx.source_adapter / "causal_adapter/adapter_model.safetensors"
        reference = json.loads(
            (ctx.source_adapter / "parity_reference.json").read_text()
        )
        if file_hash(master) != reference["master_sha256"]:
            raise ValueError("original parity/master drift")
        complete = json.loads((ctx.source_adapter / "complete.json").read_text())
        complete.update(
            master_sha256=file_hash(master),
            serving_sha256=file_hash(
                ctx.source_adapter / "model/adapter_model.safetensors"
            ),
            source=str(ctx.source_adapter.relative_to(ROOT)),
            policy="normalized_receipt_only_original_weights_unchanged",
        )
        write_json(ctx.adapter / "complete.json", complete)
        canonical = read_rows(prior / "inputs.jsonl")
        workload = read_rows(prior / "workload.jsonl")
        assert len(canonical) == len(workload) == 8688
        for raw, item, work in zip(rows, canonical, workload, strict=True):
            key = (raw["dataset"], str(raw["index"]))
            q = targets[key]
            assert item["id"] == work["id"] == "/".join(key)
            assert item["prompt"] == raw["student_prompt"]
            assert item["metadata"]["teacher_probability"] == q["soft_target"]
            assert item["metadata"]["ground_truth"] == raw["label"] == q["label"]
            assert q["rendered_prompt_sha256"] == raw["teacher_rendered_prompt_sha256"]
            assert 0 < work["prompt_tokens"] < 32768
        ctx.config_path.write_text(yaml.safe_dump(config, sort_keys=False))
        files = dict(old["files_sha256"])
        for path in (
            ctx.config_path,
            data / "workload.jsonl",
            ctx.adapter / "complete.json",
            ctx.adapter / "parity_reference.json",
        ):
            files[str(path.relative_to(ROOT))] = file_hash(path)
        write_json(
            data / "manifest.json",
            {
                "files_sha256": files,
                "source_sha256": ctx.sources(),
                "training_rows": 8688,
                "selection": "fixed final checkpoints; original 01 workloads reused",
            },
        )
        print(
            "fit_prepared",
            size,
            len(workload),
            sum(r["prompt_tokens"] for r in workload),
            flush=True,
        )


def replace_merge(ctx):
    if ctx.serving.joinpath("server.json").exists():
        raise ValueError("foreign server active")
    if SLOT.is_symlink():
        raise ValueError("unsafe shared merge slot")
    if SLOT.exists():
        artifact = json.loads((SLOT / "merge_manifest.json").read_text())
        for name, digest in artifact["files_sha256"].items():
            if file_hash(SLOT / name) != digest:
                raise ValueError("old merge drift")
        previous_adapter = Path(artifact["adapter"])
        if not previous_adapter.is_relative_to(ROOT / "results"):
            raise ValueError("previous adapter outside persistent results")
        previous_complete = json.loads(
            (previous_adapter.parent / "complete.json").read_text()
        )
        if (
            file_hash(previous_adapter / "adapter_model.safetensors")
            != artifact["adapter_sha256"]
        ):
            raise ValueError("previous persistent export drift")
        if (
            file_hash(
                previous_adapter.parent / "causal_adapter/adapter_model.safetensors"
            )
            != previous_complete["master_sha256"]
        ):
            raise ValueError("previous persistent master drift")
        # Both old and new persistent training masters remain untouched.
        write_json(ctx.output / "previous_merge_manifest.json", artifact)
        shutil.rmtree(SLOT)
    complete = json.loads((ctx.adapter / "complete.json").read_text())
    assert (
        file_hash(ctx.adapter / "causal_adapter/adapter_model.safetensors")
        == complete["master_sha256"]
    )
    report = merge_checkpoint(
        ROOT / ctx.config["base_model"],
        ctx.adapter / "model",
        SLOT,
        expected_adapter_sha256=complete["serving_sha256"],
        model_id=ctx.config["model"]["id"],
        revision=ctx.config["model"]["revision"],
    )
    write_json(ctx.output / "merged_artifact.json", report)


def summarize(ctx):
    done = json.loads((ctx.output / "evaluation/complete.json").read_text())
    path = ctx.output / "evaluation/id.jsonl"
    assert (
        done["rows"] == 8688
        and done["parity_passed"]
        and file_hash(path) == done["files_sha256"]["id"]
    )
    audit = json.loads((ctx.output / "bf16_audit.json").read_text())
    assert (
        audit["head_computation"]["passed"]
        and audit["head_computation"]["output_dtype"] == "torch.float32"
    )
    groups = defaultdict(list)
    for row in read_rows(path):
        q = row["teacher_probability"]
        values = {
            **row_fit(q, row["logits"][1] - row["logits"][0]),
            "score_mae": abs(q - row["score"]),
        }
        groups["pooled"].append(values)
        groups[row["source_dataset"]].append(values)
        groups[f"{row['source_dataset']}/label{row['ground_truth']}"].append(values)
    means = {
        key: {
            "rows": len(values),
            **{k: float(np.mean([v[k] for v in values])) for k in values[0]},
        }
        for key, values in groups.items()
    }
    sources = [v for k, v in means.items() if k != "pooled" and "/label" not in k]
    result = {
        "status": "complete",
        "population": "training_seen",
        "size": ctx.config["size"],
        "groups": means,
        "source_macro": {
            k: float(np.mean([v[k] for v in sources]))
            for k in sources[0]
            if k != "rows"
        },
        "predictions_sha256": file_hash(path),
    }
    write_json(ctx.output / "fit.json", result)
    print("final_checkpoint_fit", json.dumps(result), flush=True)


def stage_worker(ctx, stage):
    ctx.check()
    ctx.output.mkdir(parents=True, exist_ok=True)
    if stage == "merge":
        replace_merge(ctx)
    elif stage == "reference":
        evaluation.merged_reference(ctx)
    else:
        install_head_override()
        from gleipnir.serving import bf16_monitor
        from gleipnir.serving.monitor_score import classification_overrides

        original_command = bf16_monitor.command

        def model_command(parent, sources):
            command = original_command(parent, sources)
            model_config = json.loads(
                (Path(ctx.config["merged_model"]) / "config.json").read_text()
            )
            command[command.index("--hf-overrides") + 1] = json.dumps(
                classification_overrides(model_config)
            )
            return command

        bf16_monitor.command = model_command
        evaluation.frozen_workloads = lambda current: {
            "id": read_rows(current.data / "workload.jsonl")
        }
        asyncio.run(evaluation.optimized(ctx))
        summarize(ctx)
        retire(ctx)


def run():
    gpu = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"], text=True
    ).strip()
    if gpu != "GPU-c4ea12ea-10d1-23f8-b026-f5cead839de9":
        raise ValueError("authorized GPU identity drift")
    active = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True
    ).strip()
    if active:
        raise ValueError("foreign GPU process active")
    out = ROOT / "results" / NAME
    out.mkdir(parents=True, exist_ok=True)
    status = out / "status.json"
    if status.exists():
        raise ValueError("preserve existing queue state")
    state = {
        "status": "running",
        "driver_pid": os.getpid(),
        "started_at_unix": time.time(),
        "stages": {},
    }
    try:
        for size in ("4b", "9b"):
            ctx = context(size)
            ctx.logs.mkdir(parents=True, exist_ok=True)
            for stage in ("merge", "reference", "score"):
                state.update(size=size, stage=stage)
                write_json(status, state)
                python = (
                    ctx.config["serving_python"]
                    if stage == "score"
                    else ctx.config["training_python"]
                )
                cmd = [
                    python,
                    "-m",
                    "experiments.final_training_fit.run",
                    "--size",
                    size,
                    "--stage",
                    stage,
                ]
                if stage == "score":
                    cmd[1:1] = ["-m", "experiments.b200_vllm031.runtime"]
                env = (
                    training_environment(ROOT, ctx.config["campaign_id"])
                    if stage != "score"
                    else dict(os.environ)
                )
                env.update(
                    PYTHONPATH=f"{ROOT / 'src'}:{ROOT}:{env.get('PYTHONPATH', '')}",
                    HF_HOME=str(ROOT / ".cache/huggingface"),
                    HF_HUB_CACHE=str(ROOT / ".cache/huggingface/hub"),
                    HF_HUB_OFFLINE="1",
                    TRANSFORMERS_OFFLINE="1",
                    TOKENIZERS_PARALLELISM="false",
                    PYTHONUNBUFFERED="1",
                    GLEIPNIR_CAMPAIGN_DRIVER_PID=str(os.getpid()),
                    PATH=(
                        f"{Path(python).parent}:/usr/local/cuda/bin:"
                        f"{env.get('PATH', '')}"
                    ),
                )
                with (ctx.logs / f"{stage}.log").open("x") as log:
                    process = subprocess.Popen(
                        cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT
                    )
                    try:
                        code = process.wait(timeout=7200)
                    except BaseException:
                        terminate_owned(process, {})
                        raise
                if code:
                    raise RuntimeError(f"{size} {stage} failed: {code}")
                state["stages"][f"{size}/{stage}"] = {
                    "status": "complete",
                    "finished_at_unix": time.time(),
                }
        state.update(status="complete", finished_at_unix=time.time())
        write_json(
            out / "comparison.json",
            {
                size: json.loads((context(size).output / "fit.json").read_text())
                for size in ("4b", "9b")
            },
        )
    except BaseException as error:
        state.update(status="failed", error=str(error))
        try:
            retire(ctx)
        except Exception as cleanup:
            state["cleanup_error"] = str(cleanup)
        raise
    finally:
        write_json(status, state)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--size", choices=("4b", "9b"))
    parser.add_argument("--stage", choices=("merge", "reference", "score"))
    args = parser.parse_args()
    if args.prepare:
        prepare()
    elif args.stage:
        stage_worker(context(args.size), args.stage)
    else:
        run()
