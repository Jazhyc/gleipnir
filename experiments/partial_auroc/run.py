"""Automatically train and evaluate one immutable partial-AUC campaign."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np

from experiments.monitor_score_resolution.run import install_head_override
from experiments.teacher_disagreement_filtering.run import retire, terminate_owned
from gleipnir.campaigns.monitoring.__main__ import STAGES, execute
from gleipnir.campaigns.monitoring.contract import Campaign
from gleipnir.campaigns.monitoring.evaluation import attach_inputs, id_metrics
from gleipnir.campaigns.runtime import training_environment
from gleipnir.data.monitoring import file_hash, read_rows, write_json
from gleipnir.serving.monitor_score import validate_score_response

from .prepare import DATA, NAME, OUT, ROOT


class PaucContext(Campaign):
    @property
    def splits(self) -> tuple[str, ...]:
        return ("id", "ood")

    def job(self) -> dict:
        return {**super().job(), "partial_auc": self.config["partial_auc"]}


def context() -> PaucContext:
    return PaucContext.load(ROOT, DATA / "config.yaml")


def compare(ctx: PaucContext) -> dict:
    done = json.loads((OUT / "evaluation/complete.json").read_text())
    if done["rows"] != 9407 or not done["parity_passed"]:
        raise ValueError("candidate evaluation incomplete")
    result = {"status": "complete", "promotion": False, "cases": {}}
    for split, count in [("id", 3012), ("ood", 6395)]:
        path = OUT / "evaluation" / f"{split}.jsonl"
        rows = read_rows(path)
        if len(rows) != count or done["files_sha256"][split] != file_hash(path):
            raise ValueError("candidate count/hash drift")
        for row in rows:
            validate_score_response(row, row["prompt_tokens"])
        inputs = read_rows(ctx.input(split))
        baseline = read_rows(ctx.input("baseline_" + split))
        before = id_metrics(attach_inputs(baseline, inputs))
        after = id_metrics(attach_inputs(rows, inputs))
        for row, old in zip(rows, baseline, strict=True):
            if (
                row["prompt_sha256"] != old["prompt_sha256"]
                or row["prompt_tokens"] != old["prompt_tokens"]
            ):
                raise ValueError("matched rendered prompt drift")
        a, b = after["metrics"]["macro"]["macro"], before["metrics"]["macro"]["macro"]
        timing = json.loads((OUT / "evaluation" / f"{split}_timing.json").read_text())
        result["cases"][split] = {
            "candidate": after,
            "baseline": before,
            "source_macro_difference": {k: a[k] - b[k] for k in a},
            "unique_scores": len({r["score"] for r in rows}),
            "input_tokens_per_second": timing["tokens"] / timing["seconds"],
            "requests_per_second": count / timing["seconds"],
            "latency_seconds_median": float(
                np.median([r["latency_seconds"] for r in rows])
            ),
            "predictions_sha256": file_hash(path),
        }
    write_json(OUT / "comparison.json", result)
    print(
        "single_partial_auc_verified_comparison",
        json.dumps(
            {s: r["source_macro_difference"] for s, r in result["cases"].items()}
        ),
        flush=True,
    )
    return result


def replace_previous_merge(ctx: PaucContext) -> None:
    slot = Path(ctx.config["merged_model"])
    if (
        slot != Path("/tmp/gleipnir-merged/teacher-disagreement-sweep")
        or slot.is_symlink()
    ):
        raise ValueError("unexpected disposable merge slot")
    if not slot.exists():
        raise ValueError("expected strict baseline merge missing")
    if (ctx.serving / "server.json").exists():
        raise ValueError("active server must retire before merge replacement")
    previous = ROOT / "results/teacher-disagreement-strict-ood01/merged_artifact.json"
    artifact = json.loads(previous.read_text())
    if json.loads((slot / "merge_manifest.json").read_text()) != artifact:
        raise ValueError("previous strict merge identity drift")
    master = (
        ROOT
        / "results/teacher-disagreement-delta000-sdpa04/4b/monitor"
        / "causal_adapter/adapter_model.safetensors"
    )
    complete = json.loads((master.parents[1] / "complete.json").read_text())
    if file_hash(master) != complete["master_sha256"]:
        raise ValueError("persistent strict master changed")
    export = master.parents[1] / "model/adapter_model.safetensors"
    if file_hash(export) != artifact["adapter_sha256"]:
        raise ValueError("persistent strict inference adapter changed")
    for filename, sha in artifact["files_sha256"].items():
        if file_hash(slot / filename) != sha:
            raise ValueError("previous merge weights changed")
    shutil.copyfile(slot / "merge_manifest.json", OUT / "previous_merge_manifest.json")
    write_json(
        OUT / "previous_merge_retired.json",
        {
            "master_preserved": True,
            "policy": "user_requested_shared_ephemeral_merge_slot",
            "source": str(previous.relative_to(ROOT)),
        },
    )
    shutil.rmtree(slot)


def stage_worker(ctx: PaucContext, stage: str) -> None:
    ctx.check()
    from gleipnir.campaigns.monitoring import evaluation, training

    if stage == "train":
        original = training.training_command

        def command(job):
            result = original(job)
            result[1:2] = ["-m", "experiments.partial_auroc.train"]
            result[result.index("--config-path") + 1] = str(
                ROOT / "experiments/tool_trajectory_monitoring"
            )
            return result

        training.training_command = command
        os.environ["GLEIPNIR_PAUC_CONFIG"] = str(ctx.config_path)
        os.environ["GLEIPNIR_PAUC_RECEIPT"] = str(OUT / "partial_auc_training.json")
        training.train(ctx)
        receipt = json.loads((OUT / "partial_auc_training.json").read_text())
        records = receipt["records"]
        if (
            not receipt["gradient_canary"]["passed"]
            or len(records) != 245
            or sum(r["scout_tokens"] for r in records) != 75851675
            or not any(r["active_sources"] for r in records)
            or any(r["maximum_replay_margin_error"] > 1e-5 for r in records)
        ):
            raise ValueError("partial-AUC training objective/coverage check failed")
        path = ctx.adapter / "causal_adapter/training_metadata.json"
        meta = json.loads(path.read_text())
        meta["partial_auc"] = receipt
        write_json(path, meta)
    elif stage == "master-reference":
        training.master_reference(ctx)
    elif stage == "merge":
        replace_previous_merge(ctx)
        training.merge(ctx)
    elif stage == "merged-reference":
        evaluation.merged_reference(ctx)
    else:
        install_head_override()
        import asyncio

        asyncio.run(evaluation.optimized(ctx))
        compare(ctx)
        retire(ctx)


def launch(ctx: PaucContext, stage: str) -> None:
    python = (
        ctx.config["serving_python"]
        if stage == "evaluate"
        else ctx.config["training_python"]
    )
    command = [python, "-m", "experiments.partial_auroc.run", "--stage", stage]
    if stage == "evaluate":
        command[1:1] = ["-m", "experiments.b200_vllm031.runtime"]
    env = training_environment(ROOT, NAME) if stage != "evaluate" else dict(os.environ)
    env.update(
        PYTHONPATH=f"{ROOT / 'src'}:{ROOT}:{env.get('PYTHONPATH', '')}",
        GLEIPNIR_CAMPAIGN_DRIVER_PID=str(os.getpid()),
        PYTHONUNBUFFERED="1",
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        HF_HOME=str(ROOT / ".cache/huggingface"),
        HF_HUB_CACHE=str(ROOT / ".cache/huggingface/hub"),
        TOKENIZERS_PARALLELISM="false",
        PATH=f"{Path(python).parent}:/usr/local/cuda/bin:{env.get('PATH', '')}",
    )
    ctx.logs.mkdir(parents=True, exist_ok=True)
    with (ctx.logs / f"{stage}.stage.log").open("x") as log:
        process = subprocess.Popen(
            command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT
        )
        try:
            result = process.wait(timeout=10800)
        except BaseException:
            terminate_owned(process, {})
            raise
    if result:
        raise RuntimeError(f"{stage} failed with exit {result}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=STAGES)
    args = parser.parse_args()
    ctx = context()
    if args.stage:
        stage_worker(ctx, args.stage)
    else:
        try:
            ctx.prepare()
            execute(ctx, launch=launch)
        except BaseException:
            if (ctx.serving / "server.json").exists():
                retire(ctx)
            raise
