"""Recover an arm whose merge failed only because the container disk filled.

Scored arms' merged weights are retired after the release_merge checks (their
slots are per-arm, not the shared slot release_merge expects). The failed
attempt is archived, its partial merge removed, and only that stage is dropped
before resuming; completed training stays hash-bound.
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

from experiments.distillation_temperature.prepare import ARMS, ROOT
from experiments.distillation_temperature.run import STATUS
from gleipnir.campaigns.monitoring.__main__ import execute
from gleipnir.campaigns.monitoring.contract import Campaign
from gleipnir.data.monitoring import file_hash, write_json


def retire_scored_merge(ctx: Campaign) -> None:
    slot = Path(ctx.config["merged_model"])
    if slot.name != ctx.config["campaign_id"] or slot.is_symlink():
        raise ValueError("unexpected merge slot")
    complete = json.loads((ctx.output / "evaluation/complete.json").read_text())
    if not complete["parity_passed"] or complete["rows"] != 3012:
        raise ValueError("preserve merge until evaluation completes")
    artifact = json.loads((ctx.output / "merged_artifact.json").read_text())
    master = ctx.adapter / "causal_adapter/adapter_model.safetensors"
    if file_hash(master) != complete["master_sha256"]:
        raise ValueError("persistent master changed; preserve merged weights")
    for name, expected in artifact["files_sha256"].items():
        if file_hash(slot / name) != expected:
            raise ValueError("merged weights changed before slot retirement")
    manifest = slot / "merge_manifest.json"
    if json.loads(manifest.read_text()) != artifact:
        raise ValueError("merge manifest changed before slot retirement")
    shutil.copyfile(manifest, ctx.output / "archived_merge_manifest.json")
    write_json(
        ctx.output / "merged_slot_retired.json",
        {
            "destination": str(slot),
            "retired_at_unix": time.time(),
            "manifest_sha256": file_hash(ctx.output / "archived_merge_manifest.json"),
            "master_preserved": True,
            "weights_preserved": False,
            "reason": "container disk full during a later arm's merge",
        },
    )
    shutil.rmtree(slot)


def recover(name: str) -> None:
    if (ROOT / "results/b200_attention_gdn_serving/server.json").exists():
        raise ValueError("retire the scorer first")
    status = json.loads(STATUS.read_text())
    if status["status"] != "failed" or "merge exited" not in status["error"]:
        raise ValueError("only a failed-merge queue is recoverable here")
    for done in ARMS:
        if status["stages"].get(done, {}).get("status") == "complete":
            ctx = Campaign.load(ROOT, ROOT / "data" / done / "config.yaml")
            if Path(ctx.config["merged_model"]).exists():
                retire_scored_merge(ctx)
    ctx = Campaign.load(ROOT, ROOT / "data" / name / "config.yaml")
    runner = ctx.output / "runner.json"
    state = json.loads(runner.read_text())
    failed = state["stages"]["merge"]
    if (
        failed["status"] != "failed"
        or "No space left" not in (ctx.logs / "stages/merge.log").read_text()
    ):
        raise ValueError("merge did not fail on disk space")
    shutil.copyfile(runner, ctx.output / "runner_failed_merge_attempt.json")
    shutil.copyfile(
        ctx.logs / "stages/merge.log", ctx.logs / "stages/merge_failed_attempt.log"
    )
    (ctx.logs / "stages/merge.log").unlink()
    shutil.rmtree(ctx.config["merged_model"], ignore_errors=True)
    del state["stages"]["merge"]
    state["status"] = "paused_after_stage"
    state["recovery"] = {"stage": "merge", "reason": "No space left on device"}
    write_json(runner, state)
    status["stages"][name]["recovery"] = state["recovery"]
    status["status"] = "running"
    status.pop("error")
    write_json(STATUS, status)
    execute(ctx, resume=True)
    from experiments.teacher_disagreement_filtering.run import retire

    retire(ctx)
    status["stages"][name].update(status="complete", finished=time.time())
    status["status"] = "complete"
    write_json(STATUS, status)
    print("distillation_temperature_recovered", name, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("name", choices=list(ARMS))
    recover(parser.parse_args().name)
