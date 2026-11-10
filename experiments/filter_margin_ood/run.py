"""Rebuild one margin arm's merged checkpoint and score it with stock vLLM."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import httpx
import numpy as np

from experiments.release_serving.stock_vllm import classify, serve_command
from experiments.release_serving.targets import ROOT
from experiments.release_serving.validate import labels, macro_pauroc, rows
from gleipnir.adapters.merge import merge_checkpoint
from gleipnir.data.monitoring import read_rows, write_json, write_rows

ARMS = ("control", "delta000", "delta005", "delta010", "delta020")
WORKLOADS = {
    "id": ("results/b200_optimized_id/id02/workload.json", 3012),
    "ood": ("data/teacher-disagreement-strict-ood01/workload.jsonl", 6395),
}
GATES = {"score_mae": 0.01, "pauroc_difference": 0.005}
BATCH_ROWS = 128
MERGED = Path("/tmp/gleipnir-merged/filter-margin-ood")
OUT = ROOT / "results/filter_margin_ood"


def campaign(arm: str) -> Path:
    return ROOT / f"results/teacher-disagreement-{arm}-sdpa04"


def agreement(new: dict[str, float], recorded: dict[str, float], split: str) -> dict:
    """Score agreement and macro pAUROC@20 of new rows against recorded rows."""
    if set(new) != set(recorded):
        raise ValueError(f"{split}: scored and recorded rows differ")
    ids = sorted(new)
    a = np.array([new[i] for i in ids])
    b = np.array([recorded[i] for i in ids])
    if not np.isfinite(a).all():
        raise ValueError(f"{split}: non-finite score")
    truth = labels(split)
    result = {
        "rows": len(ids),
        "score_mae": float(np.abs(a - b).mean()),
        "score_max_abs": float(np.abs(a - b).max()),
        "pauroc_at_20": macro_pauroc(new, truth, split),
        "recorded_pauroc_at_20": macro_pauroc(recorded, truth, split),
    }
    result["passed"] = (
        result["score_mae"] <= GATES["score_mae"]
        and abs(result["pauroc_at_20"] - result["recorded_pauroc_at_20"])
        <= GATES["pauroc_difference"]
    )
    return result


def merge(arm: str, adapter: Path, base: Path) -> dict:
    recorded = json.loads((campaign(arm) / "merged_artifact.json").read_text())
    if MERGED.exists():
        shutil.rmtree(MERGED)
    report = merge_checkpoint(
        base,
        adapter,
        MERGED,
        expected_adapter_sha256=recorded["adapter_sha256"],
        model_id=recorded["model"],
        revision=recorded["revision"],
    )
    identical = report["files_sha256"] == recorded["files_sha256"]
    if not identical:
        raise ValueError(f"{arm}: merged checkpoint differs from the recorded merge")
    return {"identical": identical, "files_sha256": report["files_sha256"]}


def start_server(port: int, log: Path) -> subprocess.Popen:
    command = serve_command(str(MERGED), "local", port)
    # The image's ``vllm`` script has a stale shebang; run the module instead.
    command[0:1] = [sys.executable, "-m", "vllm.entrypoints.cli.main"]
    at = command.index("--revision")
    del command[at : at + 2]
    handle = log.open("w")
    process = subprocess.Popen(  # noqa: S603
        command, stdout=handle, stderr=subprocess.STDOUT, start_new_session=True
    )
    deadline = time.time() + 900
    while time.time() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"vLLM exited with {process.returncode}; see {log}")
        try:
            if httpx.get(f"http://127.0.0.1:{port}/health", timeout=5).is_success:
                return process
        except httpx.HTTPError:
            pass
        time.sleep(5)
    stop_server(process)
    raise TimeoutError("vLLM did not become ready in 900 s")


def stop_server(process: subprocess.Popen) -> None:
    if process.poll() is None:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=60)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)


def score_split(split: str, port: int) -> dict[str, float]:
    path, expected = WORKLOADS[split]
    workload = rows(path)
    scored: list[dict] = []
    for start in range(0, len(workload), BATCH_ROWS):
        scored += asyncio.run(classify(workload[start : start + BATCH_ROWS], port))
    if len(scored) != expected:
        raise ValueError(f"{split}: incomplete coverage")
    return {r["id"]: float(r["score"]) for r in scored}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8100)
    args = parser.parse_args()
    out = OUT / args.arm
    out.mkdir(parents=True, exist_ok=True)
    report: dict = {"arm": args.arm, "gates": GATES}
    report["merge"] = merge(args.arm, args.adapter, args.base)
    process = start_server(args.port, out / "vllm.log")
    try:
        new = score_split("id", args.port)
        recorded = {
            str(r["id"]): float(r["score"])
            for r in read_rows(campaign(args.arm) / "evaluation/id.jsonl")
        }
        report["id"] = agreement(new, recorded, "id")
        write_rows(out / "id.jsonl", [{"id": i, "score": s} for i, s in new.items()])
        write_json(out / "report.json", report)
        if not report["id"]["passed"]:
            raise ValueError(f"{args.arm}: validation gate failed: {report['id']}")
        ood = score_split("ood", args.port)
        write_rows(out / "ood.jsonl", [{"id": i, "score": s} for i, s in ood.items()])
        report["ood"] = {
            "rows": len(ood),
            "pauroc_at_20": macro_pauroc(ood, labels("ood"), "ood"),
        }
    finally:
        stop_server(process)
        shutil.rmtree(MERGED, ignore_errors=True)
    write_json(out / "report.json", report)
    print("arm_complete", json.dumps(report["ood"]), flush=True)


if __name__ == "__main__":
    main()
