"""Score the 27B ID set through a serving condition and compare with BF16.

Launches the server like ``bench.run``, scores all 3,012 frozen ID workload
rows (the campaign's rendered student prompts) at concurrency 128, and
reports macro pAUROC@20/AUROC and score drift against the campaign's BF16 ID
predictions (`results/qwen38-27b-s0-01/evaluation/id.jsonl`).
"""

from __future__ import annotations

import argparse
import asyncio
import json

import numpy as np

from experiments.b200_monitor_score.run import trial
from experiments.qwen27b_generations_ood.compare_id_train import macro
from experiments.qwen38_27b_serving.bench import (
    LOGS,
    MODEL,
    OUT,
    ROOT,
    SETTINGS,
    quantized_command,
    sources,
)
from gleipnir.data.monitoring import write_json
from gleipnir.serving import bf16_launch
from gleipnir.serving.benchmark import measurement_summary

WORKLOAD = ROOT / "results/b200_optimized_id/id02/workload.json"
BF16 = ROOT / "results/qwen38-27b-s0-01/evaluation/id.jsonl"


def compare(rows: list[dict]) -> dict:
    meta = {r["id"]: r for r in json.loads(WORKLOAD.read_text())}
    reference = {
        json.loads(line)["id"]: json.loads(line)
        for line in BF16.read_text().splitlines()
    }
    if set(meta) != {r["id"] for r in rows} or set(meta) != set(reference):
        raise ValueError("ID membership differs")
    scores = {r["id"]: r["score"] for r in rows}
    records = [(m["dataset"], int(m["label"]), scores[i]) for i, m in meta.items()]
    base = [
        (m["dataset"], int(m["label"]), float(reference[i]["score"]))
        for i, m in meta.items()
    ]
    a = np.array([r[2] for r in records])
    b = np.array([r[2] for r in base])
    return {
        "condition": macro(records),
        "bf16": macro(base),
        "score_mae": float(np.abs(a - b).mean()),
        "score_max": float(np.abs(a - b).max()),
        "correlation": float(np.corrcoef(a, b)[0, 1]),
        "flips": int(((a >= 0.5) != (b >= 0.5)).sum()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("name")
    parser.add_argument("--worker", default=bf16_launch.BF16_WORKER)
    parser.add_argument("--additional", default="{}")
    parser.add_argument("--quantization")
    parser.add_argument("--compilation", default="{}", help="vLLM compilation config")
    args = parser.parse_args()
    out = OUT / args.name
    out.mkdir(parents=True, exist_ok=False)
    selection, command = bf16_launch.bf16_command(
        ROOT,
        MODEL,
        sources(),
        worker=args.worker,
        additional=json.loads(args.additional),
    )
    if args.quantization:
        quantized_command(command, args.quantization, json.loads(args.additional))
    if json.loads(args.compilation):
        command += ["--compilation-config", args.compilation]
    env = bf16_launch.environment(ROOT, selection, out / "bf16_audit.json")
    process = bf16_launch.start(
        ROOT, command, env, LOGS / f"{args.name}.log", out / "server.json"
    )
    try:
        bf16_launch.wait_ready(process, SETTINGS["port"])
        workload = json.loads(WORKLOAD.read_text())
        rows, seconds = asyncio.run(trial(workload, 128, SETTINGS))
        write_json(out / "id_predictions.json", {"rows": rows, "seconds": seconds})
        report = {"timing": measurement_summary(rows, seconds), **compare(rows)}
        write_json(out / "id_drift.json", report)
        c, b = report["condition"], report["bf16"]
        print(
            f"ID pAUROC@20 {c['macro_pauroc_at_20']:.4f} "
            f"(BF16 {b['macro_pauroc_at_20']:.4f}) "
            f"AUROC {c['macro_auroc']:.4f} (BF16 {b['macro_auroc']:.4f}) "
            f"MAE {report['score_mae']:.4f} corr {report['correlation']:.5f} "
            f"flips {report['flips']}",
            flush=True,
        )
    finally:
        bf16_launch.stop(process)


if __name__ == "__main__":
    main()
