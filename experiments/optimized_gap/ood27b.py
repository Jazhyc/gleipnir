"""Score Gleipnir 27B on the frozen OOD suite through a serving condition.

The OOD counterpart of `qwen38_27b_serving/id_drift.py`: launch the server
like ``bench.run`` on the campaign's merged BF16 checkpoint (rebuilt by
`qwen38_27b_serving/remerge.py`), score all 6,395 frozen OOD workload rows
(the rendered prompts of the campaign's BF16 OOD run) at concurrency 128, and
compare with that run's BF16 predictions.
"""

from __future__ import annotations

import argparse
import asyncio
import json

from experiments.b200_monitor_score.run import trial
from experiments.optimized_gap.prepare import OUT, ROOT
from experiments.qwen38_27b_serving.bench import MODEL, SETTINGS, quantized_command
from experiments.qwen38_27b_serving.bench import sources as serving_sources
from experiments.teacher_seed_variation.compare import load, score_difference, summary
from gleipnir.data.monitoring import file_hash, read_rows, write_json, write_rows
from gleipnir.serving import bf16_launch
from gleipnir.serving.benchmark import measurement_summary

WORKLOAD = ROOT / "data/teacher-disagreement-strict-ood01/workload.jsonl"
WORKLOAD_SHA256 = "9319514878c9ee468b117c4a622847baf025070f6b5abacee4d90af456cf9f4a"
BF16 = "results/qwen38-27b-s0-ood01/evaluation/ood.jsonl"


def sources() -> dict[str, str]:
    here = sorted((ROOT / "experiments/optimized_gap").glob("*.py"))
    return {
        **serving_sources(),
        **{str(p.relative_to(ROOT)): file_hash(p) for p in here},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("name")
    parser.add_argument("--worker", default=bf16_launch.BF16_WORKER)
    parser.add_argument("--additional", default="{}")
    parser.add_argument("--quantization")
    parser.add_argument("--compilation", default="{}", help="vLLM compilation config")
    args = parser.parse_args()
    if file_hash(WORKLOAD) != WORKLOAD_SHA256:
        raise ValueError("frozen OOD workload drift")
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
    logs = ROOT / "logs/runpod/optimized_gap"
    process = bf16_launch.start(
        ROOT, command, env, logs / f"{args.name}.log", out / "server.json"
    )
    try:
        bf16_launch.wait_ready(process, SETTINGS["port"])
        workload = read_rows(WORKLOAD)
        rows, seconds = asyncio.run(trial(workload, 128, SETTINGS))
    finally:
        bf16_launch.stop(process)
    if sorted(r["id"] for r in rows) != sorted(r["id"] for r in workload):
        raise ValueError("incomplete OOD coverage")
    write_rows(out / "ood.jsonl", [{"id": r["id"], "score": r["score"]} for r in rows])
    path = str((out / "ood.jsonl").relative_to(ROOT))
    condition, reference = load(path, "ood"), load(BF16, "ood")
    by_id = {r["id"]: r for r in condition}
    condition = [by_id[r["id"]] for r in reference]
    report = {
        "condition": summary(condition),
        "bf16": summary(reference),
        "scores_vs_bf16": score_difference(condition, reference),
        "timing": measurement_summary(rows, seconds),
        "command": {
            "worker": args.worker,
            "quantization": args.quantization,
            "additional": json.loads(args.additional),
            "compilation": json.loads(args.compilation),
        },
    }
    write_json(out / "ood_gap.json", report)
    c, b = report["condition"]["macro"], report["bf16"]["macro"]
    print(
        f"OOD pAUROC@20 {c['pauroc_at_20']:.4f} (BF16 {b['pauroc_at_20']:.4f}) "
        f"AUROC {c['auroc']:.4f} (BF16 {b['auroc']:.4f}) "
        f"MAE {report['scores_vs_bf16']['mae']:.4f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
