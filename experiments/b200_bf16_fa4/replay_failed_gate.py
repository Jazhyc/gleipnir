"""Replay the rejected eager gate to retain receipts, with no optimizer updates."""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

from experiments.b200_bf16_fa4.run import ROOT, benchmark_environment
from gleipnir.monitoring_systems_screen import sha256_file
from gleipnir.monitoring_training_command import training_command


def main() -> None:
    source = ROOT / "results/b200_bf16_fa4/summary.json"
    contract = json.loads(source.read_text())
    output = ROOT / "results/b200_bf16_fa4_eager_gate_replay"
    output.mkdir(parents=True, exist_ok=False)
    logs = ROOT / "logs/runpod/b200_bf16_fa4_eager_gate_replay"
    logs.mkdir(parents=True, exist_ok=True)
    job = {
        **contract["conditions"]["flash_attention_4"]["job"],
        "output_dir": str(output),
        "causal_adapter_dir": str(output / "causal_adapter"),
        "model_dir": str(output / "model"),
        "hydra_log_dir": str(logs),
    }
    for path_key, hash_key in [
        ("student_rows", "student_rows_sha256"),
        ("soft_targets", "soft_targets_sha256"),
        ("selection_manifest", "selection_sha256"),
    ]:
        if sha256_file(Path(job[path_key])) != job[hash_key]:
            raise ValueError(f"diagnostic input drift: {path_key}")
    command = training_command(job) + [
        f"student.init_adapter={ROOT / contract['config']['initial_adapter']}",
        "student.training.adaptive_microbatching.canary_only=true",
    ]
    report = {
        "status": "running",
        "purpose": "retain full receipt for rejected eager gate",
        "source_contract_sha256": sha256_file(source),
        "job": job,
        "command": command,
        "optimizer_updates": 0,
        "canary_only": True,
        "source_sha256": {
            name: sha256_file(ROOT / name)
            for name in [
                "experiments/b200_bf16_fa4/replay_failed_gate.py",
                "experiments/b200_bf16_fa4/run.py",
                "experiments/deception_distillation/train_student_sft.py",
                "src/gleipnir/packed_training.py",
                "src/gleipnir/packed_training_screen.py",
                "src/gleipnir/packed_sequences.py",
            ]
        },
    }
    target = output / "summary.json"
    target.write_text(json.dumps(report, indent=2) + "\n")
    started = time.perf_counter()
    with (logs / "diagnostic.log").open("w") as handle:
        result = subprocess.run(
            command,
            cwd=ROOT,
            env=benchmark_environment(contract["config"]),
            stdout=handle,
            stderr=subprocess.STDOUT,
        )
    receipt_path = Path(job["causal_adapter_dir"]) / "packing_canary.json"
    if receipt_path.exists():
        receipt = json.loads(receipt_path.read_text())
        eager = receipt["eager_canary"]
        report.update(
            status="reproduced_failure" if not eager["passed"] else "unexpected_pass",
            eager_canary=eager,
            initial_master_sha256=receipt["initial_master_sha256"],
            receipt_sha256=sha256_file(receipt_path),
        )
        if (
            receipt["initial_master_sha256"]
            != contract["config"]["expected_initial_master_sha256"]
        ):
            report["status"] = "initial_master_identity_drift"
    else:
        report.update(status="failed_without_packing_receipt")
    report.update(
        returncode=result.returncode, wall_seconds=time.perf_counter() - started
    )
    target.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                k: v
                for k, v in report.items()
                if k not in {"job", "command", "eager_canary"}
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
