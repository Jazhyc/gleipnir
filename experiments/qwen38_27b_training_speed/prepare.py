"""Freeze 27B jobs for the ten-update packed training-speed benchmark.

The jobs copy the frozen 4B benchmark jobs (same 320 rows, Kimi targets,
selection, seed, LR, rank and logical batch) and change only the model, the
output paths and the memory policy: every decoder layer is checkpointed
(non-reentrant), because the 4B's uncheckpointed recipe needs ~330 GB of
activations for a 16K-token 27B pack.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "results/b200_adaptive_microbatching"
OUTPUT = ROOT / "results/qwen38_27b_training_speed"
REMOTE = Path("/workspace/gleipnir/results/qwen38_27b_training_speed")
MODEL = "Qwen/Qwen3.8-27B"
REVISION = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"


def job_27b(job: dict, prefix: str) -> dict:
    out = REMOTE / prefix / job["job_name"]
    return {
        **job,
        "model": MODEL,
        "model_revision": REVISION,
        "gradient_checkpointing": True,
        "gradient_checkpointing_policy": "all",
        "gradient_checkpointing_layer_indices": None,
        "nonreentrant_checkpointing": True,
        "output_dir": str(out),
        "causal_adapter_dir": str(out / "causal_adapter"),
        "model_dir": str(out / "model"),
        "hydra_log_dir": "logs/runpod/qwen38_27b_training_speed/hydra",
    }


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for name, prefix in (("jobs", "runs"), ("preflight_jobs", "preflight")):
        rows = [
            json.loads(line)
            for line in (SOURCE / f"{name}.jsonl").read_text().splitlines()
            if line
        ]
        text = "".join(
            json.dumps(job_27b(row, prefix), sort_keys=True) + "\n" for row in rows
        )
        (OUTPUT / f"{name}.jsonl").write_text(text)
        print(f"wrote {OUTPUT / f'{name}.jsonl'} ({len(rows)} jobs)")


if __name__ == "__main__":
    main()
