"""Freeze 27B jobs for the ten-update packed training-speed benchmark.

The jobs copy the frozen 4B benchmark jobs (same 320 rows, Kimi targets,
selection, seed, LR, rank and logical batch) and change only the model, the
output paths and the memory policy: every decoder layer is checkpointed
(non-reentrant) in the control, because the 4B's uncheckpointed recipe needs ~330 GB of
activations for a 16K-token 27B pack.
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "results/b200_adaptive_microbatching"
OUTPUT = ROOT / "results/qwen38_27b_training_speed"
REMOTE = Path("/workspace/gleipnir/results/qwen38_27b_training_speed")
MODEL = "Qwen/Qwen3.8-27B"
REVISION = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"


GDN_LAYERS = [i for i in range(64) if i % 4 != 3]
# Arm name -> (checkpoint policy, explicit layer indices or None).
CHECKPOINTING = {
    "": ("all", None),
    # OOM: storing all 16 attention layers needs >178 GiB.
    "gdn_ckpt": ("linear_attention_only", None),
    # Store every other attention layer (7, 15, ..., 63); checkpoint the rest.
    "attn8": ("explicit", sorted(GDN_LAYERS + list(range(3, 64, 8)))),
}


def job_27b(job: dict, prefix: str, arm: str = "") -> dict:
    policy, indices = CHECKPOINTING[arm]
    out = REMOTE / (f"{prefix}_{arm}" if arm else prefix) / job["job_name"]
    return {
        **job,
        "model": MODEL,
        "model_revision": REVISION,
        "gradient_checkpointing": True,
        "gradient_checkpointing_policy": policy,
        "gradient_checkpointing_layer_indices": indices,
        "nonreentrant_checkpointing": True,
        "output_dir": str(out),
        "causal_adapter_dir": str(out / "causal_adapter"),
        "model_dir": str(out / "model"),
        "hydra_log_dir": "logs/runpod/qwen38_27b_training_speed/hydra",
    }


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for (name, prefix), arm in itertools.product(
        (("jobs", "runs"), ("preflight_jobs", "preflight")), CHECKPOINTING
    ):
        rows = [
            json.loads(line)
            for line in (SOURCE / f"{name}.jsonl").read_text().splitlines()
            if line
        ]
        text = "".join(
            json.dumps(job_27b(row, prefix, arm), sort_keys=True) + "\n"
            for row in rows
        )
        path = OUTPUT / (f"{name}_{arm}.jsonl" if arm else f"{name}.jsonl")
        path.write_text(text)
        print(f"wrote {path} ({len(rows)} jobs)")


if __name__ == "__main__":
    main()
