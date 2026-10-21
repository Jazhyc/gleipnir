"""Release targets: one validated optimized-serving configuration per model.

Each target replays the server command recorded by the run that produced the
model's published optimized scores, on the same merged-checkpoint path (it is
part of the compiler-cache key), and is validated against that run's rows.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# Served context: the checkpoints' native RoPE range. The recorded runs used
# 32,768; passing that limit replays them unchanged.
RECORDED_CONTEXT = 32768
CONTEXT_LIMIT = 262144


@dataclass(frozen=True)
class LongContext:
    """The hash-bound MXFP8 envelope receipt that permits contexts beyond 32K.

    ``receipt_env`` names the worker environment variable that carries the
    receipt (4B ``LongContextWorker`` lineage); otherwise the receipt is bound
    under ``additional_key`` in ``--additional-config`` (27B ``QuantFullWorker``).
    """

    receipt: str
    receipt_sha256: str
    worker: tuple[str, str] | None = None
    sources: tuple[str, ...] = ()
    receipt_env: str | None = None
    additional_key: str | None = None


@dataclass(frozen=True)
class Target:
    key: str
    version: str
    adapter_repo: str
    merged_repo: str
    # Recorded optimized server receipt and merged-checkpoint identity.
    server_receipt: str
    merged_artifact: str
    # How the campaign built its environment: the 4B used the selected
    # optimized default; the 27B used the BF16 launch environment.
    launch: str
    # (workload, recorded optimized rows, recorded macro pAUROC@20) per split.
    validation: dict[str, tuple[str, str, float]]
    # How the target raises the MXFP8 length bound beyond the recorded 32K.
    long_context: LongContext

    @property
    def merged_path(self) -> Path:
        import json

        return Path(
            json.loads((ROOT / self.merged_artifact).read_text())["destination"]
        )


TARGETS = {
    "4b": Target(
        key="4b",
        version="v0.4",
        adapter_repo="Jazhyc/Gleipnir-4B",
        merged_repo="Jazhyc/Gleipnir-4B-BF16",
        server_receipt="results/qad-v02-optimized01/server.json",
        merged_artifact="results/qad-v02-optimized01/merged_artifact.json",
        launch="optimized",
        validation={
            "id": (
                "results/b200_optimized_id/id02/workload.json",
                "results/qad-v02-optimized01/evaluation/id.jsonl",
                0.9117,
            ),
            "ood": (
                "data/teacher-disagreement-strict-ood01/workload.jsonl",
                "results/qad-v02-optimized01/evaluation/ood.jsonl",
                0.7703,
            ),
        },
        # FastGatedTrimWorker is FastGdnWorker with LongContextWorker, which
        # raises the bound after validating the 1M envelope receipt.
        long_context=LongContext(
            receipt="results/b200_context_million/million01/native.json",
            receipt_sha256="c4f77178894708d3050ad4f70b3d225d47b96a936914725a2c4d8fc94ba2573b",
            worker=(
                "experiments.b200_gdn_fast.worker.FastGdnWorker",
                "experiments.b200_gdn_fast.worker.FastGatedTrimWorker",
            ),
            sources=(
                "experiments/b200_context_million/worker.py",
                "experiments/b200_context_million/envelope.py",
            ),
            receipt_env="GLEIPNIR_LONG_CONTEXT_VALIDATION",
        ),
    ),
    "27b": Target(
        key="27b",
        version="v0.2",
        adapter_repo="Jazhyc/Gleipnir-27B",
        merged_repo="Jazhyc/Gleipnir-27B-BF16",
        server_receipt="results/optimized_gap/opt27b_teacher_eu01/server.json",
        merged_artifact="results/qwen38-27b-s0-01/merged_artifact.json",
        launch="bf16_launch",
        validation={
            "id": (
                "data/optimized_gap/teacher_id_workload.jsonl",
                "results/optimized_gap/opt27b_teacher_id_eu01/id.jsonl",
                0.9444,
            ),
            "ood": (
                "data/optimized_gap/teacher_ood_workload.jsonl",
                "results/optimized_gap/opt27b_teacher_eu01/ood.jsonl",
                0.9508,
            ),
        },
        # QuantFullWorker raises the bound itself when this receipt is bound.
        long_context=LongContext(
            receipt="results/qwen38_27b_serving/mxfp8_envelope01.json",
            receipt_sha256="PENDING",
            additional_key="mxfp8_long_context",
        ),
    ),
}

# Diagnostic sources whose bytes differ between the two recorded stacks. The
# 4B restores its archived bytes at launch (``restore_diagnostic_sources``);
# the 27B was validated with the repository versions, saved at build time.
SHARED_DIAGNOSTIC_SOURCES = [
    "experiments/b200_attention_gdn_serving/fp4_gemm_tune.py",
    "experiments/b200_attention_gdn_serving/attention_fp4_canary.py",
    "experiments/b200_attention_gdn_serving/fp4_fusion_canary.py",
    "experiments/b200_attention_gdn_serving/fp4_prepare_canary.py",
]
