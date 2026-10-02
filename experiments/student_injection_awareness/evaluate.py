"""Parity-gated persistent-vLLM evaluation of both matched student variants."""

from __future__ import annotations

import argparse
import json

import yaml

from experiments.student_injection_awareness.prepare import (
    CONFIG,
    DATA,
    OUTPUT,
    VARIANTS,
    file_hash,
    templates,
)
from gleipnir.monitoring_campaign_evaluation import (
    EvaluationContext,
)
from gleipnir.monitoring_campaign_evaluation import (
    canary as campaign_canary,
)
from gleipnir.monitoring_campaign_evaluation import (
    evaluation_contract as evaluation_contract,
)
from gleipnir.monitoring_campaign_evaluation import (
    reference as campaign_reference,
)
from gleipnir.monitoring_campaign_evaluation import render as render
from gleipnir.monitoring_campaign_evaluation import (
    serving as campaign_serving,
)


def context() -> EvaluationContext:
    return EvaluationContext(
        DATA, OUTPUT, VARIANTS, {v: t.template_sha256 for v, t in templates().items()}
    )


def canary(variant: str, count: int) -> list[dict]:
    return campaign_canary(context(), variant, count)


def reference(config: dict, size: str) -> None:
    campaign_reference(context(), config, size)


def serving(config: dict, size: str, gdn_prefill_backend: str = "flashinfer") -> None:
    campaign_serving(context(), config, size, gdn_prefill_backend)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", choices=("4b", "9b"), required=True)
    parser.add_argument("--backend", choices=("reference", "vllm"), required=True)
    parser.add_argument(
        "--gdn-prefill-backend", choices=("flashinfer", "triton"), default="flashinfer"
    )
    args = parser.parse_args()
    config = yaml.safe_load(CONFIG.read_text())
    manifest = json.loads((DATA / "manifest.json").read_text())
    if manifest["config_sha256"] != file_hash(CONFIG):
        raise ValueError("prepared evaluation config drift")
    if args.backend == "reference":
        reference(config, args.size)
    else:
        serving(config, args.size, args.gdn_prefill_backend)


if __name__ == "__main__":
    main()
