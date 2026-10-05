"""Checksum-bound selection of independently validated Meta-inspired components."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from experiments.b200_mxfp8_fused.training_screen import accept_native
from gleipnir.monitoring_systems_screen import sha256_file
from gleipnir.nvidia_mxfp8_meta_training import validate_options

ROOT = Path(__file__).resolve().parents[2]


def bound_receipt(config: dict, name: str) -> dict:
    path = ROOT / config[f"{name}_reference"]
    if sha256_file(path) != config[f"{name}_reference_sha256"]:
        raise ValueError(f"{name} receipt checksum drift")
    return json.loads(path.read_text())


def accept_meta(native: dict, config: dict) -> dict:
    """Bind scheduling and producer receipts to the precise selected options."""
    norm, variant = validate_options(config["meta_attention_options"])
    acceptance = accept_native(native, square=True)
    tested = bound_receipt(config, "variant")
    expected = asdict(variant)
    if (
        tested.get("status") != "complete"
        or tested.get("spec") != expected
        or tested.get("canary_sha256") != config["native_reference_sha256"]
    ):
        raise ValueError("native receipt does not validate selected variant")
    if norm:
        producer = bound_receipt(config, "producer")
        if producer.get("status") != "complete" or producer.get("spec") != expected:
            raise ValueError("producer does not validate selected combined variant")
        checks = producer.get("producer_checks", [])
        if len(checks) != 2 or {x["heads"] for x in checks} != {16, 4}:
            raise ValueError("incomplete producer checks")
        if any(
            x["code_agreement"] < 0.999
            or x["native_scales_bitexact"] != [True, True, True]
            or not 0 <= x["backward_relative_l2"] <= 0.01
            for x in checks
        ):
            raise ValueError("producer arithmetic failed")
        comparison = producer["attention_comparison"]
        errors = [
            comparison["forward_relative_l2"],
            *comparison["gradient_relative_l2"],
        ]
        if comparison["finite"] is not True or not all(0 <= x <= 0.01 for x in errors):
            raise ValueError("producer integration failed")
        if not all(
            producer[g]["passed"] is True for g in ("isolation", "graph_replay")
        ):
            raise ValueError("producer isolation/replay failed")
    return {**acceptance, "selected_options": {"norm_rope": norm, **expected}}


def main() -> None:
    from experiments.b200_nvidia_mxfp8_varlen.training_screen import main as shared_main

    shared_main(Path(__file__).with_name("training_config.yaml"))


if __name__ == "__main__":
    main()
