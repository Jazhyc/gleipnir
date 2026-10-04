"""Audit collected weights and every score before authorized GPU shutdown."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from jinja2.sandbox import ImmutableSandboxedEnvironment
from safetensors import safe_open

from experiments.monitor_injection_augmentation_9b.evaluate import (
    DEST,
    adapter_paths,
    identity,
    populations,
)
from experiments.monitor_injection_augmentation_9b.prepare import OUTPUT, configuration
from experiments.monitor_injection_augmentation_9b.train import validate_metadata
from gleipnir.metrics import normalized_partial_auroc
from gleipnir.monitoring_campaign_data import digest, file_hash, read_rows, write_json
from gleipnir.monitoring_scoring import completed_predictions
from gleipnir.qwen35_adapter_rebase import rebase_key


def main() -> None:
    ident = identity()
    paths = adapter_paths()
    metadata = json.loads((paths["master"] / "training_metadata.json").read_text())
    validate_metadata(metadata, configuration())
    parity = json.loads((DEST / "serving_parity.json").read_text())
    if parity["identity"] != ident or set(parity["cells"]) != {"base", "augmented"}:
        raise ValueError("collection parity identity drift")
    for cell in parity["cells"].values():
        if (
            not cell["passed"]
            or cell["correlation"] < 0.99
            or cell["mean_absolute_difference"] > 0.02
        ):
            raise ValueError("collection parity gate failed")
    if (
        min(
            parity["cells"]["augmented"][key]
            for key in ("adapter_effect", "reference_adapter_effect")
        )
        < 1e-6
    ):
        raise ValueError("adapter effect missing")
    with (
        safe_open(
            str(paths["master"] / "adapter_model.safetensors"), framework="np"
        ) as master,
        safe_open(
            str(paths["serving"] / "adapter_model.safetensors"), framework="np"
        ) as served,
    ):
        keys = list(master.keys())
        if len(keys) != 256 or set(map(rebase_key, keys)) != set(served.keys()):
            raise ValueError("collected adapter layout drift")
        for key in keys:
            a, b = master.get_tensor(key), served.get_tensor(rebase_key(key))
            if (
                a.dtype != np.float32
                or b.dtype != np.float32
                or not np.isfinite(a).all()
                or not np.array_equal(a, b)
            ):
                raise ValueError("master/serving FP32 tensor integrity failed")
    renderer = ImmutableSandboxedEnvironment().from_string(
        (paths["master"] / "chat_template.jinja").read_text()
    )
    files, scored = {}, {}
    for name, input_path in populations().items():
        inputs = read_rows(input_path)
        output_path = DEST / "augmented" / (name + ".jsonl")
        rows = completed_predictions(output_path, inputs, ident)
        by_id = {r["id"]: r for r in rows}
        contract = json.loads(output_path.with_suffix(".contract.json").read_text())
        if contract["decision_ids"] != [15, 16] or contract["truncated"]:
            raise ValueError("binary decision or truncation contract drift")
        for source in inputs:
            row = by_id[source["id"]]
            prompt = (
                renderer.render(
                    messages=[{"role": "user", "content": source["prompt"]}],
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
                + "Prediction:"
            )
            if (
                digest(prompt) != row["prompt_sha256"]
                or row["completion_tokens"] != 1
                or row["generated_token_id"] not in (15, 16)
                or row["prompt_tokens"] >= 32768
            ):
                raise ValueError("collected rendered prompt/one-token/context drift")
            if any(
                row["raw_returned_logprobs"][k] != value
                for k, value in row["raw_decision_logprobs"].items()
            ):
                raise ValueError("raw returned/decision logprobs disagree")
        scored[name] = rows
        files[name] = {
            "rows": len(rows),
            "sha256": file_hash(output_path),
            "maximum_tokens": max(r["prompt_tokens"] for r in rows),
        }
        print("collection_verified", name, len(rows), flush=True)
    audit = json.loads((DEST / "completion_audit.json").read_text())
    if (
        sum(map(len, scored.values())) != 15138
        or audit["fresh_scores"] != 15138
        or audit["summary_sha256"] != file_hash(DEST / "summary.json")
        or any(audit["files_sha256"][name] != v["sha256"] for name, v in files.items())
    ):
        raise ValueError("collected report checksum/coverage drift")
    partials = {}
    for name in ("id", "synthetic_id"):
        rows = scored[name]
        values = []
        for source in sorted({r["source_dataset"] for r in rows}):
            subset = [r for r in rows if r["source_dataset"] == source]
            values.append(
                normalized_partial_auroc(
                    [r["label"] for r in subset], [r["score"] for r in subset]
                )
            )
        partials[name] = float(np.mean(values))
    write_json(
        DEST / "collection_verification.json",
        {
            "passed": True,
            "verified_unix": time.time(),
            "identity": ident,
            "weights": "256 finite FP32 master tensors, exact rebased serving equality",
            "all_rendered_prompt_hashes_match": True,
            "fresh_scores": 15138,
            "files": files,
            "summary_sha256": file_hash(DEST / "summary.json"),
            "project_partial_auroc_source_macro": partials,
            "original_adapter_rescored": False,
        "auditor_sha256": file_hash(Path(__file__)),
            "shutdown_authorized_after_artifact_inventory_verification": True,
        },
    )
    print("collection_complete monitor_injection_augmentation_9b", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        write_json(
            OUTPUT / f"failure_collection_{time.time_ns()}.json",
            {"type": type(exc).__name__, "message": str(exc)},
        )
        raise
