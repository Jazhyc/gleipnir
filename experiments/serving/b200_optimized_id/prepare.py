"""Bind the latest full-trained master and canonical same-adapter ID control."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from experiments.b200_inference_benchmark.run import ROOT, sha, write

EXPERIMENT = Path(__file__).parent
BINDING = ROOT / "data/b200_optimized_id/binding.json"


def source_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def align_inputs(rows: list[dict], control: list[dict]) -> None:
    """Reject reordered/changed prompts, labels, source lineage or duplicated IDs."""
    if len(rows) != len(control) or len({r["id"] for r in rows}) != len(rows):
        raise ValueError("ID population or uniqueness changed")
    for row, old in zip(rows, control, strict=True):
        metadata = row["metadata"]
        content_hash = hashlib.sha256(row["prompt"].encode()).hexdigest()
        if (
            row["id"] != old["id"]
            or metadata["ground_truth"] != old["label"]
            or metadata["source_dataset"] != old["source"]
            or content_hash != metadata["rendered_prompt_sha256"]
            or content_hash != old["source_prompt_sha256"]
        ):
            raise ValueError(f"ID source/prompt/label/order drift: {row['id']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.json")
    args = parser.parse_args()
    if Path(args.config).name != args.config or not args.config.endswith(".json"):
        raise ValueError("config must be an experiment JSON filename")
    config = EXPERIMENT / args.config
    settings = json.loads(config.read_text())
    binding_path = ROOT / settings.get("binding", str(BINDING.relative_to(ROOT)))
    for field in (
        "input",
        "input_manifest",
        "master",
        "serving_adapter",
        "reference_predictions",
    ):
        if sha(ROOT / settings[field]) != settings[field + "_sha256"]:
            raise ValueError(f"frozen artifact changed: {field}")
    summary = json.loads((ROOT / settings["training_summary"]).read_text())
    if (
        summary["status"] != "complete"
        or summary["updates"] != 272
        or summary["candidate_master_sha256"] != settings["master_sha256"]
        or summary["candidate_predictions_sha256"]
        != settings["reference_predictions_sha256"]
    ):
        raise ValueError("latest full-epoch adapter/control identity changed")
    rebase = json.loads((ROOT / settings["rebase"]).read_text())
    if (
        rebase["source_sha256"] != settings["master_sha256"]
        or rebase["destination_sha256"] != settings["serving_adapter_sha256"]
    ):
        raise ValueError("master-to-serving rebase identity changed")
    rows = source_rows(ROOT / settings["input"])
    old = source_rows(ROOT / settings["reference_predictions"])
    align_inputs(rows, old)
    if len(rows) != settings["rows"]:
        raise ValueError("canonical row count changed")
    files = [
        settings[k]
        for k in (
            "input",
            "input_manifest",
            "training_summary",
            "rebase",
            "reference_predictions",
            "reference_result",
            "master_canary",
            "serving_canary",
        )
    ]
    grouped = settings.get("grouped_control")
    if grouped is not None:
        for filename, expected in grouped["files_sha256"].items():
            if sha(ROOT / filename) != expected:
                raise ValueError(f"grouped control changed: {filename}")
            files.append(filename)
        files.append(settings["retired_parent"])
        control_settings = json.loads(
            (ROOT / grouped["directory"] / "settings.json").read_text()
        )
        exclusions = {"admission", "binding", "grouped_control", "retired_parent"}
        if {k: v for k, v in control_settings.items() if k not in exclusions} != {
            k: v for k, v in settings.items() if k not in exclusions
        }:
            raise ValueError("continuous comparison changed more than client admission")
    binding = {
        "config_sha256": sha(config),
        "files": {name: sha(ROOT / name) for name in files},
        "master_sha256": settings["master_sha256"],
        "serving_adapter_sha256": settings["serving_adapter_sha256"],
        "selection": "existing fixed final 272-update checkpoint; backend drift only",
        "rows": len(rows),
        "prompt_tokens": sum(r["prompt_tokens"] for r in old),
    }
    if binding_path.exists() and json.loads(binding_path.read_text()) != binding:
        raise ValueError("existing frozen ID binding changed")
    write(binding_path, binding)
    print("id_control_bound", len(rows), binding["prompt_tokens"])


if __name__ == "__main__":
    main()
