#!/usr/bin/env python3
"""Validate and stage the public Gleipnir PEFT adapter releases."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

from safetensors import safe_open

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.tool_trajectory_monitoring.prompting import (  # noqa: E402
    load_prompt_set,
)

DEFAULT_CONFIG = ROOT / "model_releases/hf/release_config.json"
DEFAULT_OUTPUT = ROOT / "results/huggingface_release"
WEIGHTS_NAME = "adapter_model.safetensors"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def resolve_repo_path(value: str) -> Path:
    path = (ROOT / value).resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError(f"release input escapes repository root: {value}")
    return path


def inspect_adapter(path: Path) -> dict[str, Any]:
    with safe_open(path, framework="pt", device="cpu") as handle:
        keys = list(handle.keys())
        dtypes = sorted({str(handle.get_tensor(key).dtype) for key in keys})
    if not keys:
        raise ValueError(f"adapter has no tensors: {path}")
    if dtypes != ["torch.float32"]:
        raise ValueError(f"release adapter must remain FP32, got {dtypes}: {path}")
    return {
        "bytes": path.stat().st_size,
        "dtypes": dtypes,
        "sha256": sha256_file(path),
        "tensor_count": len(keys),
    }


def validate_adapter_directory(
    path: Path, model: dict[str, Any], expected_sha256: str
) -> dict[str, Any]:
    config_path = path / "adapter_config.json"
    weights_path = path / WEIGHTS_NAME
    if not config_path.is_file() or not weights_path.is_file():
        raise FileNotFoundError(f"incomplete adapter directory: {path}")
    adapter_config = load_json(config_path)
    if adapter_config.get("base_model_name_or_path") != model["base_model"]:
        raise ValueError(f"base-model identity drifted in {config_path}")
    if adapter_config.get("peft_type") != "LORA" or int(
        adapter_config.get("r", -1)
    ) != 128:
        raise ValueError(f"expected a rank-128 LoRA in {config_path}")
    details = inspect_adapter(weights_path)
    if details["sha256"] != expected_sha256:
        raise ValueError(
            f"adapter checksum drifted for {path}: "
            f"{details['sha256']} != {expected_sha256}"
        )
    return details


def validate_model(
    name: str, model: dict[str, Any], common: dict[str, Any]
) -> dict[str, Any]:
    source_dir = resolve_repo_path(str(model["source_dir"]))
    serving_dir = resolve_repo_path(str(model["serving_dir"]))
    card_path = resolve_repo_path(str(model["card"]))
    if not card_path.is_file():
        raise FileNotFoundError(card_path)
    card = card_path.read_text(encoding="utf-8")
    if f"# {model['display_name']}" not in card:
        raise ValueError(f"model card title drifted for {name}")
    if f"base_model: {model['base_model']}" not in card:
        raise ValueError(f"model card base identity drifted for {name}")

    source = validate_adapter_directory(
        source_dir, model, str(model["source_sha256"])
    )
    serving = validate_adapter_directory(
        serving_dir, model, str(model["serving_sha256"])
    )
    if source["tensor_count"] != serving["tensor_count"]:
        raise ValueError(f"causal and serving tensor counts differ for {name}")

    rebase_path = serving_dir / "rebase_manifest.json"
    rebase = load_json(rebase_path)
    if (
        rebase.get("source_sha256") != model["source_sha256"]
        or rebase.get("destination_sha256") != model["serving_sha256"]
    ):
        raise ValueError(f"serving rebase provenance drifted for {name}")

    metadata_path = source_dir / "training_metadata.json"
    if sha256_file(metadata_path) != model["training_metadata_sha256"]:
        raise ValueError(f"training metadata checksum drifted for {name}")
    metadata = load_json(metadata_path)
    if metadata.get("model") != model["base_model"]:
        raise ValueError(f"training base identity drifted for {name}")

    return {
        "name": name,
        "model": model,
        "source_dir": source_dir,
        "serving_dir": serving_dir,
        "card_path": card_path,
        "source": source,
        "serving": serving,
        "rebase": rebase,
        "metadata": metadata,
        "common": common,
    }


def prompt_contract(common: dict[str, Any]) -> tuple[Path, dict[str, Any]]:
    prompt = common["prompt"]
    manifest_path = resolve_repo_path(str(prompt["manifest"]))
    instruction_path = resolve_repo_path(str(prompt["student_instruction"]))
    prompt_set = load_prompt_set(manifest_path.parent)
    if prompt_set.student.template_sha256 != prompt["template_sha256"]:
        raise ValueError("student prompt template checksum drifted")
    manifest = load_json(manifest_path)
    return instruction_path, {
        "decision_prefix": manifest["decision_prefix"],
        "negative_label": manifest["negative_label"],
        "positive_label": manifest["positive_label"],
        "prompt_role": "student",
        "prompt_set_id": manifest["prompt_set_id"],
        "student_instruction_file": "student_prompt.txt",
        "student_instruction_sha256": sha256_file(instruction_path),
        "template_sha256": prompt["template_sha256"],
        "trajectory_close": manifest["trajectory_close"],
        "trajectory_open": manifest["trajectory_open"],
    }


def training_summary(spec: dict[str, Any]) -> dict[str, Any]:
    model = spec["model"]
    metadata = spec["metadata"]
    common = spec["common"]
    return {
        "adapter": {
            "causal_master": spec["source"],
            "serving_copy": spec["serving"],
        },
        "base_model": {
            "id": model["base_model"],
            "revision": model["base_revision"],
        },
        "evaluation": {
            **common["evaluation"],
            **model["evaluation"],
        },
        "source_commit": metadata.get("commit"),
        "training": {
            **common["training"],
            "peak_cuda_memory_allocated_bytes": metadata.get(
                "peak_cuda_memory_allocated_bytes"
            ),
            "runtime_seconds": metadata.get("train_metrics", {}).get(
                "train_runtime"
            ),
            "trainable_parameters": metadata.get("trainable_parameters"),
        },
    }


def staged_files(path: Path) -> dict[str, dict[str, Any]]:
    files = {}
    for item in sorted(path.rglob("*")):
        if item.is_file() and item.name != "release_manifest.json":
            relative = item.relative_to(path).as_posix()
            files[relative] = {
                "bytes": item.stat().st_size,
                "sha256": sha256_file(item),
            }
    return files


def stage_model(
    spec: dict[str, Any],
    output_root: Path,
    instruction_path: Path,
    contract: dict[str, Any],
) -> Path:
    model = spec["model"]
    destination = output_root / str(model["repo_name"])
    if destination.exists():
        raise FileExistsError(
            f"refusing to overwrite staged release: {destination}; "
            "choose a new --output-dir"
        )

    temporary = Path(
        tempfile.mkdtemp(prefix=f".{model['repo_name']}-", dir=output_root)
    )
    try:
        shutil.copy2(spec["card_path"], temporary / "README.md")
        shutil.copy2(spec["source_dir"] / "adapter_config.json", temporary)
        shutil.copy2(spec["source_dir"] / WEIGHTS_NAME, temporary)
        shutil.copy2(instruction_path, temporary / "student_prompt.txt")
        write_json(temporary / "prompt_contract.json", contract)
        write_json(temporary / "training_summary.json", training_summary(spec))

        serving = temporary / "vllm"
        serving.mkdir()
        shutil.copy2(spec["serving_dir"] / "adapter_config.json", serving)
        shutil.copy2(spec["serving_dir"] / WEIGHTS_NAME, serving)
        write_json(
            serving / "rebase_manifest.json",
            {
                "destination_prefix": spec["rebase"]["destination_prefix"],
                "destination_sha256": spec["rebase"]["destination_sha256"],
                "format": spec["rebase"]["format"],
                "source_prefix": spec["rebase"]["source_prefix"],
                "source_sha256": spec["rebase"]["source_sha256"],
                "tensor_count": spec["rebase"]["tensor_count"],
            },
        )
        write_json(
            temporary / "release_manifest.json",
            {
                "base_model": model["base_model"],
                "base_revision": model["base_revision"],
                "display_name": model["display_name"],
                "files": staged_files(temporary),
                "format_version": spec["common"]["format_version"],
                "job_name": model["job_name"],
                "repo_name": model["repo_name"],
            },
        )
        temporary.replace(destination)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return destination


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--model", choices=("all", "4b", "9b"), default="all"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_json(args.config.resolve())
    if config.get("format_version") != 1:
        raise ValueError("unsupported release config version")
    selected = list(config["models"]) if args.model == "all" else [args.model]
    specs = [validate_model(name, config["models"][name], config) for name in selected]
    instruction_path, contract = prompt_contract(config)

    output_root = args.output_dir.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    conflicts = [
        output_root / str(spec["model"]["repo_name"])
        for spec in specs
        if (output_root / str(spec["model"]["repo_name"])).exists()
    ]
    if conflicts:
        raise FileExistsError(
            "refusing to overwrite staged releases: "
            + ", ".join(path.as_posix() for path in conflicts)
        )

    destinations = [
        stage_model(spec, output_root, instruction_path, contract) for spec in specs
    ]
    print(
        json.dumps(
            {"staged": [path.as_posix() for path in destinations]}, sort_keys=True
        )
    )


if __name__ == "__main__":
    main()
