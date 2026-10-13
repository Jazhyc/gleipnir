"""Restore the original FP4-trained adapter and score frozen OOD in BF16 only."""

import json
import shutil
import subprocess
import time
from collections import Counter
from pathlib import Path

import yaml

from experiments.b200_vllm031.runtime import candidate_environment
from gleipnir.adapters.merge import merge_checkpoint
from gleipnir.campaigns.runtime import training_environment
from gleipnir.data.monitoring import file_hash, read_rows, write_json


def main():
    root = Path.cwd()
    config_path = root / "experiments/filtered_fp4_ood/config.yaml"
    config = yaml.safe_load(config_path.read_text())
    output = root / "results" / config["experiment"]
    output.mkdir(parents=True, exist_ok=True)
    status_path = output / "status.json"
    if status_path.exists():
        raise ValueError("preserve existing OOD campaign receipts")

    def status(phase, **fields):
        write_json(status_path, {"phase": phase, "at_unix": time.time(), **fields})
        print(phase, json.dumps(fields), flush=True)

    try:
        status("preparing")
        if config["conditions"] != ["bf16"]:
            raise ValueError("the user selected BF16-only evaluation")
        gpu = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"], text=True
        ).strip()
        if gpu != config["gpu_uuid"]:
            raise ValueError("GPU identity drift")
        bindings = {
            str(p.relative_to(root)): file_hash(p)
            for p in [
                config_path,
                *sorted(config_path.parent.glob("*.py")),
                root / "experiments/quantization_recovery/run.py",
                root / "src/gleipnir/campaigns/monitoring/evaluation.py",
            ]
        }
        for spec in config["inputs"].values():
            bindings[spec["path"]] = spec["sha256"]
        bindings.update(config["reference_source"])
        bindings.update(
            {
                config["cached_references"][name]: sha
                for name, sha in config["reference_sha256"].items()
            }
        )
        bindings[config["initial_adapter"] + "/adapter_model.safetensors"] = config[
            "initial_adapter_sha256"
        ]
        bindings[config["serving_adapter"] + "/adapter_model.safetensors"] = config[
            "serving_adapter_sha256"
        ]
        for path, expected in bindings.items():
            if file_hash(root / path) != expected:
                raise ValueError(f"input/source identity drift: {path}")
        canonical, workload = (
            read_rows(root / config["canonical_ood"]),
            read_rows(root / config["ood_workload"]),
        )
        from experiments.teacher_disagreement_ood.prepare import COUNTS

        if (
            len(canonical) != 6395
            or dict(Counter(r["metadata"]["source_dataset"] for r in canonical))
            != COUNTS
        ):
            raise ValueError("frozen OOD population drift")
        if len(workload) != 6395 or any(
            a["id"] != b["id"] for a, b in zip(canonical, workload, strict=True)
        ):
            raise ValueError("rendered OOD coverage drift")
        trained = read_rows(
            root / "data/filtered-fp4-fa4-baseline01/strict_filter/training.jsonl"
        )
        seen = {row["trajectory_sha256"] for row in trained}
        if any(row["metadata"]["trajectory_sha256"] in seen for row in canonical):
            raise ValueError("exact training/OOD trajectory overlap")
        write_json(
            output / "execution_contract.json",
            {
                "config": config,
                "files_sha256": bindings,
                "rows": 6395,
                "source_counts": COUNTS,
                "selection": "fixed_original_adapter_no_ood_tuning",
            },
        )
        for path in [config_path, *sorted(config_path.parent.glob("*.py"))]:
            destination = output / "source_archive" / path.relative_to(root)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, destination)
        status("merge")
        subprocess.run(
            [config["training_python"], "-m", "experiments.filtered_fp4_ood.merge"],
            cwd=root,
            env={
                **training_environment(root, config["experiment"], native_fp4_mlp=True),
                "PYTHONPATH": f"{root}:{root}/src",
            },
            check=True,
            timeout=3600,
        )
        status("evaluate")
        subprocess.run(
            [config["serving_python"], "-m", "experiments.filtered_fp4_ood.evaluate"],
            cwd=root,
            env=candidate_environment(root),
            check=True,
            timeout=7200,
        )
        status("complete")
    except Exception as exc:
        status("failed", error=f"{type(exc).__name__}: {exc}")
        raise


def restore_merge(root: Path, config, output: Path):
    """Replace only the owned QAD64 slot and reproduce original merged weights."""
    slot = Path(config["teacher_merge"])
    if (
        slot.is_symlink()
        or str(slot) != "/tmp/gleipnir-merged/filtered-fp4-fa4-baseline"
    ):
        raise ValueError("unsafe owned merge slot")
    old = json.loads((slot / "merge_manifest.json").read_text())
    previous = json.loads((root / config["merge_predecessor"]).read_text())
    if old["adapter_sha256"] != previous["adapter_sha256"]:
        raise ValueError("merge predecessor ownership drift")
    for name, expected in old["files_sha256"].items():
        if file_hash(slot / name) != expected:
            raise ValueError("merge predecessor file drift")
    write_json(output / "replaced_merge_manifest.json", old)
    shutil.rmtree(slot)
    actual = merge_checkpoint(
        root / config["base_model"],
        root / config["serving_adapter"],
        slot,
        expected_adapter_sha256=config["serving_adapter_sha256"],
        model_id="Qwen/Qwen3.5-4B",
        revision="851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
    )
    write_json(output / "merged_artifact.json", actual)
    original = json.loads((root / config["original_merge"]).read_text())
    for name, expected in original["files_sha256"].items():
        if (
            name != "merge_manifest.json"
            and actual["files_sha256"].get(name) != expected
        ):
            raise ValueError(f"original merge reconstruction differs: {name}")
    write_json(
        output / "merge_identity.json",
        {
            "passed": True,
            "original_files_verified": True,
            "master_sha256": config["initial_adapter_sha256"],
            "serving_adapter_sha256": config["serving_adapter_sha256"],
            "policy": "reuse_original_inference_export_and_reproduce_ID_merge",
        },
    )


if __name__ == "__main__":
    main()
