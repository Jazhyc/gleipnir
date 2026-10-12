"""Resume the completed QAD master after the causal-wrapper export failure."""

import json
import subprocess
import time
from pathlib import Path

import yaml

from experiments.b200_vllm031.runtime import candidate_environment
from experiments.quantization_recovery.run import check
from gleipnir.adapters.merge import merge_checkpoint
from gleipnir.adapters.rebase import rebase_adapter
from gleipnir.data.monitoring import file_hash, write_json


def main():
    root = Path.cwd()
    config = yaml.safe_load(
        (root / "experiments/quantization_recovery/config.yaml").read_text()
    )
    check(config)
    output = root / "results" / config["experiment"]
    recovery = output / "recovery"
    recovery.mkdir(exist_ok=True)
    contract = json.loads((output / "execution_contract.json").read_text())
    for name, expected in contract["source_bindings"].items():
        if name.endswith("/run.py"):
            path = recovery / "original_run.py"
        elif name == "experiments/quantization_recovery/README.md":
            path = recovery / "original_README.md"
        else:
            path = root / name
        if file_hash(path) != expected:
            raise ValueError(f"completed training source drift: {name}")
    failed = json.loads((output / "all_status.json").read_text())
    if (
        failed["status"] != "failed"
        or "adapter/base key layout mismatch" not in failed["error"]
    ):
        raise ValueError("unexpected recovery failure scope")
    write_json(recovery / "original_failure.json", failed)
    complete = json.loads((output / "training_complete.json").read_text())
    if complete["updates"] != 245 or complete["rows"] != 7836:
        raise ValueError("incomplete QAD training")
    master = output / "causal_adapter"
    if file_hash(master / "adapter_model.safetensors") != complete["master_sha256"]:
        raise ValueError("completed FP32 master drift")
    reference = json.loads((output / "parity_reference.json").read_text())
    if reference["master_sha256"] != complete["master_sha256"]:
        raise ValueError("master reference identity drift")
    destination = Path(config["teacher_merge"])
    old = json.loads((output / "replaced_merge_manifest.json").read_text())
    for name, expected in old["source_files_sha256"].items():
        if file_hash(root / config["base_model"] / name) != expected:
            raise ValueError("pinned base identity drift")
    try:
        rebased = rebase_adapter(master, output / "merged_adapter")
        if destination.exists():
            report = json.loads((output / "merged_artifact.json").read_text())
            if report["adapter_sha256"] != rebased["destination_sha256"]:
                raise ValueError("existing merge belongs to a different adapter")
            for name, expected in report["files_sha256"].items():
                if file_hash(destination / name) != expected:
                    raise ValueError("existing candidate merge drift")
        else:
            report = merge_checkpoint(
                root / config["base_model"],
                output / "merged_adapter",
                destination,
                expected_adapter_sha256=rebased["destination_sha256"],
                model_id="Qwen/Qwen3.5-4B",
                revision="851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
            )
            write_json(output / "merged_artifact.json", report)
        write_json(
            recovery / "resume.json",
            {
                "master_sha256": complete["master_sha256"],
                "rebase": rebased,
                "intervention": "tensor_key_rename_only_FP32_values_preserved",
                "training_reused": True,
                "reference_reused": True,
                "recovery_source_sha256": file_hash(Path(__file__)),
                "fixed_driver_sha256": file_hash(
                    root / "experiments/quantization_recovery/run.py"
                ),
            },
        )
        subprocess.run(
            [
                config["serving_python"],
                "-m",
                "experiments.quantization_recovery.evaluate",
            ],
            cwd=root,
            env=candidate_environment(root),
            check=True,
        )
        write_json(
            recovery / "status.json",
            {"status": "complete", "finished_at_unix": time.time()},
        )
    except BaseException as error:
        write_json(
            recovery / "status.json",
            {"status": "failed", "error": str(error), "finished_at_unix": time.time()},
        )
        raise


if __name__ == "__main__":
    main()
