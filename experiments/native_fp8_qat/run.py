"""Automatic native-FP8 QAT, FP32 export, owned merge and frozen ID evaluation."""

import importlib
import json
import shutil
import subprocess
import time
from pathlib import Path

import yaml

from experiments.native_fp8_qat.generate import generate
from experiments.quantization_recovery.run import check
from gleipnir.adapters.merge import merge_checkpoint
from gleipnir.adapters.rebase import rebase_adapter
from gleipnir.campaigns.runtime import training_environment
from gleipnir.data.monitoring import file_hash, write_json


def stage_environment(root: Path, environment: dict[str, str]) -> dict[str, str]:
    """Archived script entrypoints need the project root for experiment imports."""
    return {
        **environment,
        "PYTHONPATH": str(root) + ":" + environment.get("PYTHONPATH", ""),
    }


def main(config_path: str = "experiments/native_fp8_qat/config.yaml"):
    root = Path.cwd()
    path = root / config_path
    config = yaml.safe_load(path.read_text())
    output = root / "results" / config["experiment"]
    output.mkdir(parents=True, exist_ok=True)
    if (output / "execution_contract.json").exists():
        raise ValueError("preserve existing campaign receipts")
    status_path = output / "status.json"

    def status(phase, **fields):
        write_json(status_path, {"phase": phase, "at_unix": time.time(), **fields})
        print(phase, json.dumps(fields), flush=True)

    try:
        status("preparing")
        check(config)
        teacher = root / "data" / config["teacher_experiment"]
        teacher_cache = root / config.get(
            "teacher_cache", str((teacher / "teacher.jsonl").relative_to(root))
        )
        if file_hash(teacher_cache) != config["teacher_sha256"]:
            raise ValueError("frozen teacher drift")
        generator = generate
        if config.get("stage_generator"):
            generator = importlib.import_module(config["stage_generator"]).generate
        stages = generator(
            root,
            output,
            config_path=config_path,
            runtime_module=config.get(
                "runtime_module", "experiments.native_fp8_qat.runtime"
            ),
            native_attention_backward=config.get("native_attention_backward", False),
        )
        sources = [
            *stages.values(),
            path,
            *sorted((root / "experiments/native_fp8_qat").glob("*.py")),
            root / "src/gleipnir/training/qat_optimization.py",
            root / "src/gleipnir/training/quantization_recovery.py",
            root / "src/gleipnir/kernels/fp4/decode.py",
            *sorted(path.parent.glob("*.py")),
            *[root / source for source in config.get("extra_sources", [])],
        ]
        bindings = {str(p.relative_to(root)): file_hash(p) for p in sources}
        write_json(
            output / "execution_contract.json",
            {
                "config": config,
                "sources": bindings,
                "workload_sha256": file_hash(teacher / "workload.jsonl"),
                "historical_reproduction_gate": "new_variant_prior_failure_preserved",
            },
        )
        for source in sources:
            target = output / "source_archive" / source.relative_to(root)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
        env = training_environment(root, config["experiment"], native_fp4_mlp=True)

        def run_stage(name, executable, environment):
            for source, checksum in bindings.items():
                if file_hash(root / source) != checksum:
                    raise ValueError(f"campaign source drift: {source}")
            status(name)
            subprocess.run(
                [executable, str(stages[name])],
                cwd=root,
                env=stage_environment(root, environment),
                check=True,
                timeout=7200,
            )

        from experiments.b200_vllm031.runtime import candidate_environment

        if "initial-serving" in stages:
            run_stage(
                "initial-serving", config["serving_python"], candidate_environment(root)
            )
        run_stage("train", config["training_python"], env)
        run_stage("reference", config["training_python"], env)
        complete = json.loads((output / "training_complete.json").read_text())
        master = output / "causal_adapter"
        if (
            complete["updates"] != config["updates"]
            or file_hash(master / "adapter_model.safetensors")
            != complete["master_sha256"]
        ):
            raise ValueError("incomplete or changed native FP8 master")
        status("merge")
        rebased = rebase_adapter(master, output / "merged_adapter")
        destination = Path(config["teacher_merge"])
        if (
            destination.is_symlink()
            or str(destination) != "/tmp/gleipnir-merged/filtered-fp4-fa4-baseline"
        ):
            raise ValueError("unsafe ephemeral merge slot")
        old = json.loads((destination / "merge_manifest.json").read_text())
        previous = json.loads(
            (
                root
                / config.get(
                    "merge_predecessor",
                    "results/quantization-recovery01/merged_artifact.json",
                )
            ).read_text()
        )
        if old["adapter_sha256"] != previous["adapter_sha256"]:
            raise ValueError("ephemeral merge predecessor ownership changed")
        for name, expected in old["files_sha256"].items():
            if file_hash(destination / name) != expected:
                raise ValueError("ephemeral predecessor file drift")
        write_json(output / "replaced_merge_manifest.json", old)
        shutil.rmtree(destination)
        report = merge_checkpoint(
            root / config["base_model"],
            output / "merged_adapter",
            destination,
            expected_adapter_sha256=rebased["destination_sha256"],
            model_id="Qwen/Qwen3.5-4B",
            revision="851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
        )
        write_json(output / "merged_artifact.json", report)
        run_stage("evaluate", config["serving_python"], candidate_environment(root))
        result = json.loads((output / "summary.json").read_text())
        baseline = json.loads((root / config["quality_reference"]).read_text())
        score = (
            sum(
                r["metrics"]["metrics"]["macro"]["macro"]["pauroc_at_20"]
                for r in result["results"]
            )
            / 2
        )
        # Metric access below is validated against the saved canonical schema.
        reference_score = (
            sum(
                r["metrics"]["metrics"]["macro"]["macro"]["pauroc_at_20"]
                for r in baseline["results"]
            )
            / 2
        )
        write_json(
            output / "quality_comparison.json",
            {
                "native_macro_pauc": score,
                "reference_macro_pauc": reference_score,
                "difference": score - reference_score,
                "descriptive_quality_gate_passed": score
                >= reference_score - config["quality_max_macro_pauc_drop"],
                "promotion": False,
                "selection": "fixed_final_epoch",
            },
        )
        status("complete")
    except Exception as exc:
        status("failed", error=f"{type(exc).__name__}: {exc}")
        raise


if __name__ == "__main__":
    main()
