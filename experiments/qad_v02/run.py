"""QAD for Gleipnir 4B v0.2: teacher cache, training, then optimized ID/OOD.

Phases (each gated, statuses in ``results/<experiment>/status.json``):

1. ``baseline``: v0.2's optimized ID score (fresh merge, optimized parity
   gate), the comparison the earlier optimized-gap run did not measure.
2. ``teacher``: v0.2's own BF16 scores on the training rows (`teacher.py`).
3. ``train`` / ``reference``: the frozen QAD stages through the selected lean
   arithmetic (`generate.py`), from the exact rank-384 v0.2 master, then the
   FP32 master canary.
4. ``evaluate``: the QAD master in campaign layout, scored through the same
   optimized path on ID and OOD (`optimized_gap.run`).
"""

import copy
import json
import shutil
import subprocess
import time
from pathlib import Path

import yaml

from experiments.optimized_gap.run import score
from experiments.qad_v02.generate import config as load_config
from experiments.qad_v02.generate import generate
from experiments.quantization_recovery.run import check
from gleipnir.adapters.rebase import rebase_adapter
from gleipnir.campaigns.monitoring.contract import Campaign
from gleipnir.campaigns.runtime import training_environment
from gleipnir.data.monitoring import digest, file_hash, read_rows, write_json

ROOT = Path.cwd()
HERE = ROOT / "experiments/qad_v02"
V02 = "results/teacher-seed-sdpa-avg012-01"
TEMPLATE = "data/gleipnir4b-v02-ood-optimized01/config.yaml"
BASELINE_ID = "gleipnir4b-v02-id-optimized01"
QAD_EVAL = "qad-v02-optimized01"
# The QAD stages were validated with the NC2 runtime's FA4 overlay, which
# bundles quack 0.6.5 for CUTLASS DSL 4.8; the EU serving overlay behind
# .cache/kernels/fa4 lacks it, so the venv's older quack fails to import.
TRAINING_FA4 = "/tmp/gleipnir-training-runtime/fa4"
# The EU network volume is at its quota; the reconstructable campaign-layout
# copy of the master lives on container disk, linked from the results tree.
PACKAGE_ROOT = Path("/root/qad-v02-package")


def training_stage_environment(environment: dict[str, str]) -> dict[str, str]:
    paths = environment["PYTHONPATH"].split(":")
    matches = [i for i, p in enumerate(paths) if p.endswith(".cache/kernels/fa4")]
    if len(matches) != 1:
        raise ValueError("training FA4 overlay layout changed")
    paths[matches[0]] = TRAINING_FA4
    return {**environment, "PYTHONPATH": ":".join(paths)}


def scoring_context(name: str, parent: str, splits: list[str]) -> None:
    """An optimized scoring context like the v0.2 OOD one, for new splits."""
    data = ROOT / "data" / name
    if data.exists():
        raise ValueError(f"refuse to overwrite {data}")
    config = copy.deepcopy(yaml.safe_load((ROOT / TEMPLATE).read_text()))
    config.update(
        campaign_id=name,
        merged_model=f"/tmp/gleipnir-merged/{name}",
        parent_campaign=parent,
        score_splits=splits,
        extra_sources=sorted(
            {
                *config["extra_sources"],
                *(str(p.relative_to(ROOT)) for p in HERE.glob("*.py")),
            }
        ),
    )
    config["evaluations"] = [s for s in ("id",) if s in splits]
    config["evaluation"]["populations"] = {
        s: {"id": 3012, "ood": 6395}[s] for s in splits
    }
    data.mkdir(parents=True)
    path = data / "config.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False))
    ctx = Campaign(ROOT, path, yaml.safe_load(path.read_text()))
    write_json(
        data / "manifest.json",
        {
            "files_sha256": {
                str(path.relative_to(ROOT)): file_hash(path),
                **{s["path"]: s["sha256"] for s in ctx.config["inputs"].values()},
            },
            "source_sha256": ctx.sources(),
            "parent": parent,
        },
    )


def package(output: Path) -> None:
    """Lay the QAD master out as a campaign adapter for the shared evaluator."""
    complete = json.loads((output / "training_complete.json").read_text())
    master = output / "causal_adapter"
    if file_hash(master / "adapter_model.safetensors") != complete["master_sha256"]:
        raise ValueError("completed QAD master drift")
    if (output / "4b").exists() or (output / "4b").is_symlink():
        raise ValueError("refuse to overwrite an existing campaign-layout package")
    adapter = PACKAGE_ROOT / "4b/monitor"
    shutil.copytree(master, adapter / "causal_adapter")
    (output / "4b").symlink_to(PACKAGE_ROOT / "4b")
    rebase_adapter(adapter / "causal_adapter", adapter / "model")
    reference = json.loads((output / "parity_reference.json").read_text())
    if reference["master_sha256"] != complete["master_sha256"]:
        raise ValueError("QAD master reference identity drift")
    shutil.copy2(output / "parity_reference.json", adapter / "parity_reference.json")
    canary = read_rows(ROOT / V02 / "canary_workload.jsonl")
    if [r["prompt_sha256"] for r in canary] != reference["prompt_sha256"] or any(
        digest(r["prompt"]) != r["prompt_sha256"] for r in canary
    ):
        raise ValueError("v0.2 canary prompts differ from the QAD master reference")
    shutil.copy2(ROOT / V02 / "canary_workload.jsonl", output / "canary_workload.jsonl")
    write_json(
        adapter / "complete.json",
        {
            "status": "qad",
            "steps": complete["updates"],
            "epochs": 1.0,
            "master_sha256": complete["master_sha256"],
            "serving_sha256": file_hash(adapter / "model/adapter_model.safetensors"),
            "initial_master_sha256": load_config(ROOT)["initial_adapter_sha256"],
            "teacher_sha256": complete["teacher_sha256"],
        },
    )


def resume_from_package() -> None:
    """Repackage and evaluate after a driver lost during packaging.

    Training and the FP32 reference must be complete and hash-consistent; the
    incomplete package is kept aside, never overwritten.
    """
    config = load_config(ROOT)
    output = ROOT / "results" / config["experiment"]
    previous = json.loads((output / "status.json").read_text())
    if previous["phase"] != "package":
        raise ValueError(f"resume expects a lost package phase, not {previous}")
    complete = json.loads((output / "training_complete.json").read_text())
    reference = json.loads((output / "parity_reference.json").read_text())
    master = output / "causal_adapter/adapter_model.safetensors"
    if (
        complete["updates"] != config["updates"]
        or file_hash(master) != complete["master_sha256"]
        or reference["master_sha256"] != complete["master_sha256"]
    ):
        raise ValueError("training/reference receipts are not complete and consistent")
    # The lost package is a reconstructable duplicate of the master (copy plus
    # partial rebase, empty manifest) that filled the volume quota: record and
    # remove it rather than keep a second 4 GB copy.
    incomplete = output / "4b"
    removed = []
    if incomplete.exists() and not incomplete.is_symlink():
        manifest = incomplete / "monitor/model/rebase_manifest.json"
        if manifest.exists() and manifest.stat().st_size:
            raise ValueError("existing package looks complete; inspect before resuming")
        removed = [
            {"path": str(p.relative_to(ROOT)), "bytes": p.stat().st_size}
            for p in sorted(incomplete.rglob("*"))
            if p.is_file()
        ]
        shutil.rmtree(incomplete)
    write_json(
        output / "resume_from_package.json",
        {
            "previous_status": previous,
            "cause": "network volume quota exceeded during packaging",
            "removed_incomplete_package": removed,
            "resumed_at_unix": time.time(),
        },
    )

    def status(phase, **fields):
        write_json(
            output / "status.json", {"phase": phase, "at_unix": time.time(), **fields}
        )
        print("qad_v02", phase, json.dumps(fields), flush=True)

    try:
        status("package", resumed=True)
        package(output)
        status("evaluate")
        scoring_context(QAD_EVAL, f"results/{config['experiment']}", ["id", "ood"])
        score(QAD_EVAL)
        status("complete")
    except BaseException as error:
        status("failed", error=f"{type(error).__name__}: {error}")
        raise


def main():
    config = load_config(ROOT)
    output = ROOT / "results" / config["experiment"]
    output.mkdir(parents=True, exist_ok=True)
    if (output / "execution_contract.json").exists():
        raise ValueError("preserve existing campaign receipts")
    status_path = output / "status.json"
    gpu_uuid = config["gpu_uuid"]

    def status(phase, **fields):
        write_json(status_path, {"phase": phase, "at_unix": time.time(), **fields})
        print("qad_v02", phase, json.dumps(fields), flush=True)

    def run(name, executable, environment, *module):
        status(name)
        subprocess.run(
            [executable, *module],
            cwd=ROOT,
            env={
                **environment,
                "PYTHONPATH": f"{ROOT}:{environment.get('PYTHONPATH', '')}",
            },
            check=True,
            timeout=4 * 3600,
        )

    try:
        status("preparing")
        check(config)
        from experiments.b200_vllm031.runtime import candidate_environment

        status("baseline")
        if not (ROOT / "results" / BASELINE_ID / "evaluation/complete.json").exists():
            scoring_context(BASELINE_ID, V02, ["id"])
            score(BASELINE_ID)
        run(
            "teacher",
            config["serving_python"],
            candidate_environment(ROOT),
            "-m",
            "experiments.qad_v02.teacher",
        )
        stages = generate(
            ROOT,
            output,
            config_path="experiments/qad_v02/config.yaml",
            runtime_module="experiments.qad_v02.runtime",
            native_attention_backward=False,
        )
        sources = [
            *stages.values(),
            HERE / "config.yaml",
            *sorted(HERE.glob("*.py")),
            *sorted((ROOT / "experiments/native_fp8_qat").glob("*.py")),
            ROOT / "src/gleipnir/training/qat_reference.py",
            ROOT / "src/gleipnir/training/qat_optimization.py",
            ROOT / "src/gleipnir/training/quantization_recovery.py",
            ROOT / "src/gleipnir/kernels/fp4/decode.py",
        ]
        bindings = {str(p.relative_to(ROOT)): file_hash(p) for p in sources}
        teacher = ROOT / "data" / config["teacher_experiment"]
        write_json(
            output / "execution_contract.json",
            {
                "config": config,
                "sources": bindings,
                "teacher_sha256": file_hash(teacher / "teacher.jsonl"),
                "workload_sha256": file_hash(teacher / "workload.jsonl"),
                "gpu_uuid": gpu_uuid,
                "training_fa4_overlay": TRAINING_FA4,
            },
        )
        for source in sources:
            target = output / "source_archive" / source.relative_to(ROOT)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
        environment = training_stage_environment(
            training_environment(ROOT, config["experiment"], native_fp4_mlp=True)
        )
        for name in ("train", "reference"):
            for source, checksum in bindings.items():
                if file_hash(ROOT / source) != checksum:
                    raise ValueError(f"campaign source drift: {source}")
            run(name, config["training_python"], environment, str(stages[name]))
        status("package")
        package(output)
        status("evaluate")
        scoring_context(QAD_EVAL, f"results/{config['experiment']}", ["id", "ood"])
        score(QAD_EVAL)
        status("complete")
    except BaseException as error:
        status("failed", error=f"{type(error).__name__}: {error}")
        raise


if __name__ == "__main__":
    import sys

    resume_from_package() if sys.argv[1:] == ["--resume-package"] else main()
