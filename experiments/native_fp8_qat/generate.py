"""Archive checked derivatives of the frozen QAD stage entrypoints."""

from pathlib import Path

from gleipnir.data.monitoring import file_hash, write_json


def transform(source: str, stage: str) -> tuple[str, list[dict]]:
    """Change only the named config/cache and explicit training-runtime hooks."""
    replacements = [
        (
            "experiments/quantization_recovery/config.yaml",
            "experiments/native_fp8_qat/config.yaml",
        ),
        (
            'root / "data" / config["experiment"]',
            'root / "data" / config["teacher_experiment"]',
        ),
    ]
    if stage == "train":
        replacements.extend(
            [
                (
                    "from gleipnir.training.quantization_recovery "
                    "import attention_runtime, install",
                    "from experiments.native_fp8_qat.runtime "
                    "import attention_runtime, install",
                ),
                (
                    "from experiments.quantization_recovery.kernel_canary "
                    "import main as decoder_canary",
                    "from experiments.native_fp8_qat.runtime "
                    "import verify_operator_reference",
                ),
                ("decoder_canary()", "verify_operator_reference(root, output)"),
            ]
        )
    elif stage not in {"reference", "evaluate"}:
        raise ValueError("unknown frozen QAD stage")
    receipts = []
    for old, new in replacements:
        if source.count(old) != 1:
            raise ValueError(f"template substitution count drift: {old}")
        source = source.replace(old, new)
        receipts.append({"old": old, "new": new, "count": 1})
    compile(source, f"native_fp8_qat/{stage}.py", "exec")
    return source, receipts


def generate(root: Path, output: Path) -> dict[str, Path]:
    """Preserve exact input/output source bytes and replacement receipts."""
    directory = output / "stage_sources"
    directory.mkdir(parents=True, exist_ok=True)
    stages, receipts = {}, {}
    for stage in ("train", "reference", "evaluate"):
        original = root / f"experiments/quantization_recovery/{stage}.py"
        generated, changes = transform(original.read_text(), stage)
        archived = directory / f"original_{stage}.py"
        target = directory / f"{stage}.py"
        archived.write_text(original.read_text())
        target.write_text(generated)
        stages[stage] = target
        receipts[stage] = {
            "original_sha256": file_hash(original),
            "generated_sha256": file_hash(target),
            "replacements": changes,
        }
    write_json(output / "stage_generation.json", receipts)
    return stages
