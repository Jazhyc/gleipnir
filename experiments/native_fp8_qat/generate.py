"""Archive checked derivatives of the frozen QAD stage entrypoints."""

from pathlib import Path

from gleipnir.data.monitoring import file_hash, write_json


def transform(
    source: str,
    stage: str,
    *,
    config_path: str = "experiments/native_fp8_qat/config.yaml",
    runtime_module: str = "experiments.native_fp8_qat.runtime",
    native_attention_backward: bool = False,
) -> tuple[str, list[dict]]:
    """Change only the named config/cache and explicit training-runtime hooks."""
    replacements = [
        (
            "experiments/quantization_recovery/config.yaml",
            config_path,
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
                    f"from {runtime_module} import attention_runtime, install",
                ),
                (
                    "from experiments.quantization_recovery.kernel_canary "
                    "import main as decoder_canary",
                    f"from {runtime_module} import verify_operator_reference",
                ),
                ("decoder_canary()", "verify_operator_reference(root, output)"),
            ]
        )
        if native_attention_backward:
            replacements.extend(
                [
                    (
                        "        model.train()\n"
                        "        model.zero_grad(set_to_none=True)\n        longest =",
                        f"        from {runtime_module} "
                        "import validate_model_backward\n"
                        "        validate_model_backward(model, logits, "
                        "short, output)\n"
                        "        model.train()\n"
                        "        model.zero_grad(set_to_none=True)\n        longest =",
                    ),
                    (
                        '"backward": "approximate_low_rank_STE_and_quantized_QKV_FA4",',
                        '"backward": "approximate_low_rank_STE_and_native_MXFP8",',
                    ),
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


def generate(root: Path, output: Path, **variant) -> dict[str, Path]:
    """Preserve exact input/output source bytes and replacement receipts."""
    directory = output / "stage_sources"
    directory.mkdir(parents=True, exist_ok=True)
    stages, receipts = {}, {}
    for stage in ("train", "reference", "evaluate"):
        original = root / f"experiments/quantization_recovery/{stage}.py"
        generated, changes = transform(original.read_text(), stage, **variant)
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
