"""Archive narrowly checked direct-target derivatives of frozen QAD stages."""

from pathlib import Path

import yaml

from experiments.native_fp8_qat.generate import transform as reference_transform
from gleipnir.data.monitoring import file_hash, write_json


def transform(
    source: str,
    stage: str,
    *,
    initial_alignment_diagnostic_only: bool = False,
    **variant,
):
    source, receipts = reference_transform(source, stage, **variant)
    if stage == "train":
        start = (
            '        receipt = json.loads((directory / "teacher_complete.json")'
            ".read_text())"
        )
        end = "        params = [p for p in model.parameters() if p.requires_grad]"
        if source.count(start) != 1 or source.count(end) != 1:
            raise ValueError("teacher loading template drift")
        old = source[source.index(start) : source.index(end)]
        new = (
            "        from experiments.kimi_direct_qat.runtime import load_teachers\n"
            "        teachers, receipt = load_teachers(workload, "
            "root, output, config)\n"
        )
        replacements = [
            (old, new),
            (
                'root / "results/filtered-fp4-fa4-optimized-id01/canary_repeat1.json"',
                'root / config["initial_server_canary"]',
            ),
            (
                'reference["base"], config["alignment"]',
                'reference["base"], config["initial_alignment"]',
            ),
        ]
        if initial_alignment_diagnostic_only:
            replacements.append(
                (
                    '        if not gate["passed"]:\n'
                    '            raise ValueError("QAD simulator/server '
                    'score alignment failed")',
                    '        if not gate["passed"]:\n'
                    '            print("untrained_base_alignment_diagnostic_failed", '
                    "json.dumps(gate), flush=True)",
                )
            )
        for old, new in replacements:
            count = source.count(old)
            expected = 2 if old.startswith('root / "results/filtered') else 1
            if count != expected:
                raise ValueError("direct Kimi substitution count drift")
            source = source.replace(old, new)
            receipts.append({"old": old, "new": new, "count": count})
    compile(source, f"kimi_direct_qat/{stage}.py", "exec")
    return source, receipts


def generate(root: Path, output: Path, **variant):
    config = yaml.safe_load((root / variant["config_path"]).read_text())
    directory = output / "stage_sources"
    directory.mkdir(parents=True, exist_ok=True)
    stages, receipts = {}, {}
    for stage in ("train", "reference", "evaluate"):
        original = root / f"experiments/quantization_recovery/{stage}.py"
        generated, changes = transform(
            original.read_text(),
            stage,
            initial_alignment_diagnostic_only=config.get(
                "initial_alignment_diagnostic_only", False
            ),
            **variant,
        )
        target = directory / f"{stage}.py"
        (directory / f"original_{stage}.py").write_text(original.read_text())
        target.write_text(generated)
        stages[stage] = target
        receipts[stage] = {
            "original_sha256": file_hash(original),
            "generated_sha256": file_hash(target),
            "replacements": changes,
        }
    if not config.get("reuse_initial_server_canary", False):
        stages["initial-serving"] = (
            root / "experiments/kimi_direct_qat/initial_serving.py"
        )
    write_json(output / "stage_generation.json", receipts)
    return stages
