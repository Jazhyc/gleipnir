"""Preserve frozen QAD arithmetic while changing subset, order and schedule."""

from pathlib import Path

from experiments.native_fp8_qat.generate import transform as base_transform
from gleipnir.data.monitoring import file_hash, write_json


def transform(source: str, stage: str, **variant):
    source, changes = base_transform(source, stage, **variant)
    replacements = []
    if stage == "train":
        start = source.index("    reference = json.loads(")
        end = source.index(
            '        receipt = json.loads((directory / "teacher_complete.json")'
            ".read_text())"
        )
        logits = source[
            source.index("    def logits(rows):") : source.index(
                "    with attention_runtime():"
            )
        ]
        replacements = [
            (
                source[start:end],
                logits + "    with attention_runtime():\n        model.train()\n",
            ),
            (
                "order = torch.randperm(len(workload), generator=generator).tolist()",
                "order = list(range(len(workload)))",
            ),
        ]
    else:
        replacements = [
            (
                'root / "data" / config["teacher_experiment"]',
                'root / "data" / config["reference_workload_experiment"]',
            )
        ]
    for old, new in replacements:
        if source.count(old) != 1:
            raise ValueError("short QAD template drift")
        source = source.replace(old, new)
        changes.append({"old": old, "new": new, "count": 1})
    compile(source, f"qad_budget/{stage}.py", "exec")
    return source, changes


def generate(root: Path, output: Path, **variant):
    folder = output / "stage_sources"
    folder.mkdir(parents=True, exist_ok=True)
    stages, receipts = {}, {}
    for stage in ("train", "reference", "evaluate"):
        original = root / f"experiments/quantization_recovery/{stage}.py"
        source, replacements = transform(original.read_text(), stage, **variant)
        target = folder / f"{stage}.py"
        target.write_text(source)
        (folder / f"original_{stage}.py").write_text(original.read_text())
        stages[stage] = target
        receipts[stage] = {
            "original_sha256": file_hash(original),
            "generated_sha256": file_hash(target),
            "replacements": replacements,
        }
    write_json(output / "stage_generation.json", receipts)
    return stages
